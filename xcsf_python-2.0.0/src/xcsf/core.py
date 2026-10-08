"""Supervised XCSF engine, separated from the scikit-learn interface.

For regression the match set is the action set of the single dummy action.
The target is supplied externally (Lanzi & Loiacono, 2025, Algorithm 7).
"""

from collections import deque
from types import SimpleNamespace

import numpy as np

from .conditions import IntervalCondition
from .prediction import LocalPredictor
from .rule import Classifier


def probabilities(weights):
    """Normalize nonnegative weights without overflowing their sum."""
    weights = np.asarray(weights, dtype=float)
    if np.any(np.isposinf(weights)):
        weights = np.isposinf(weights).astype(float)
    maximum = weights.max()
    if maximum <= 0:
        return np.full(len(weights), 1 / len(weights))
    weights = weights / maximum
    return weights / weights.sum()


class XCSFCore:
    """Population lifecycle and online learning; no dependency on an environment."""

    def __init__(self, parameters, rng, bounds=None):
        self.config = SimpleNamespace(**parameters)
        self.rng = rng
        self.bounds = bounds
        self.population = []
        self.time = 0
        self._next_id = 0
        self._bounds_cache = None
        self.stats = dict(coverings=0, ga_runs=0, crossovers=0, subsumptions=0,
                          deletions=0, condensation_runs=0)

    @property
    def numerosity(self):
        return sum(cl.numerosity for cl in self.population)

    def _identifier(self):
        identifier = self._next_id
        self._next_id += 1
        return identifier

    def condition_arrays(self):
        if self._bounds_cache is None:
            self._bounds_cache = (
                np.array([cl.condition.lower for cl in self.population]),
                np.array([cl.condition.upper for cl in self.population]),
            )
        return self._bounds_cache

    def match(self, x):
        if not self.population:
            return []
        lower, upper = self.condition_arrays()
        indices = np.flatnonzero(np.all((x >= lower) & (x <= upper), axis=1))
        return [self.population[i] for i in indices]

    def _cover(self, x, phi):
        c = self.config
        cl = Classifier(
            IntervalCondition.cover(x, c.cover_radius, self.rng, self.bounds),
            LocalPredictor(
                len(phi), method=c.prediction, learning_rate=c.prediction_learning_rate,
                delta=c.rls_delta, forgetting_factor=c.forgetting_factor,
                process_noise=c.process_noise, kalman_noise=c.kalman_noise,
                initial_prediction=c.initial_prediction, x0=c.x0,
                lasso_alpha=c.lasso_alpha, lasso_window=c.lasso_window,
                lasso_max_iter=c.lasso_max_iter, lasso_tol=c.lasso_tol,
                lasso_learning_rate_decay=c.lasso_learning_rate_decay,
            ),
            identifier=self._identifier(), error=c.initial_error,
            squared_error=c.initial_error ** 2, fitness=c.initial_fitness,
            timestamp=self.time, created_at=self.time,
            match_history=deque(maxlen=c.niche_history),
        )
        # Make room BEFORE insertion: even a population of size 1 then covers
        # the current sample without a potentially unbounded delete/cover loop.
        self._delete_to_size(c.population_size - 1)
        self._insert(cl)
        self.stats["coverings"] += 1
        return [cl]

    def _insert(self, child):
        for cl in self.population:
            if cl.condition.same_as(child.condition):
                cl.numerosity += child.numerosity
                return
        self.population.append(child)
        self._bounds_cache = None

    def update(self, x, target, phi, *, condensation=False):
        self.time += 1
        c = self.config
        niche = self.match(x)
        if not niche:
            niche = self._cover(x, phi)
        prediction = self.aggregate(niche, phi)
        size = sum(cl.numerosity for cl in niche)
        for cl in niche:
            cl.experience += 1
            cl.last_match = self.time
            cl.match_history.append(self.time)
            rate = max(c.learning_rate, 1 / cl.experience) if c.use_mam else c.learning_rate
            residual = target - cl.predictor.predict(phi)
            if c.error_before_prediction:
                cl.error += rate * (abs(residual) - cl.error)
                cl.squared_error += rate * (residual ** 2 - cl.squared_error)
            cl.predictor.update(phi, target, cl.squared_error)
            if not c.error_before_prediction:
                residual = target - cl.predictor.predict(phi)
                cl.error += rate * (abs(residual) - cl.error)
                cl.squared_error += rate * (residual ** 2 - cl.squared_error)
            cl.set_size += rate * (size - cl.set_size)
        self._update_fitness(niche)
        if c.match_subsumption:
            niche = self._subsume_match_set(niche)
        age = self.time - np.average(
            [cl.timestamp for cl in niche], weights=[cl.numerosity for cl in niche],
        )
        if (c.discovery or condensation) and age >= c.theta_ga:
            self._evolve(niche, condensation=condensation)
        return float(prediction)

    def _update_fitness(self, niche):
        c = self.config
        # Log space preserves relative accuracy for large errors / small epsilon.
        log_accuracy = np.array([
            0.0 if cl.error < c.epsilon_0
            else np.log(c.alpha) - c.nu * (np.log(cl.error) - np.log(c.epsilon_0))
            for cl in niche
        ])
        log_accuracy += np.log([cl.numerosity for cl in niche])
        accuracy = np.exp(log_accuracy - log_accuracy.max())
        accuracy /= accuracy.sum()
        for cl, relative in zip(niche, accuracy):
            cl.fitness += c.learning_rate * (relative - cl.fitness)

    @staticmethod
    def aggregate(niche, phi):
        weights = probabilities([cl.fitness for cl in niche])
        return np.dot(weights, [cl.predictor.predict(phi) for cl in niche])

    def _select(self, niche):
        c = self.config
        if c.selection == "roulette":
            p = probabilities([cl.fitness for cl in niche])
            return niche[self.rng.choice(len(niche), p=p)]
        # Each microclassifier participates independently. Sampling directly
        # conditional on a nonempty tournament avoids unbounded rejection loops.
        if c.tournament_fraction == 1:
            candidates = niche
        else:
            log_fail = np.log1p(-c.tournament_fraction)
            remaining = sum(cl.numerosity for cl in niche)
            candidates = []
            for cl in niche:
                participation = -np.expm1(cl.numerosity * log_fail)
                if not candidates:
                    participation /= -np.expm1(remaining * log_fail)
                if self.rng.random_sample() < participation:
                    candidates.append(cl)
                remaining -= cl.numerosity
        return max(candidates, key=lambda cl: cl.fitness / cl.numerosity)

    def _subsumer(self, child, candidates, threshold):
        c = self.config
        return next((cl for cl in candidates
                     if cl.can_subsume(c.epsilon_0, threshold)
                     and cl.condition.contains(child.condition)), None)

    def _subsume_match_set(self, niche):
        c = self.config
        best = None
        for cl in niche:
            if cl.can_subsume(c.epsilon_0, c.theta_match_subsume):
                if best is None or cl.condition.contains(best.condition):
                    best = cl
        if best is None:
            return niche
        victims = [cl for cl in niche if cl is not best and best.condition.contains(cl.condition)]
        for cl in victims:
            best.numerosity += cl.numerosity
            self.population.remove(cl)
            niche.remove(cl)
            self.stats["subsumptions"] += 1
        self._bounds_cache = None
        return niche

    def _evolve(self, niche, *, condensation):
        c = self.config
        for cl in niche:
            cl.timestamp = self.time
        parents = [self._select(niche), self._select(niche)]
        if condensation:
            self.stats["condensation_runs"] += 1
            for parent in parents:
                subsumer = self._subsumer(parent, niche, c.theta_subsume) if c.ga_subsumption else None
                (subsumer or parent).numerosity += 1
            self._delete_to_size(c.population_size)
            return
        self.stats["ga_runs"] += 1
        children = [cl.offspring(self._identifier(), self.time, c.niche_history) for cl in parents]
        if self.rng.random_sample() < c.crossover_probability:
            children[0].condition.crossover(children[1].condition, c.crossover, self.rng, self.bounds)
            for name in ("error", "squared_error", "fitness", "set_size"):
                average = (getattr(parents[0], name) + getattr(parents[1], name)) / 2
                for child in children:
                    setattr(child, name, average)
            self.stats["crossovers"] += 1
        for child in children:
            child.condition.mutate(c.mutation_probability, c.mutation_scale,
                                   c.mutation, self.rng, self.bounds)
            child.fitness *= 0.1
            subsumer = self._subsumer(child, parents + niche, c.theta_subsume) if c.ga_subsumption else None
            if subsumer is not None:
                subsumer.numerosity += 1
                self.stats["subsumptions"] += 1
            else:
                self._insert(child)
        self._delete_to_size(c.population_size)

    def deletion_votes(self):
        """XCS deletion vote: niche size * numerosity, penalizing weak old rules."""
        c = self.config
        average = sum(cl.fitness for cl in self.population) / self.numerosity
        votes = []
        for cl in self.population:
            vote = cl.set_size * cl.numerosity
            per_micro = cl.fitness / cl.numerosity
            if cl.experience > c.theta_delete and per_micro < c.delta * average:
                vote *= average / max(per_micro, np.finfo(float).tiny)
            votes.append(vote)
        return np.asarray(votes)

    def _delete_to_size(self, limit):
        size = self.numerosity
        while size > limit:
            i = self.rng.choice(len(self.population), p=probabilities(self.deletion_votes()))
            cl = self.population[i]
            cl.numerosity -= 1
            if cl.numerosity == 0:
                self.population.pop(i)
                self._bounds_cache = None
            size -= 1
            self.stats["deletions"] += 1

