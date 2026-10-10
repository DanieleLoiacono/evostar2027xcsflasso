"""XCSF classifier system (cf. xcslib ``xcsf/xcsf_classifier_system``).

The system owns the population and its lifecycle: matching, covering, the
update of the action set, the genetic algorithm, subsumption and deletion. It
knows neither how conditions are represented, nor how predictors learn, nor
where problems come from: those are the condition representation, the
predictor factory and the environment (Lanzi & Loiacono, 2025, Algorithm 7).
"""

from collections import deque
from types import SimpleNamespace

import numpy as np

from .actions import DummyAction
from .classifier import Classifier
from .conditions import RealIntervalRepresentation
from .prediction import PredictorFactory, design_matrix


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


class XCSFClassifierSystem:
    """Population lifecycle and online learning for single-step function approximation.

    ``parameters`` holds the XCSFRegressor parameters. ``representation`` and
    ``predictors`` default to interval conditions and the predictor selected by
    ``parameters["prediction"]``, both configured from ``parameters``.
    """

    def __init__(self, parameters, rng, *, bounds=None, representation=None, predictors=None,
                 action=None):
        self.config = c = SimpleNamespace(**parameters)
        self.rng = rng
        self.representation = representation or RealIntervalRepresentation(
            c.cover_radius, c.mutation_scale, c.mutation, c.crossover, bounds)
        self.predictors = predictors or PredictorFactory(c.prediction, parameters)
        self.action = DummyAction() if action is None else action
        self.population = []
        self.time = 0
        self._next_id = 0
        self._matcher = None
        self.stats = dict(coverings=0, ga_runs=0, crossovers=0, subsumptions=0,
                          deletions=0, condensation_runs=0)

    # ------------------------------------------------------------------ population

    @property
    def numerosity(self):
        return sum(cl.numerosity for cl in self.population)

    def _identifier(self):
        identifier = self._next_id
        self._next_id += 1
        return identifier

    def matcher(self):
        """Match queries over the current population, cached until it changes."""
        if self._matcher is None:
            self._matcher = self.representation.matcher(
                [cl.condition for cl in self.population])
        return self._matcher

    def match_set(self, x):
        """[M]: the classifiers whose condition matches x."""
        if not self.population:
            return []
        return [self.population[i] for i in np.flatnonzero(self.matcher().matching(x))]

    def _cover(self, x, size):
        c = self.config
        cl = Classifier(
            self.representation.cover(x, self.rng), self.predictors(size), self.action,
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
            if cl.same_rule_as(child):
                cl.numerosity += child.numerosity
                return
        self.population.append(child)
        self._matcher = None

    # ------------------------------------------------------------------------ step

    def step(self, x, target, phi=None, *, condensation=False):
        """One learning problem: return the prediction made before learning from it.

        With the single implicit action of function approximation the action
        set [A] is the whole match set [M], and the reward is the target.
        """
        self.time += 1
        c = self.config
        if phi is None:
            phi = design_matrix(x, c.degree, c.x0)
        action_set = self.match_set(x)
        if not action_set:
            action_set = self._cover(x, len(phi))
        predictions = [cl.predictor.predict(phi) for cl in action_set]
        prediction = self.system_prediction(action_set, phi, predictions)
        self._update_set(action_set, phi, target, predictions)
        self._update_fitness(action_set)
        if c.match_subsumption:
            action_set = self._subsume_action_set(action_set)
        age = self.time - np.average(
            [cl.timestamp for cl in action_set], weights=[cl.numerosity for cl in action_set],
        )
        if (c.discovery or condensation) and age >= c.theta_ga:
            self._evolve(action_set, condensation=condensation)
        return float(prediction)

    @staticmethod
    def system_prediction(action_set, phi, predictions=None):
        """Fitness-weighted average of the classifier predictions."""
        if predictions is None:
            predictions = [cl.predictor.predict(phi) for cl in action_set]
        return np.dot(probabilities([cl.fitness for cl in action_set]), predictions)

    def _update_set(self, action_set, phi, target, predictions):
        """Update experience, error, predictor and niche size of every classifier in [A]."""
        c = self.config
        size = sum(cl.numerosity for cl in action_set)
        for cl, predicted in zip(action_set, predictions):
            cl.experience += 1
            cl.last_match = self.time
            cl.match_history.append(self.time)
            rate = max(c.learning_rate, 1 / cl.experience) if c.use_mam else c.learning_rate
            residual = target - predicted
            if c.error_before_prediction:
                cl.error += rate * (abs(residual) - cl.error)
                cl.squared_error += rate * (residual ** 2 - cl.squared_error)
            cl.predictor.update(phi, target, cl.squared_error)
            if not c.error_before_prediction:
                residual = target - cl.predictor.predict(phi)
                cl.error += rate * (abs(residual) - cl.error)
                cl.squared_error += rate * (residual ** 2 - cl.squared_error)
            cl.set_size += rate * (size - cl.set_size)

    def _update_fitness(self, action_set):
        c = self.config
        # Log space preserves relative accuracy for large errors / small epsilon.
        log_accuracy = np.array([
            0.0 if cl.error < c.epsilon_0
            else np.log(c.alpha) - c.nu * (np.log(cl.error) - np.log(c.epsilon_0))
            for cl in action_set
        ])
        log_accuracy += np.log([cl.numerosity for cl in action_set])
        accuracy = np.exp(log_accuracy - log_accuracy.max())
        accuracy /= accuracy.sum()
        for cl, relative in zip(action_set, accuracy):
            cl.fitness += c.learning_rate * (relative - cl.fitness)

    # ----------------------------------------------------------------- subsumption

    def _subsumer(self, child, candidates, threshold):
        c = self.config
        return next((cl for cl in candidates
                     if cl.can_subsume(c.epsilon_0, threshold)
                     and cl.is_more_general_than(child)), None)

    def _subsume_action_set(self, action_set):
        c = self.config
        best = None
        for cl in action_set:
            if cl.can_subsume(c.epsilon_0, c.theta_match_subsume):
                if best is None or cl.is_more_general_than(best):
                    best = cl
        if best is None:
            return action_set
        victims = [cl for cl in action_set if cl is not best and best.is_more_general_than(cl)]
        for cl in victims:
            best.numerosity += cl.numerosity
            self.population.remove(cl)
            action_set.remove(cl)
            self.stats["subsumptions"] += 1
        self._matcher = None
        return action_set

    # ----------------------------------------------------------- genetic algorithm

    def _select(self, action_set):
        c = self.config
        if c.selection == "roulette":
            p = probabilities([cl.fitness for cl in action_set])
            return action_set[self.rng.choice(len(action_set), p=p)]
        # Each microclassifier participates independently. Sampling directly
        # conditional on a nonempty tournament avoids unbounded rejection loops.
        if c.tournament_fraction == 1:
            candidates = action_set
        else:
            log_fail = np.log1p(-c.tournament_fraction)
            remaining = sum(cl.numerosity for cl in action_set)
            candidates = []
            for cl in action_set:
                participation = -np.expm1(cl.numerosity * log_fail)
                if not candidates:
                    participation /= -np.expm1(remaining * log_fail)
                if self.rng.random_sample() < participation:
                    candidates.append(cl)
                remaining -= cl.numerosity
        return max(candidates, key=lambda cl: cl.fitness / cl.numerosity)

    def _evolve(self, action_set, *, condensation):
        c = self.config
        for cl in action_set:
            cl.timestamp = self.time
        parents = [self._select(action_set), self._select(action_set)]
        if condensation:
            self.stats["condensation_runs"] += 1
            for parent in parents:
                subsumer = (self._subsumer(parent, action_set, c.theta_subsume)
                            if c.ga_subsumption else None)
                (subsumer or parent).numerosity += 1
            self._delete_to_size(c.population_size)
            return
        self.stats["ga_runs"] += 1
        children = [cl.offspring(self._identifier(), self.time, c.niche_history) for cl in parents]
        if self.rng.random_sample() < c.crossover_probability:
            self.representation.crossover(children[0].condition, children[1].condition, self.rng)
            for name in ("error", "squared_error", "fitness", "set_size"):
                average = (getattr(parents[0], name) + getattr(parents[1], name)) / 2
                for child in children:
                    setattr(child, name, average)
            self.stats["crossovers"] += 1
        for child in children:
            self.representation.mutate(child.condition, c.mutation_probability, self.rng)
            child.fitness *= 0.1
            subsumer = (self._subsumer(child, parents + action_set, c.theta_subsume)
                        if c.ga_subsumption else None)
            if subsumer is not None:
                subsumer.numerosity += 1
                self.stats["subsumptions"] += 1
            else:
                self._insert(child)
        self._delete_to_size(c.population_size)

    # -------------------------------------------------------------------- deletion

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
                self._matcher = None
            size -= 1
            self.stats["deletions"] += 1
