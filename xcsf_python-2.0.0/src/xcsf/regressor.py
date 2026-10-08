"""scikit-learn estimator for scalar XCSF regression."""

from copy import deepcopy
from numbers import Integral, Real

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from .core import XCSFCore, probabilities
from .prediction import PREDICTION_METHODS, design_matrix


class XCSFRegressor(RegressorMixin, BaseEstimator):
    """Evolve a population of local regressors using accuracy-based XCSF.

    Parameters
    ----------
    population_size : int, default=400
        Maximum number of microclassifiers (sum of rule numerosities).
    n_epochs : int, default=20
        Passes over the data in fit. partial_fit always performs one pass.
    prediction : str, default='nlms'
        One of 'lms', 'nlms', 'rls', 'lasso_online', 'lasso_batch', 'rlsk',
        'constant'. RLS uses QR updates. Lasso uses proximal online updates
        or coordinate descent on retained local samples, respectively.
    degree : int, default=1
        Per-feature polynomial powers, without interaction terms.
    learning_rate : float in (0, 1], default=0.2
        Beta: learning rate for error, niche size, and relative accuracy fitness.
    prediction_learning_rate : float in (0, 1], default=0.2
        Eta for LMS, NLMS, online Lasso and constant prediction. Ignored by
        RLS/RLSK and batch Lasso. Scale inputs when using LMS or online Lasso.
    epsilon_0 : float > 0, default=0.05
        Accuracy threshold, in target units. Targets are not normalized.
    alpha : float in (0, 1], default=0.1
        Accuracy multiplier for errors at least epsilon_0.
    nu : float > 0, default=5.0
        Exponent of the accuracy function.
    theta_ga : float >= 0, default=25
        Minimum numerosity-weighted average niche age to trigger the GA.
    crossover_probability : float in [0, 1], default=0.8
        Probability of crossing the two offspring conditions.
    mutation_probability : float in [0, 1], default=0.04
        Mutation probability per endpoint (per interval for proportional).
    cover_radius : float > 0, default=0.2
        Maximum independent extension on either side of a covered point.
    mutation_scale : float >= 0, default=0.2
        Maximum fixed displacement / Gaussian standard deviation.
    mutation : {'fixed', 'proportional', 'gaussian'}, default='fixed'
        Condition mutation operator.
    crossover : {'one_point', 'two_point', 'uniform'}, default='one_point'
        Condition recombination operator.
    theta_delete : float >= 0, default=20
        Experience threshold for fitness-penalized deletion.
    delta : float in (0, 1], default=0.1
        Fraction of mean microclassifier fitness below which deletion increases.
    theta_subsume : float >= 0, default=20
        Minimum experience (strictly exceeded) for GA subsumption.
    ga_subsumption : bool, default=True
        Let accurate parents or matching classifiers subsume contained offspring.
    match_subsumption : bool, default=False
        Apply action-set subsumption to the regression match set after updating.
    theta_match_subsume : float >= 0, default=100
        Experience threshold for match-set subsumption.
    selection : {'roulette', 'tournament'}, default='roulette'
        GA parent selection based on fitness.
    tournament_fraction : float in (0, 1], default=0.4
        Independent participation probability of each microclassifier.
    initial_fitness : float > 0, default=0.01
        Fitness assigned on covering.
    initial_error : float >= 0, default=0.0
        Absolute prediction error assigned on covering.
    initial_prediction : float, default=0.0
        Constant initial output of a newly covered predictor.
    use_mam : bool, default=True
        Use sample-average error/set-size updates until experience reaches 1/beta.
    error_before_prediction : bool, default=True
        Estimate error before, rather than after, the local prediction update.
    rls_delta : float > 0, default=1000.0
        Initial covariance multiplier for RLS/RLSK, unrelated to deletion delta.
    forgetting_factor : float in (0, 1], default=1.0
        RLS/RLSK forgetting factor. Values below one favor recent observations.
        In QR RLS it also discounts the initial prior.
    process_noise : float >= 0, default=0.0
        Q in the RLSK covariance update V <- V + Q*I.
    kalman_noise : bool, default=False
        Use max(rule squared error, 1e-4) as RLSK measurement variance.
    lasso_alpha : float >= 0, default=0.001
        L1 penalty on slopes (never the intercept), distinct from fitness alpha.
        Batch objective: mean squared residual / 2 + lasso_alpha * sum(abs(w)).
    lasso_window : int >= 1 or None, default=256
        Maximum observations retained per batch Lasso classifier. None retains
        all observations since birth. Repeated epochs count as new observations.
    lasso_max_iter : int >= 1, default=1000
        Maximum coordinate-descent sweeps per batch Lasso update. Failure to
        meet tolerance emits ConvergenceWarning once per classifier.
    lasso_tol : float > 0, default=1e-6
        Absolute tolerance on batch Lasso KKT violations, in gradient units.
    lasso_learning_rate_decay : float in [0, 1], default=0.0
        Online Lasso step at local update t (starting at 1):
        prediction_learning_rate / t**lasso_learning_rate_decay. Zero uses a
        constant step. This is a stochastic update, not a batch Lasso solution.
    x0 : float > 0, default=1.0
        Constant bias input for local predictors.
    normalize : bool, default=True
        Scale features using training minima/ranges, frozen on the first fit or
        partial_fit. No clipping; constant features use a range of one.
    bounded : bool, default=False
        Restrict conditions to the first training batch's feature bounds.
        Subsequent partial_fit batches must remain within these bounds.
    discovery : bool, default=True
        Enable genetic discovery. Covering and parameter learning always run.
    condensation_epochs : int, default=0
        Extra fit passes reproducing selected rules without crossover/mutation.
    niche_history : int, default=0
        Maximum recent matching timestamps stored per rule. Zero disables it.
    history_interval : int or None, default=100
        Record online error and population size every this many training steps.
        Error is averaged over consecutive, non-overlapping windows; population
        size is sampled at the window endpoint, after updating/evolution.
        A final incomplete window is also exposed and extended by further
        training. None disables step history; epoch history remains available.
    unmatched : {'nearest', 'mean', 'raise'}, default='nearest'
        For uncovered predictions, use nearest regions' local extrapolations,
        the observed target mean, or raise ValueError. Never cover at prediction.
    shuffle : bool, default=True
        Shuffle each fit epoch. partial_fit preserves the supplied order.
    random_state : int, RandomState or None, default=None
        Controls sample order and all evolutionary randomness.

    Attributes
    ----------
    population_ : list of Classifier
        Learned macroclassifiers, exposed for inspection. Treat as read-only.
    n_features_in_ : int
        Number of input features.
    feature_names_in_ : ndarray of str
        Input feature names, when all names are strings.
    n_iter_ : int
        Number of completed passes, including condensation/partial_fit passes.
    n_samples_seen_ : int
        Number of online updates (includes repeated epochs).
    n_microclassifiers_, n_macroclassifiers_ : int
        Learned population sizes.
    history_ : list of dict
        Online, pre-update MAE/MSE/RMSE and population size after each pass,
        including the cumulative step and the number of samples in that pass.
        This is training history, not a held-out validation metric.
    performance_history_ : list of dict
        Step-window records: step, n_samples, mae, mse, rmse, macroclassifiers,
        microclassifiers, complete. The last record may be an incomplete window.
        Window averages use the actual n_samples, including for a short tail.
    stats_ : dict
        Counts of covering, GA, crossover, subsumption, deletion, condensation.
    feature_offset_, feature_scale_ : ndarray
        Frozen affine input transformation; rules use these internal coordinates.
    target_mean_ : float
        Mean of all target observations processed during training.

    Notes
    -----
    Dense, finite, numeric features and one continuous target are supported.
    fit resets the population; partial_fit continues it. Neither method accepts
    sample_weight. Use sklearn.multioutput.MultiOutputRegressor for many targets.
    See docs/algorithm.md for source correspondence and intentional corrections.
    """

    def __init__(self, *, population_size=400, n_epochs=20, prediction="nlms",
                 degree=1, learning_rate=0.2, prediction_learning_rate=0.2,
                 epsilon_0=0.05, alpha=0.1, nu=5.0, theta_ga=25,
                 crossover_probability=0.8, mutation_probability=0.04,
                 cover_radius=0.2, mutation_scale=0.2, mutation="fixed",
                 crossover="one_point", theta_delete=20, delta=0.1,
                 theta_subsume=20, ga_subsumption=True, match_subsumption=False,
                 theta_match_subsume=100, selection="roulette", tournament_fraction=0.4,
                 initial_fitness=0.01, initial_error=0.0, initial_prediction=0.0,
                 use_mam=True, error_before_prediction=True, rls_delta=1000.0,
                 forgetting_factor=1.0, process_noise=0.0, kalman_noise=False,
                 lasso_alpha=0.001, lasso_window=256, lasso_max_iter=1000,
                 lasso_tol=1e-6, lasso_learning_rate_decay=0.0,
                 x0=1.0, normalize=True, bounded=False, discovery=True,
                 condensation_epochs=0, niche_history=0, history_interval=100, unmatched="nearest",
                 shuffle=True, random_state=None):
        self.population_size = population_size
        self.n_epochs = n_epochs
        self.prediction = prediction
        self.degree = degree
        self.learning_rate = learning_rate
        self.prediction_learning_rate = prediction_learning_rate
        self.epsilon_0 = epsilon_0
        self.alpha = alpha
        self.nu = nu
        self.theta_ga = theta_ga
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.cover_radius = cover_radius
        self.mutation_scale = mutation_scale
        self.mutation = mutation
        self.crossover = crossover
        self.theta_delete = theta_delete
        self.delta = delta
        self.theta_subsume = theta_subsume
        self.ga_subsumption = ga_subsumption
        self.match_subsumption = match_subsumption
        self.theta_match_subsume = theta_match_subsume
        self.selection = selection
        self.tournament_fraction = tournament_fraction
        self.initial_fitness = initial_fitness
        self.initial_error = initial_error
        self.initial_prediction = initial_prediction
        self.use_mam = use_mam
        self.error_before_prediction = error_before_prediction
        self.rls_delta = rls_delta
        self.forgetting_factor = forgetting_factor
        self.process_noise = process_noise
        self.kalman_noise = kalman_noise
        self.lasso_alpha = lasso_alpha
        self.lasso_window = lasso_window
        self.lasso_max_iter = lasso_max_iter
        self.lasso_tol = lasso_tol
        self.lasso_learning_rate_decay = lasso_learning_rate_decay
        self.x0 = x0
        self.normalize = normalize
        self.bounded = bounded
        self.discovery = discovery
        self.condensation_epochs = condensation_epochs
        self.niche_history = niche_history
        self.history_interval = history_interval
        self.unmatched = unmatched
        self.shuffle = shuffle
        self.random_state = random_state

    def _validate_parameters(self):
        if self.history_interval is not None:
            if (isinstance(self.history_interval, (bool, np.bool_))
                    or not isinstance(self.history_interval, Integral)
                    or self.history_interval < 1):
                raise ValueError("history_interval must be a positive integer or None.")
        if self.lasso_window is not None:
            if (isinstance(self.lasso_window, (bool, np.bool_))
                    or not isinstance(self.lasso_window, Integral) or self.lasso_window < 1):
                raise ValueError("lasso_window must be a positive integer or None.")
        for name in ("population_size", "n_epochs", "degree", "condensation_epochs", "niche_history", "lasso_max_iter"):
            value = getattr(self, name)
            minimum = 0 if name in ("condensation_epochs", "niche_history") else 1
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        positive = ("learning_rate", "prediction_learning_rate", "epsilon_0", "alpha", "nu",
                    "cover_radius", "delta", "tournament_fraction", "initial_fitness", "rls_delta",
                    "forgetting_factor", "x0", "lasso_tol")
        nonnegative = ("theta_ga", "crossover_probability", "mutation_probability", "mutation_scale",
                       "theta_delete", "theta_subsume", "theta_match_subsume", "initial_error", "process_noise",
                       "lasso_alpha", "lasso_learning_rate_decay")
        unit = ("learning_rate", "prediction_learning_rate", "alpha", "delta", "tournament_fraction",
                "forgetting_factor", "crossover_probability", "mutation_probability", "lasso_learning_rate_decay")
        for name in positive + nonnegative + ("initial_prediction",):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite real number.")
            if (name in positive and value <= 0) or (name in nonnegative and value < 0):
                raise ValueError(f"{name} is outside its valid range.")
            if name in unit and value > 1:
                raise ValueError(f"{name} must be <= 1.")
        for name in ("ga_subsumption", "match_subsumption", "use_mam", "error_before_prediction",
                     "kalman_noise", "normalize", "bounded", "discovery", "shuffle"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean.")
        for name, choices in {
            "prediction": PREDICTION_METHODS,
            "mutation": ("fixed", "proportional", "gaussian"),
            "crossover": ("one_point", "two_point", "uniform"),
            "selection": ("roulette", "tournament"),
            "unmatched": ("nearest", "mean", "raise"),
        }.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {choices}.")
        check_random_state(self.random_state)

    def _initialize(self, X):
        self.feature_offset_ = X.min(axis=0) if self.normalize else np.zeros(X.shape[1])
        self.feature_scale_ = np.ptp(X, axis=0) if self.normalize else np.ones(X.shape[1])
        self.feature_scale_[self.feature_scale_ < 10 * np.finfo(float).eps] = 1.0
        transformed = self._transform(X)
        bounds = (transformed.min(axis=0), transformed.max(axis=0)) if self.bounded else None
        self._rng = check_random_state(self.random_state)
        self._training_parameters = {k: deepcopy(v) for k, v in self.get_params().items()
                                     if k not in ("random_state", "n_epochs", "unmatched", "shuffle", "condensation_epochs")}
        self._core = XCSFCore(self.get_params(), self._rng, bounds)
        self.population_ = self._core.population
        self.history_ = []
        self.performance_history_ = []
        self._performance_count = 0
        self._performance_absolute_error = 0.0
        self._performance_squared_error = 0.0
        self.stats_ = self._core.stats
        self.n_iter_ = 0
        self.n_samples_seen_ = 0
        self.target_mean_ = 0.0
        self.n_microclassifiers_ = 0
        self.n_macroclassifiers_ = 0

    def _transform(self, X):
        return (X - self.feature_offset_) / self.feature_scale_

    def _training_data(self, X, y, *, reset):
        self._validate_parameters()
        if not reset:
            for name, value in self._training_parameters.items():
                if getattr(self, name) != value:
                    raise ValueError(f"{name} changed after training; call fit to reset the model.")
        X, y = validate_data(self, X, y, reset=reset, dtype=np.float64, y_numeric=True)
        if reset:
            self._initialize(X)
        X = self._transform(X)
        if self.bounded:
            lower, upper = self._core.bounds
            if np.any((X < lower) | (X > upper)):
                raise ValueError("Training inputs exceed the frozen bounds; use bounded=False or refit.")
        return X, y, design_matrix(X, self.degree, self.x0)

    def _pass(self, X, y, phi, *, shuffle, condensation=False):
        order = self._rng.permutation(len(X)) if shuffle else np.arange(len(X))
        error_sum = 0.0
        absolute_error_sum = 0.0
        # The tail is a snapshot of a still-open window. Keep its accumulators
        # across epochs/partial_fit calls and replace the snapshot as it grows.
        if self.performance_history_ and not self.performance_history_[-1]["complete"]:
            self.performance_history_.pop()
        for i in order:
            prediction = self._core.update(X[i], y[i], phi[i], condensation=condensation)
            error = y[i] - prediction
            error_sum += error ** 2
            absolute_error_sum += abs(error)
            self.n_samples_seen_ += 1
            self.target_mean_ += (y[i] - self.target_mean_) / self.n_samples_seen_
            if self.history_interval is not None:
                self._performance_count += 1
                self._performance_absolute_error += abs(error)
                self._performance_squared_error += error ** 2
                if self._performance_count == self.history_interval:
                    self._record_performance(complete=True)
                    self._performance_count = 0
                    self._performance_absolute_error = 0.0
                    self._performance_squared_error = 0.0
        self.n_iter_ += 1
        self.n_microclassifiers_ = self._core.numerosity
        self.n_macroclassifiers_ = len(self.population_)
        self._core.condition_arrays()
        if self._performance_count:
            self._record_performance(complete=False)
        self.history_.append(dict(epoch=self.n_iter_, step=self.n_samples_seen_, n_samples=len(X),
                                  mae=float(absolute_error_sum / len(X)),
                                  mse=float(error_sum / len(X)), rmse=float(np.sqrt(error_sum / len(X))),
                                  macroclassifiers=self.n_macroclassifiers_,
                                  microclassifiers=self.n_microclassifiers_,
                                  condensation=condensation))

    def _record_performance(self, *, complete):
        count = self._performance_count
        mse = float(self._performance_squared_error / count)
        self.performance_history_.append(dict(
            step=self.n_samples_seen_, n_samples=count,
            mae=float(self._performance_absolute_error / count), mse=mse, rmse=float(np.sqrt(mse)),
            macroclassifiers=len(self.population_), microclassifiers=self._core.numerosity,
            complete=complete,
        ))

    def fit(self, X, y):
        """Reset and learn from (n_samples, n_features) X and scalar targets y."""
        X, y, phi = self._training_data(X, y, reset=True)
        for _ in range(self.n_epochs):
            self._pass(X, y, phi, shuffle=self.shuffle)
        for _ in range(self.condensation_epochs):
            self._pass(X, y, phi, shuffle=self.shuffle, condensation=True)
        return self

    def partial_fit(self, X, y):
        """Perform one ordered online pass, preserving population and input scaling.

        The first batch defines normalization/bounds; use normalize=False with
        consistently scaled inputs when these should be controlled externally.
        Evolutionary parameters cannot be changed during an incremental run.
        """
        X, y, phi = self._training_data(X, y, reset=not hasattr(self, "_core"))
        self._pass(X, y, phi, shuffle=False)
        return self

    def _prediction_data(self, X):
        check_is_fitted(self, "population_")
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return self._transform(X)

    def predict(self, X):
        """Predict scalar targets without modifying population, statistics, or RNG."""
        X = self._prediction_data(X)
        phi = design_matrix(X, self._core.config.degree, self._core.config.x0)
        lower, upper = self._core.condition_arrays()
        fitness = np.array([cl.fitness for cl in self.population_])
        result = np.empty(len(X))
        for i, x in enumerate(X):
            mask = np.all((x >= lower) & (x <= upper), axis=1)
            if not mask.any():
                if self.unmatched == "raise":
                    raise ValueError(f"No classifier matches prediction sample {i}.")
                if self.unmatched == "mean":
                    result[i] = self.target_mean_
                    continue
                distances = np.linalg.norm(np.maximum(np.maximum(lower - x, x - upper), 0), axis=1)
                mask = distances == distances.min()
            indices = np.flatnonzero(mask)
            values = [self.population_[j].predictor.predict(phi[i]) for j in indices]
            result[i] = np.dot(probabilities(fitness[mask]), values)
        return result

    def match(self, X):
        """Return (n_samples, n_macroclassifiers) boolean rule membership.

        Columns follow population_ order. The shape and rule IDs can change
        after further training. Uncovered points produce all-false rows.
        """
        X = self._prediction_data(X)
        return np.column_stack([cl.condition.matches(X) for cl in self.population_])

    def get_rules(self):
        """Return independent inspection records with bounds in original X units.

        weights multiply [x0, z1, ..., zd, z1**2, ...], where z is the normalized
        input. Constant predictors instead use value. No global coef_ exists.
        """
        check_is_fitted(self, "population_")
        return [dict(
            id=cl.identifier,
            lower=cl.condition.lower * self.feature_scale_ + self.feature_offset_,
            upper=cl.condition.upper * self.feature_scale_ + self.feature_offset_,
            weights=cl.predictor.weights.copy(), value=cl.predictor.value,
            prediction=cl.predictor.method, fitness=cl.fitness, error=cl.error,
            prediction_diagnostics=dict(
                n_updates=cl.predictor.n_updates_, n_samples=cl.predictor.n_samples_,
                n_iter=cl.predictor.n_iter_, converged=cl.predictor.converged_,
                kkt_violation=cl.predictor.kkt_violation_,
            ),
            squared_error=cl.squared_error, numerosity=cl.numerosity,
            experience=cl.experience, set_size=cl.set_size,
            timestamp=cl.timestamp, created_at=cl.created_at, last_match=cl.last_match,
            match_history=list(cl.match_history),
        ) for cl in self.population_]
