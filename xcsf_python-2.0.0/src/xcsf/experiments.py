"""Learning loop and online statistics (cf. xcslib ``experiments/experiment_mgr``).

The loop only connects an environment to a classifier system, one single-step
learning problem at a time; it holds no learning logic of its own.
"""

import numpy as np


def run_problems(system, environment, n_problems, *, condensation=False, on_problem=None):
    """Run ``n_problems`` learning problems of ``environment`` on ``system``.

    ``on_problem(prediction, target)`` receives the system prediction made
    before the problem was learned, i.e. an online test of the current model.
    """
    for _ in range(n_problems):
        environment.begin_problem()
        state = environment.state()
        environment.perform(system.action)
        target = environment.reward()
        prediction = system.step(state, target, environment.features(), condensation=condensation)
        environment.end_problem()
        if on_problem is not None:
            on_problem(prediction, target)


class TrainingMonitor:
    """Online error and population size over consecutive windows of problems.

    ``interval`` problems make one window; None disables the windows. The last
    record may describe a still-open window (``complete=False``): it is
    replaced as the window grows, across passes and partial_fit calls.
    """

    def __init__(self, interval):
        self.interval = interval
        self.records = []
        self._count = 0
        self._absolute_error = 0.0
        self._squared_error = 0.0

    def begin_pass(self):
        if self.records and not self.records[-1]["complete"]:
            self.records.pop()

    def add(self, error, step, system):
        if self.interval is None:
            return
        self._count += 1
        self._absolute_error += abs(error)
        self._squared_error += error ** 2
        if self._count == self.interval:
            self._record(step, system, complete=True)
            self._count = 0
            self._absolute_error = 0.0
            self._squared_error = 0.0

    def end_pass(self, step, system):
        if self._count:
            self._record(step, system, complete=False)

    def _record(self, step, system, *, complete):
        mse = float(self._squared_error / self._count)
        self.records.append(dict(
            step=step, n_samples=self._count, mae=float(self._absolute_error / self._count),
            mse=mse, rmse=float(np.sqrt(mse)), macroclassifiers=len(system.population),
            microclassifiers=system.numerosity, complete=complete,
        ))
