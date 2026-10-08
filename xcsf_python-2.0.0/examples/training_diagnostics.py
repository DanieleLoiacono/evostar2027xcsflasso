"""Print learned rules and save error/population curves every 100 updates.

Install plotting dependencies first: python -m pip install -e '.[plot]'
Run: python examples/training_diagnostics.py --output training_history.png
"""

import argparse
from pathlib import Path

import numpy as np

from xcsf import XCSFRegressor
from xcsf.utils import plot_training_history, print_population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("training_history.png"))
    args = parser.parse_args()
    rng = np.random.RandomState(42)
    X = rng.uniform(0, 1, (500, 1))
    y = np.sin(2 * np.pi * X[:, 0])
    model = XCSFRegressor(prediction="rls", n_epochs=20,
                          history_interval=100, random_state=42).fit(X, y)
    print_population(model)
    figure, _ = plot_training_history(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    print(f"Saved {args.output}: {model.n_samples_seen_} updates, "
          f"{len(model.performance_history_)} recording windows.")


if __name__ == "__main__":
    main()
