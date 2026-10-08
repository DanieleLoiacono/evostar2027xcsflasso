"""Run after `python -m pip install -e .`; no graphical dependencies required."""

import numpy as np
from sklearn.model_selection import train_test_split

from xcsf import XCSFRegressor


def main():
    rng = np.random.RandomState(42)
    X = rng.uniform(0, 1, (800, 1))
    y = np.sin(2 * np.pi * X[:, 0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = XCSFRegressor(prediction="rls", random_state=42, n_epochs=200, condensation_epochs=50).fit(X_train, y_train)
    print(f"Test R²: {model.score(X_test, y_test):.6f}")
    print(f"Population: {model.n_macroclassifiers_} macro / {model.n_microclassifiers_} micro")
    print(f"Covered test samples: {model.match(X_test).any(axis=1).mean():.1%}")
    print(f"Evolution: {model.stats_}")



if __name__ == "__main__":
    main()

