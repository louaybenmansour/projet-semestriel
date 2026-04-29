"""Simple dense neural network for tabular regression."""

from __future__ import annotations

from tensorflow import keras


def build_model(input_dim: int) -> keras.Model:
    """Build a compact MLP with 2 hidden layers."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(1, activation="linear"),
        ],
        name="simple_dense_nn",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="mse",
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model
