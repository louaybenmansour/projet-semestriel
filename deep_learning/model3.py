"""Regularized deep neural network with L2 penalties."""

from __future__ import annotations

from tensorflow import keras


def build_model(input_dim: int) -> keras.Model:
    """Build a regularized MLP suitable for noisy tabular data."""
    reg = keras.regularizers.l2(1e-4)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128, activation="relu", kernel_regularizer=reg),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation="relu", kernel_regularizer=reg),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(32, activation="relu", kernel_regularizer=reg),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="linear"),
        ],
        name="regularized_dense_nn",
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
