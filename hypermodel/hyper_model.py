from typing import Optional, Tuple

import kerastuner as kt
from kerastuner import HyperModel
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, TimeDistributed
from tensorflow.keras.models import Sequential

# All recorded metrics
PRECISION_LIMIT = 0.1
METRICS = [
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.FalseNegatives(name="fn"),
    keras.metrics.BinaryAccuracy(name="accuracy"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.AUC(name="auc"),
    keras.metrics.sparse_categorical_accuracy,
    keras.metrics.RecallAtPrecision(PRECISION_LIMIT, name="rap"),
]


# HyperModel
class HyperModel(HyperModel):
    def __init__(self, layer_added_options: Optional[list] = None):
        if layer_added_options:
            self.layer_added_options = layer_added_options
        else:
            self.layer_added_options = ["dropouts", "dropout"]
        self.HPS = kt.engine.hyperparameters.HyperParameters()
        self.HPS.Choice("layer", values=self.layer_added_options)
        self.HPS.Int("unit_1", min_value=32, max_value=64, step=8)
        self.HPS.Int("unit_2", min_value=16, max_value=32, step=8)
        self.HPS.Int("unit_3", min_value=4, max_value=16, step=4)
        self.HPS.Choice("dropout_1", values=[0.05, 0.1, 0.15, 0.2])
        self.HPS.Choice("dropout_2", values=[0.05, 0.1, 0.15, 0.2])
        self.HPS.Choice("dropout_3", values=[0.05, 0.1, 0.15, 0.2, 0.25])
        self.HPS.Choice("batch_normalization", values=[0.8, 0.9, 0.7, 0.6, 0.5])
        self.HPS.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    def build(
        self,
        hp: kt.engine.hyperparameters.HyperParameters,
        predicted: Optional[Tuple[int]] = None,
    ) -> Sequential:

        model = Sequential()
        if predicted:
            model.add(Input(shape=predicted))
        layer_added = hp.get("layer")
        if layer_added == "dropouts":
            model.add(
                LSTM(
                    hp.get("unit_1"),
                    return_sequences=True,
                )
            )
            model.add(Dropout(hp.get("dropout_1")))
            model.add(
                LSTM(
                    hp.get("unit_2"),
                    return_sequences=True,
                )
            )
            model.add(Dropout(hp.get("dropout_2")))
        elif layer_added == "dropout":
            model.add(
                LSTM(
                    hp.get("unit_1"),
                    return_sequences=True,
                )
            )
            model.add(
                LSTM(
                    hp.get("unit_2"),
                    return_sequences=True,
                )
            )
        else:
            model.add(
                LSTM(
                    hp.get("unit_1"),
                    return_sequences=True,
                )
            )
            model.add(
                LSTM(
                    hp.get("unit_2"),
                    return_sequences=True,
                )
            )
            model.add(
                keras.layers.BatchNormalization(
                    axis=-1,
                    momentum=hp.get("batch_normalization"),
                )
            )

        model.add(Dropout(hp.get("dropout_3")))

        model.add(
            TimeDistributed(
                Dense(
                    hp.get("unit_3"),
                    activation="relu",
                )
            )
        )

        model.add(TimeDistributed(Dense(units=1, activation="sigmoid")))
        adam = keras.optimizers.Adam(
            hp.get("learning_rate"),
        )
        model.compile(
            optimizer=adam,
            loss="binary_crossentropy",
            metrics=METRICS,
            sample_weight_mode="temporal",
        )
        return model
