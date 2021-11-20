import io
import json

import kerastuner as kt
from hypermodel.auto_segment_hyper_model import HyperModel
from numpy.random import seed
from settings import FIT_TUNE_LOG_PATH
from tensorflow import keras
from utils.cloud_storage_client import (
    TUNER_HYPER_PARAMETER_GCS_PATH,
    storage_client,
    upload_blob,
)
from utils.data_utils import get_sample_weighted, scheduler


class ModelTuner:
    def __init__(
        self,
        model: HyperModel,
        data_train: tuple,
        data_val: tuple,
        config: dict,
    ):
        self.model = model
        self.config = config
        self.callbacks = []
        self.data_train = data_train
        self.data_val = data_val
        self.tuner = None
        self.init_callbacks()
        self.init_sample_weight()
        seed(config.seed)

    def init_callbacks(self):
        self.callbacks = [
            keras.callbacks.LearningRateScheduler(scheduler),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=6),
            keras.callbacks.CSVLogger(
                FIT_TUNE_LOG_PATH.format(org_id=self.config.org_id),
                separator=",",
                append=True,
            ),
        ]

    def init_sample_weight(self):
        self.sample_weights = get_sample_weighted(self.data_train[1])

    def tune_random_search(self):
        self.tuner = kt.tuners.randomsearch.RandomSearch(
            self.model,
            hyperparameters=self.model.HPS,
            objective=kt.Objective("val_rap", direction="max"),
            max_trials=self.config.max_trials,
            executions_per_trial=self.config.executions_per_trial,
            overwrite=True,
            directory=f"./random_search",
            seed=self.config.seed,
        )
        # the args `overwrite` and `directory` added to loop the airflow tasks
        self.tuner.search(
            self.data_train[0],
            self.data_train[1],
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=self.data_val,
            callbacks=self.callbacks,
            sample_weight=self.sample_weights,
        )

    def tune_hyperband(self):
        self.tuner = kt.tuners.Hyperband(
            self.model,
            hyperparameters=self.model.HPS,
            objective=kt.Objective("val_rap", direction="max"),
            max_trials=self.config.max_trials,
            executions_per_trial=self.config.executions_per_trial,
            overwrite=True,
            directory=f"./hyperband",
            seed=self.config.seed,
        )
        # the args `overwrite` and `directory` added to loop the airflow tasks
        self.tuner.search(
            self.data_train[0],
            self.data_train[1],
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=self.data_val,
            callbacks=self.callbacks,
            sample_weight=self.sample_weights,
        )

    def tune_bayesianOptimization(self):
        self.tuner = kt.tuners.BayesianOptimization(
            self.model,
            hyperparameters=self.model.HPS,
            objective=kt.Objective("val_rap", direction="max"),
            max_trials=self.config.max_trials,
            executions_per_trial=self.config.executions_per_trial,
            overwrite=True,
            directory=f"./{self.config.org_id}",
            seed=self.config.seed,
        )
        # the args `overwrite` and `directory` added to loop the airflow tasks
        self.tuner.search(
            self.data_train[0],
            self.data_train[1],
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=self.data_val,
            callbacks=self.callbacks,
            sample_weight=self.sample_weights,
        )

    def save_hyper_parameter(self):
        best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        buffer = io.StringIO()
        json.dump(best_hps.values, buffer)
        upload_blob(
            client=storage_client,
            upload_path=TUNER_HYPER_PARAMETER_GCS_PATH.format(
                org_id=self.config.org_id
            ),
            upload_file=io.BytesIO(buffer.getvalue().encode()),
        )
