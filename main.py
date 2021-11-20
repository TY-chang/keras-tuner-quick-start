from tensorflow import keras

from data.data_loader import load_data
from hypermodel.auto_segment_hyper_model import AutoSegmentHyperModel
from tuner.auto_segment_tuner import AutoSegmentModelTuner
from utils.data_utils import Task, process_config


def main():
    config = process_config(Task.TUNE)

    data_loader = load_data()
    x_train, y_train, x_val, y_val = data_loader.get_dataset_for_trainer(
        config.moving_day, config.target
    )

    hypermodel = AutoSegmentHyperModel()

    tuner = AutoSegmentModelTuner(
        hypermodel, (x_train, y_train), (x_val, y_val), config
    )
    tuner.tune()
    tuner.save_hyper_parameter()


if __name__ == "__main__":
    main()
