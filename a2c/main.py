import os

from absl import app, flags
from ml_collections import config_flags

import train, train2


config_flags.DEFINE_config_file("config", default="configs/dmc.py")
FLAGS = flags.FLAGS


def main(argv):
    try:
        config = FLAGS.config
        train.train_and_evaluate(config)
        # train2.train_and_evaluate(config)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
