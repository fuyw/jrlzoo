import os

from absl import app, flags
from ml_collections import config_flags

import train

config_flags.DEFINE_config_file("config", default="configs/dmc.py")
FLAGS = flags.FLAGS


def main(argv):
    config = FLAGS.config
    train.train_and_evaluate(config)


if __name__ == "__main__":
    app.run(main)
