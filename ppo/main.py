from absl import app, flags
from ml_collections import config_flags
import os
import train


config_flags.DEFINE_config_file("config", default="configs/atari.py")
FLAGS = flags.FLAGS


def main(argv):
    configs = FLAGS.config
    train.train_and_evaluate(configs)


if __name__ == "__main__":
    app.run(main)

