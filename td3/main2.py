from absl import app, flags
from ml_collections import config_flags
import os
import train, train_earlystop


config_flags.DEFINE_config_file("config", default="configs/mujoco.py")
FLAGS = flags.FLAGS


def main(argv):
    config = FLAGS.config
    train_earlystop.train_and_evaluate(config)


if __name__ == '__main__':
    app.run(main)
