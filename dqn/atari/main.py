import os

from absl import app, flags
from ml_collections import config_flags

import train, train_cql, train_qdagger

config_flags.DEFINE_config_file("config", default="configs/qdagger.py")
FLAGS = flags.FLAGS


def main(argv):
    config = FLAGS.config
    os.makedirs(f"{config.log_dir}/{config.env_name}", exist_ok=True)
    os.makedirs(f"{config.model_dir}/{config.env_name}", exist_ok=True)
    os.makedirs(f"{config.dataset_dir}/{config.env_name}", exist_ok=True)
    train_qdagger.train_and_evaluate(config)
    # train_cql.train_and_evaluate(config)
    # train.train_and_evaluate(config)


if __name__ == '__main__':
    app.run(main)