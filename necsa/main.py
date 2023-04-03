from absl import app, flags
from ml_collections import config_flags
import os
import train


config_flags.DEFINE_config_file("config", default="configs/mujoco.py")
FLAGS = flags.FLAGS


def main(argv):
    config = FLAGS.config
    os.makedirs(f"{config.log_dir}/{config.env_name}", exist_ok=True)
    os.makedirs(f"{config.model_dir}/{config.env_name}", exist_ok=True)
    train.train_and_evaluate(config)


if __name__ == '__main__':
    app.run(main)
