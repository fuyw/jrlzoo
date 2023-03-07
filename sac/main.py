from absl import app, flags
from ml_collections import config_flags
import os
import train, train_tandem

config_flags.DEFINE_config_file("config", default="configs/mujoco.py")
FLAGS = flags.FLAGS


def main(argv):
    configs = FLAGS.config
    os.makedirs(f"{configs.log_dir}/{configs.env_name.lower()}", exist_ok=True)
    os.makedirs(f"{configs.model_dir}/{configs.env_name.lower()}", exist_ok=True)
    train_tandem.train_and_evaluate(configs)


if __name__ == '__main__':
    app.run(main)
