import os
import sys
from absl import app, flags

from ml_collections import config_flags

import experiments


def get_config():
    config_file = "configs/baseline.py"
    config_flags.DEFINE_config_file("config", default=config_file)
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    config = FLAGS.config
    return config


if __name__ == "__main__":
    config = get_config()
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".45"

    if config.model in ("ddpg", "sac"):
        experiments.baseline.run(config)
