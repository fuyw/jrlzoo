import collections
import logging


ExpTuple = collections.namedtuple('ExpTuple', ['state', 'action', 'reward', 'value', 'log_prob', 'done'])


def get_logger(fname):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=fname,
                        filemode='w',
                        force=True)
    logger = logging.getLogger()
    return logger


