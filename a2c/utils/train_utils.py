import git
import logging


def get_logger(fname: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=fname,
                        filemode='w',
                        force=True)
    logger = logging.getLogger()
    return logger


def add_git_info(config):
    config.unlock()
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    config["commit"] = sha
