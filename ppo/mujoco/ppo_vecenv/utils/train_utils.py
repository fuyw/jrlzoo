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


def get_lr_scheduler(config, loop_steps, iterations_per_step):
    # set lr scheduler
    if config.decaying_lr_and_clip_param:
        lr = optax.linear_schedule(init_value=config.lr,
                                   end_value=0.,
                                   transition_steps=loop_steps *
                                   config.num_epochs * iterations_per_step)
    else:
        lr = config.lr
    return lr
