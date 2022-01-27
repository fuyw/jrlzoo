import time
import threading, subprocess


def single_exp(seed='0', env_name="halfcheetah-medium-v0"):
    command = ['python', 'main.py', '--seed', str(seed), '--env', env_name]
    _ = subprocess.Popen(command)


def run():
    envs = ["walker2d", "hopper", "halfcheetah"]
    levels = ["medium", "medium-replay", "medium-expert"]
    tasks = [f"{i}-{j}-v2" for i in envs for j in levels]
    seed = 0
    for i in range(0, len(tasks), 3):
        threads = []
        # for (seed, env_name) in tasks[i:i+3]:
        for env_name in tasks[i:i+3]:
            t_thread = threading.Thread(target=single_exp, args=(seed, env_name,))
            t_thread.start()
            threads.append(t_thread)
        [t.join() for t in threads]
    # time.sleep(2*3600)
    # tasks = [(i, "hopper-medium-expert-v2") for i in range(0, 5)]
    # threads = []
    # for (seed, env_name) in tasks:
    #     t_thread = threading.Thread(target=single_exp, args=(seed, env_name,))
    #     t_thread.start()
    #     threads.append(t_thread)
    # [t.join() for t in threads]


if __name__ == '__main__':
    run()
