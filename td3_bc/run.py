import time
import threading, subprocess


def single_exp(seed='0', env_name="halfcheetah-medium-v0"):
    command = ['python', 'main.py', '--seed', str(seed), '--env', env_name]
    _ = subprocess.Popen(command)


def run():
    # tasks = [(i, "halfcheetah-medium-v0") for i in range(0, 5)]
    # threads = []
    # for (seed, env_name) in tasks:
    #     t_thread = threading.Thread(target=single_exp, args=(seed, env_name,))
    #     t_thread.start()
    #     threads.append(t_thread)
    # [t.join() for t in threads]
    # time.sleep(2*3600)
    tasks = [(i, "hopper-medium-v2") for i in range(0, 5)]
    threads = []
    for (seed, env_name) in tasks:
        t_thread = threading.Thread(target=single_exp, args=(seed, env_name,))
        t_thread.start()
        threads.append(t_thread)
    [t.join() for t in threads]


if __name__ == '__main__':
    run()
