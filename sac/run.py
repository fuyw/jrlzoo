import time
import threading, subprocess


# python main.py --seed 0 --env hopper-medium-v0
def single_exp(seed='0', env_name='HalfCheetah-v2'):
    command = ['python', 'main.py', '--seed', str(seed), '--env', env_name]
    _ = subprocess.Popen(command)


def run():
    tasks = [(i, 'HalfCheetah-v2') for i in range(1, 6)]

    threads = []
    for (seed, env_name) in tasks:
        t_thread = threading.Thread(target=single_exp, args=(seed, env_name,))
        t_thread.start()
        threads.append(t_thread)
    [t.join() for t in threads]


if __name__ == '__main__':
    run()
