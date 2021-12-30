import time
import threading, subprocess


# python main.py --seed 0 --env hopper-medium-v0
def single_exp(seed='0', env_name='HalfCheetah-v2'):
    command = ['python', 'main.py', '--seed', str(seed), '--env', env_name]
    _ = subprocess.Popen(command)


def run():
    for t in range(3):
        tasks = [(i, 'hopper-medium-expert-v2') for i in range(2*t, 2*t+2)]
        threads = []
        for (seed, env_name) in tasks:
            t_thread = threading.Thread(target=single_exp, args=(seed, env_name,))
            t_thread.start()
            threads.append(t_thread)
        [t.join() for t in threads]
        time.sleep(3600 * 3)


if __name__ == '__main__':
    run()
