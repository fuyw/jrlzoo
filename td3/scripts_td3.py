import time
import threading, subprocess

# python main.py --seed 0 --env hopper-medium-v0 --metric uncertainty --sleep 1
def single_exp(seed='0', env_name='hopper-medium-v0', sleep=1):
    command = ['python', 'main.py', '--seed', str(seed), '--env', env_name, '--with_qinfo']
    _ = subprocess.Popen(command)


def run():
    tasks = [
        (0, 'Hopper-v2'),
        (0, 'Walker2d-v2'),
        (0, 'HalfCheetah-v2'),
    ]

    threads = []
    for (seed, env_name) in tasks:
        t_thread = threading.Thread(target=single_exp, args=(seed, env_name))
        t_thread.start()
        threads.append(t_thread)
    [t.join() for t in threads]


if __name__ == '__main__':
    run()
