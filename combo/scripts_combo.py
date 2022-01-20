import time
import threading, subprocess

# python main.py --seed 0 --env hopper-medium-v0 --metric uncertainty --sleep 1
def single_exp(seed='0', env_name='hopper-medium-v0', weight_decay=1e-5):
    command = ['python', 'main.py', '--seed', str(seed), '--env', env_name,
               '--weight_decay', str(weight_decay)]
    _ = subprocess.Popen(command)


def run():
    tasks = [
        (42, 'hopper-medium-v2', 5e-2),
        (42, 'hopper-medium-v2', 5e-3),
        (42, 'hopper-medium-v2', 1e-4),
    ]

    threads = []
    for (seed, env_name, weight_decay) in tasks:
        t_thread = threading.Thread(target=single_exp, args=(seed, env_name, weight_decay,))
        t_thread.start()
        threads.append(t_thread)
    [t.join() for t in threads]


if __name__ == '__main__':
    run()
