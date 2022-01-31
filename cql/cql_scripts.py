import time
import threading, subprocess


# python main.py --seed 0 --env hopper-medium-v0
def single_exp(min_q_weight=1):
    command = ['python', 'online_cql.py', '--min_q_weight', str(min_q_weight), '--subtract_likelihood']
    # command = ['python', 'main.py', '--seed', str(seed), '--env', env_name]
    _ = subprocess.Popen(command)


def run():
    tasks = [(i, 'halfcheetah-medium-expert-v2') for i in range(3)]
    threads = []
    # for (seed, env_name) in tasks:
    for min_q_weight in [1, 3, 5]:
        # t_thread = threading.Thread(target=single_exp, args=(seed, env_name,))
        t_thread = threading.Thread(target=single_exp, args=(min_q_weight,))
        t_thread.start()
        threads.append(t_thread)
    [t.join() for t in threads]


if __name__ == '__main__':
    run()
