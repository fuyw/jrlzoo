import numpy as np
from utils import discount_cumsum
from models import ActorCritic
from mpi_utils import num_procs, setup_pytorch_for_mpi

from torch.optim import Adam


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE)
    for calculating the advantages of the state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lmbda=0.95):
        self.obs_buffer = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buffer = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buffer = np.zeros(size, dtype=np.float32)
        self.rew_buffer = np.zeros(size, dtype=np.float32)
        self.ret_buffer = np.zeros(size, dtype=np.float32)
        self.val_buffer = np.zeros(size, dtype=np.float32)
        self.logp_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lmbda = gamma, lmbda
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestamp of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size
        self.obs_buffer[self.ptr] = obs
        self.act_buffer[self.ptr] = act
        self.rew_buffer[self.ptr] = rew
        self.val_buffer[self.ptr] = val
        self.logp_buffer[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE, as
        well as compute the rewards-to-go for each state, to use as the
        targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timestamps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rew_buffer[path_slice], last_val)
        values = np.append(self.val_buffer[path_slice], last_val)

        # the next two lines implement GAE-lambda advantage calculation
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.adv_buffer[path_slice] = discount_cumsum(deltas, self.gamma*self.lmbda)

        # computes rewards-to-go
        self.ret_buffer[path_slice] = discount_cumsum(rewards, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted
        to have mean zero and std one). Also, resets some pointers in
        the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buffer)



def ppo(env_fn, actor_critic, seed, ac_kwargs, clip_ratio=0.2,
        pi_lr=3e-4, vf_lr=1e-4, epochs=50, gamma=0.99, lmbda=0.97,
        train_pi_iters=80, train_v_iters=80, max_ep_len=1000,
        target_kl=0.01, save_freq=10):
    """
    Args:
        env_fn: A function which creates a copy of the environment.
        actor_critic: The actor-critic model.
        seed: The random seed.
        ac_kwargs: Any kwargs for the ActorCritic model.
        clip_ratio: Clipping the policy objective.
            Roughly: how far can the new policy go from the old policy
            while still profiting?
    """
    # Special function to avoid certain slowdowns for PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Sync params across process
    sync_params(ac)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buffer = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lmbda)

    # Setup function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_datio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    def update():
        data = buffer.get()
        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in rage(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])

    # Main loop: collect experience in env and update each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            buffer.store(o, a, r, v, logp)

            # Update obs
            o = next_o

           timeout = ep_len==max_ep_len
           terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoched_ended and not(terminal):
                    print(f'Warning: trajectory cut off by epoch at {ep_len} steps.', flush=True)
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buffer.finish_path(v)
                if terminal:
                    print(f"EpRet = {ep_ret:.2f}, EpLen = {ep_len:.2f}")
                o, ep_ret, ep_len = env.reset(), 0, 0

        update()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    # run parallel code with mpi
    mpi_fork(args.cpu)

    ppo(lambda: gym.make(args.env), actor_critivc=ActorCritic)
