"""
multiple process
mpirun -np 4 python -u train.py

single process
python train.py
"""







gym.register_envs(gymnasium_robotics)


###################
# Utils Functions #
###################

#################
# Main Function #
#################
config = get_config()


def run(config, comm):
    # random seed
    rank = comm.Get_rank()  # 0
    rank_seed = config.seed + rank
    np.random.seed(rank_seed)
    rng = jax.random.PRNGKey(rank_seed)

    # initialize vectorized env
    envs = make_env(config, rank_seed)
    act_dim = envs.action_space.shape[0]
    obs_dim = envs.observation_space["observation"].shape[0]
    goal_dim = envs.observation_space["desired_goal"].shape[0]
    max_action = envs.action_space.high[0]  # 1.0
    traj_len = envs._max_episode_steps  # 50
    num_envs = len(envs)  # 4

    # evaluation env
    eval_env = gym.make(config.env_name, render_mode="rgb_array")
    eval_env.action_space.seed(config.seed+123)
    eval_env.observation_space.seed(config.seed+123)
    _ = eval_env.reset(seed=config.seed+123)

    # initialize RL agent
    agent = DDPGAgent(obs_dim=obs_dim,
                      act_dim=act_dim,
                      goal_dim=goal_dim)

    # replay buffer
    buffer = HERBuffer(obs_dim=obs_dim,
                       act_dim=act_dim,
                       goal_dim=goal_dim,
                       replay_k=config.replay_k,
                       max_size=config.max_size,
                       epsilon=config.epsilon)

    # trajectory data
    traj_observations = np.zeros((num_envs, traj_len+1, obs_dim),
                                 dtype=np.float32)
    traj_achieved_goals = np.zeros((num_envs, traj_len+1, goal_dim),
                                   dtype=np.float32)
    traj_goals = np.zeros((num_envs, traj_len, goal_dim),
                          dtype=np.float32)
    traj_actions = np.zeros((num_envs, traj_len, act_dim),
                            dtype=np.float32)
    traj_dones = np.zeros((num_envs, traj_len), dtype=bool)
    traj_ptr = 0

    # reset the environment
    observations, _ = envs.reset(seed=config.seed)
    traj_observations[:, 0, :] = observations["observation"]
    traj_achieved_goals[:, 0, :] = observations["achieved_goal"]

    # imageio.mimsave("a.mp4", eval_frames)

    # online interaction
    for t in trange(N, config.max_timesteps+N, N):
        if t <= config.start_timesteps:
            actions = np.array([envs.action_space.sample()
                                for _ in range(num_envs)])
        else:
            # add exploration noises
            noises = np.random.normal(0, max_action * config.expl_noise,
                                      size=(config.num_envs, act_dim))
            actions = agent.sample_action(
                observations["observation"],
                observations["desired_goal"])
            actions = (actions + noises).clip(-max_action, max_action)

        # interact with the environment
        observations, rewards, terminals, truncations, _ = envs.step(actions)

        # save trajectory data
        traj_observations[:, traj_ptr+1, :] = observations["observation"]
        traj_achieved_goals[:, traj_ptr+1, :] = observations["achieved_goal"]
        traj_goals[:, traj_ptr, :] = observations["desired_goal"]
        traj_actions[:, traj_ptr, :] = actions
        traj_dones[:, traj_ptr] = terminals
        traj_ptr += 1

        # check any(terminals) 
        if any(truncations) or any(terminals):
            # add trajectory data to trajectory buffer
            buffer.add(traj_observations,
                       traj_achieved_goals,
                       traj_goals,
                       traj_actions,
                       traj_dones)

            # reset
            traj_ptr = 0
            envs.reset()

            # TODO: remove zero operation
            traj_observations[:] = 0
            traj_achieved_goals[:] = 0
            traj_actions[:] = 0
            traj_dones[:] = 0
            traj_observations[:, 0, :] = observations["observation"]
            traj_achieved_goals[:, 0, :] = observations["achieved_goal"]

        if t > config.start_timesteps:
            batch = buffer.sample(config.batch_size)
            log_info = agent.update(batch) 

            # {'actor_loss': Array(0.02045047, dtype=float32),
            # 'critic_loss': Array(0.37976623, dtype=float32),
            # 'max_q1': Array(0.11077082, dtype=float32),
            # 'min_q1': Array(-0.12474349, dtype=float32),
            # 'q1': Array(-0.00610746, dtype=float32)}

            # batch.observations.shape = (256, 25)
            # batch.actions.shape = (256, 4)
            # batch.discounts = (256,)
            # batch.next_observations.shape = (256, 25)
            # batch.goals = (256, 3)
            # batch.rewards = (256,)

        if rank == 0 and t % config.eval_freq == 0:
            eval_reward, eval_frames = eval_agent(agent, eval_env, 10)
