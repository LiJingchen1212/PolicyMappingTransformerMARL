import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.Plastic_sac import PlasticSAC


def make_parallel_env(env_id, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=False)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def abstract_observation(obs_n):

    obs_s=obs_n.transpose(0,2,1)
    return obs_s

def abstract_action(abs_action):

    return [(torch.cat(abs_action).reshape(len(abs_action),
            abs_action[0].shape[0],-1))[:,:,i].T for i in range (abs_action[0].shape[1])]


def abstract_reward(rewards,action_space):
    rew = np.ones((1,action_space))*np.mean(rewards)
    return rew

def abstract_done(dones,action_space):
    if dones.all():
        return np.ones((1,action_space),dtype=bool)
    else:
        return np.zeros((1,action_space),dtype=bool)

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(run_num)
    np.random.seed(run_num)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, run_num)
    model = PlasticSAC.init_from_env(env,
                                       tau=config.tau,
                                       pi_lr=config.pi_lr,
                                       q_lr=config.q_lr,
                                       gamma=config.gamma,
                                       pol_hidden_dim=config.pol_hidden_dim,
                                       critic_hidden_dim=config.critic_hidden_dim,
                                       attend_heads=config.attend_heads,
                                       reward_scale=config.reward_scale)
    replay_buffer = ReplayBuffer(config.buffer_length, env.action_space[0].shape[0],
                                 [(env.observation_space[0].shape[0],len(env.action_space)) for i in range(env.action_space[0].shape[0])],
                                 [len(env.action_space) for i in range(env.action_space[0].shape[0])],config.pol_hidden_dim)

    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()


        obs = abstract_observation(obs)
        pre_actions = [torch.zeros((config.n_rollout_threads,config.pol_hidden_dim)) for i in range(env.action_space[0].shape[0])] #2,8,3
        model.prep_rollouts(device='cpu')

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable

            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i,:])),
                                  requires_grad=False)
                         for i in range(env.observation_space[0].shape[0])]
            #torch_obs=abstract_observation(torch_obs)
            inps = [torch_obs, pre_actions]

            # get actions as torch Variables
            torch_agent_actions = model.step(inps, explore=True)
            abs_actions = [acts for acts, log, mean, pre_hidden in torch_agent_actions]
            next_hidden = [pre_hidden for acts, log, mean, pre_hidden in torch_agent_actions]
            true_actions = abstract_action(abs_actions)

            abs_action = [ac.data.numpy() for ac in abs_actions]
            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in true_actions]
            # rearrange actions to be per environment

            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos= env.step(actions)

            #env.render()
            next_obs=abstract_observation(next_obs)
            abs_rewards = abstract_reward(rewards,env.action_space[0].shape[0])
            abs_dones = abstract_done(dones,env.action_space[0].shape[0])
            replay_buffer.push(obs, pre_actions, abs_action, abs_rewards, next_obs, next_hidden, abs_dones)

            obs = next_obs
            pre_actions = [aa.squeeze(dim=0) for aa in next_hidden]
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if config.use_gpu:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')
                for u_i in range(config.num_updates):
                    sample = replay_buffer.sample(config.batch_size,
                                                  to_gpu=config.use_gpu)
                    model.update_critic(sample, logger=logger)
                    model.update_policies(sample, logger=logger)
                    model.update_all_targets()
                model.prep_rollouts(device='cpu')
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i,
                              a_ep_rew * config.episode_length, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            model.prep_rollouts(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            model.save(run_dir / 'model.pt')

    model.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", help="Name of environment")
    parser.add_argument("--model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--n_rollout_threads", default=8, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=50000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--num_updates", default=4, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for training")
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--pol_hidden_dim", default=128, type=int)
    parser.add_argument("--critic_hidden_dim", default=128, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", action='store_true')

    config = parser.parse_args()

    run(config)
