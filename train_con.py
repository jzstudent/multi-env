# -*- coding: utf-8 -*-
import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Tuple, Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG
import matplotlib.pyplot as plt
from ma_meta_env.envs.object import HeavyObjectEnv


def make_parallel_env(num_agents, n_rollout_threads, seed,shape_file):
    def get_env_fn(rank):
        def init_env():
            env = HeavyObjectEnv( num_agents=num_agents,shape_file=shape_file)
            #env.seed(seed + rank * 1000)
            #np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config):
    model_dir = Path('./models') / config.env_name / config.model_name
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
    os.system("cp shape.txt {}".format(run_dir))
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(run_num)
    np.random.seed(run_num)

    #training时的线程数
    if not config.use_cuda:
        torch.set_num_threads(config.n_training_threads)

    #env并行采样的进程

    env = make_parallel_env(
                            config.num_agents, 
                            config.n_rollout_threads, 
                            run_num,
                            config.shape_file)

    maddpg = MADDPG.init_from_env(env=env,
                                  agent_alg = config.agent_alg,
                                  cripple_alg = config.cripple_alg,
                                  tau = config.tau,
                                  lr = config.lr,
                                  hidden_dim = config.hidden_dim, 
                                  discrete_action = config.discrete_action)
    replay_buffer = ReplayBuffer(config.buffer_length, 
                                 maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space])
    
    t = 0
    a_loss = []
    c_loss = []
    rewss=[]

    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        
        obs = env.reset()

        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='cpu')# show for the first time

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        #if config.display:
        #    for env_show in env.envs:
        #        env_show.render('human', close=False)

        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)

            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]

            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]

            #actions = [np.array([i.tolist().index(1.0) for i in action]) for action in actions_one_hot]

            for i in actions:
            #    print(i)
                for j in i:
                    j[1]*=np.pi
            #print(actions[0])

            next_obs, rewards, dones, infos = env.step(actions)

            #print(len(agent_actions),len(next_obs))
            #if config.display:
            #    for env_show in env.envs:
            #        env_show.render('human', close=False)

            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones)
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                #print(t)
                if config.use_cuda:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=config.use_cuda,norm_rews=False)
                        maddpg.update(sample, a_i, logger=logger, actor_loss_list=a_loss, critic_loss_list=c_loss)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        rewss.append(ep_rews)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
            # print('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(str(run_dir / 'incremental'), exist_ok=True)
            maddpg.save(str(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1))))
            maddpg.save(str(run_dir / 'model.pt'))
    maddpg.save(str(run_dir / 'model.pt'))
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    '''
    index_aloss = list(range(1, len(a_loss) + 1))

    plt.plot(index_aloss, a_loss)
    plt.ylabel('actor_loss')
    plt.savefig("./results/" + config.env_name + "_" + config.model_name + "_actor_loss.jpg")
    #plt.savefig("_actor_loss.jpg")
    plt.close()
    

    index_closs = list(range(1, len(c_loss) + 1))

    plt.plot(index_closs, c_loss)
    plt.ylabel('critic_loss')
    plt.savefig("./results/" + config.env_name + "_" + config.model_name + "_critic_loss.jpg")
    #plt.savefig("_critic_loss.jpg")
    plt.close()

    index_closs = list(range(1, len(rewss) + 1))

    plt.plot(index_closs, rewss)
    plt.ylabel('reward')
    plt.savefig("./results/" + config.env_name + "_" + config.model_name + "_reward.jpg")
    # plt.savefig("_critic_loss.jpg")
    plt.close()
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("shape_file", help="Shape file")
    parser.add_argument("num_agents", default=3, type=int)
    # env params
    parser.add_argument("--n_rollout_threads", default=256, type=int)
    parser.add_argument("--n_training_threads", default=8, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)

    parser.add_argument("--n_episodes", default=25000, type=int)
    parser.add_argument("--episode_length", default=45, type=int)
    parser.add_argument("--steps_per_update", default=200, type=int)

    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")

    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)

    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--cripple_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action", action='store_true')
    parser.add_argument("--display", action='store_true')
    parser.add_argument("--use_cuda", action='store_true')

    config = parser.parse_args()

    run(config)
