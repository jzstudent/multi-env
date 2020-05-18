import numpy as np
from ma_meta_env.envs.object_discre import HeavyObjectEnv


def run_env(fix_goal, num_agents,shape_file):
    env = HeavyObjectEnv(fix_goal=fix_goal, num_agents=num_agents,shape_file=shape_file)
    env.reset()
    step = 0
    obs_space = env.observation_space
    acs_space = env.action_space
    #print(obs_space[0].shape)
    #print(acs_space,acs_space[0].n)
    total_reward = 0
    while True:
        #ac = np.array([[0.2, 0.3]] * env.num_agents)
        ac = np.array([ac_space.sample() for ac_space in acs_space])
        #print("sample",ac)
        ob, rew, done, _ = env.step(ac)
        #print("ob",ob)
        total_reward += rew[0]
        env.render()
        step += 1
        if done[0]:
            print("Total rew", total_reward)
            total_reward = 0
            ob = env.reset()

if __name__ == "__main__":
    run_env(True, 3,"shape.txt")
