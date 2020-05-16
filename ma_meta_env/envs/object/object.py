import os
import gym
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from gym.spaces import Box
import math

"""
Multiple agents carry one heavy object
The object is too heavy for anyone to singly carry
The object's shape can be arbitrary, only need to know its centroid and mass and where agents hold on it
The object's centroid and mass can be changed during execution

Author: JiangZhuo
"""

PAPER_FIX_DIST = 3.5


class HeavyObjectEnv(gym.Env):
    """
    Distinct agent
    """

    def __init__(
        self,
        goal=None,
        observable_target=True,
        fix_goal=True,
        fix_dist=None,
        num_agents=2,
        centralized=False,
        max_episode_steps=50,
        shape_file="shape.txt",
    ):

        # Properties
        self.observable_target = observable_target
        self.centralized = centralized
        assert num_agents in [1, 2, 3, 4, 5]
        self.num_agents = num_agents
        self.goal = goal
        self.fix_goal = fix_goal
        self.fix_dist = fix_dist
        self._max_episode_steps = max_episode_steps

        # Environment & Episode settings
        self.map_W = 15.0
        self.map_H = 15.0

        #read where agents hold on and centroid of object
        xyc=[]
        with open(shape_file, "r") as f:
            data = f.readlines()
            for line in data:
                line=line[1:-2]
                line=line.split(",")
                for i in range(len(line)):
                    line[i]=float(line[i])
                xyc.append(line)
        self.agents_x, self.agents_y, self.raw_centroid=xyc[0],xyc[1],xyc[2]
        self.centroid=self.raw_centroid
        self.cen_change_x=0
        self.cen_change_y=0
        #self.goal_r_lenth=self.get_r_length()
        self.mass=0.2*num_agents
        self.inertia = 0.1 * num_agents
        self.max_robot_force = 1

        #执行时间
        self.dt = 0.5


        # Book keepings
        self._last_value = 0
        self._step_count = 0
        self._total_reward = 0
        self._fig = None  # only for visualization
        self._fig_folder = None
        self._hist = {}  # only for making figure
        self._last_hist = {}  # only for making figure
        self._recordings = {}  # only for making figure
        self._last_actions = None
        self._state = None
        self._last_state = None

        # Action space
        self.action_low = np.array([-self.max_robot_force, -np.pi])
        self.action_high = np.array([self.max_robot_force, np.pi])
        if self.centralized:
            # Centralized control: x, y, r
            #x轴力，y轴力，r是转矩
            self.action_low = np.array(
                [-self.max_robot_force, -self.max_robot_force, -self.max_robot_force]
            )
            self.action_high = np.array(
                [self.max_robot_force, self.max_robot_force, self.max_robot_force]
            )

        # Observation space
        self.obs_low = np.array([-self.map_W / 2, -self.map_H / 2, -np.pi])
        self.obs_high = np.array([self.map_W / 2, self.map_H / 2, np.pi])
        #if self.observable_target:
            #self.obs_low = np.concatenate([self.obs_low, self.obs_low])
            #self.obs_high = np.concatenate([self.obs_high, self.obs_high])

        # Sample one goal and keep it fixed
        if self.goal is None:
            self.goal = self.sample_goals(1)[0]

    def get_angle_offsets(self):
        #agents_x = self.agents_x - self.centroid[0]
        #print(agents_x)
        #agents_y = self.agents_y - self.centroid[1]
        agents_x=[x-self.centroid[0] for x in self.agents_x]
        #print(agents_x)
        agents_y = [x - self.centroid[1] for x in self.agents_y]
        angle_offset = []
        for i in range(len(agents_x)):
            angle_offset.append(math.atan2(agents_y[i], agents_x[i]))
        return angle_offset

    def get_r_length(self):
        r_length = []
        #agents_x = self.agents_x - self.centroid[0]
        #agents_y = self.agents_y - self.centroid[1]
        agents_x = [x - self.centroid[0] for x in self.agents_x]
        agents_y = [x - self.centroid[1] for x in self.agents_y]
        for i in range(len(agents_x)):
            r_length.append(math.sqrt(math.pow(agents_x[i], 2) + math.pow(agents_y[i], 2)))
        return r_length
    #质心改变接口
    def change_centroid(self,x,y):
        self.centroid[0]+=x
        self.centroid[1]+=y
        self.cen_change_x=x
        self.cen_change_y=y

    @property
    def observation_space(self):
        if self.centralized:
            return [
                Box(
                    np.array([self.obs_low] * self.num_agents),
                    np.array([self.obs_high] * self.num_agents),
                    dtype=np.float32
                )
            ]
        else:
            return [
                Box(self.obs_low, self.obs_high, dtype=np.float32)
                for _ in range(self.num_agents)
            ]

    @property
    def action_space(self):
        if self.centralized:
            return [Box(self.action_low, self.action_high, dtype=np.float32)]
        else:
            return [
                Box(self.action_low, self.action_high, dtype=np.float32)
                for _ in range(self.num_agents)
            ]

    @property
    def num_decentralized(self):
        return self.num_agents

    @property
    def identity_space(self):
        low = [0]
        high = [self.num_agents]
        return Box(low=np.array(low), high=np.array(high), dtype=np.float32)

    def generate_random_goal(self, fix_dist=None):
        if fix_dist is not None:
            assert type(fix_dist) == float
            angle = np.random.uniform(low=-np.pi, high=np.pi)
            theta = np.random.uniform(low=-np.pi, high=np.pi)
            pair = np.array([fix_dist * np.cos(angle), fix_dist * np.sin(angle), theta])
        else:
            pair = np.random.uniform(
                low=[-self.map_W / 2, -self.map_H / 2, -np.pi],
                high=[self.map_W / 2, self.map_H / 2, np.pi],
            )
        return pair

    def generate_default_start(self):
        return [0, 0, 0]

    def sample_paper_goals(self):
        goals = None
        if self.num_agents == 2:
            num_goals = 5
            angle = np.array(
                [0, np.pi * 2 / 5, np.pi * 4 / 5, np.pi * 6 / 5, np.pi * 8 / 5]
            )
            theta = np.random.uniform(low=[-np.pi], high=[np.pi], size=(num_goals,))
            goals = np.array(
                [PAPER_FIX_DIST * np.cos(angle), PAPER_FIX_DIST * np.sin(angle), theta]
            ).T
        elif self.num_agents == 3:
            num_goals = 5
            angle = np.array(
                [np.pi * 2 / 5, np.pi * 4 / 5, np.pi * 6 / 5, np.pi * 8 / 5, 0]
            )
            theta = np.random.uniform(low=[-np.pi], high=[np.pi], size=(num_goals,))
            goals = np.array(
                [PAPER_FIX_DIST * np.cos(angle), PAPER_FIX_DIST * np.sin(angle), theta]
            ).T
        return goals

    def sample_table_goals(self):
        goals = None
        num_goals = 1
        angle = np.random.uniform(low=[-np.pi], high=[np.pi], size=(num_goals,))
        theta = np.random.uniform(low=[-np.pi], high=[np.pi], size=(num_goals,))
        goals = np.array(
            [PAPER_FIX_DIST * np.cos(angle), PAPER_FIX_DIST * np.sin(angle), theta]
        ).T
        return goals

    def sample_goals(self, num_goals, make_figure=False, make_table=False):
        goals = np.zeros((num_goals, 3), dtype=np.float32)
        if make_figure:
            goals = self.sample_paper_goals()
        elif make_table:
            goals = self.sample_table_goals()
        else:
            for i in range(num_goals):
                if self.fix_goal:
                    goals[i, :3] = [1, 1, np.pi / 2]
                elif self.fix_dist is not None:
                    goals[i, :3] = self.generate_random_goal(fix_dist=self.fix_dist)
                else:
                    goals[i, :3] = self.generate_random_goal()
        return goals

    def sample_tasks(self, num_tasks, **kargs):
        return self.sample_goals(num_tasks, **kargs)

    def reset_task(self, task):
        self.reset_goal(task)

    def reset_goal(self, goal):
        self.goal = goal

    def reset(self):
        self._state = np.zeros((3,), dtype=np.float32)
        self._state[:3] = self.generate_default_start()
        self._last_value = 0
        self._step_count = 0
        self.cen_change_x=0
        self.cen_change_y=0
        self._total_reward = 0
        self._last_actions = None
        self._last_state = self._state
        obs = self._get_obs()
        return obs

    def _get_obs(self):
        state = self._state.copy()
        xs, ys, gxs, gys, angles, goal_angles = self.get_pos_gpos()
        #print(state,"state")
        #print(angles,goal_angles,"原来")
        #print(self.get_angle_offsets())
        #相对于原来调整了观测的角度
        obs = np.stack([xs, ys, angles]).T
        obs=np.clip(obs, self.obs_low, self.obs_high)

        if self.observable_target:
            obs = np.stack([(gxs-xs), (gys-ys), self.regulate_radians(goal_angles-angles)]).T
            obs = np.clip(obs, self.obs_low, self.obs_high)
            #print(obs,"obs")
        return obs.astype(np.float32)

    def _action_reward(self, actions):
        assert actions is not None
        new_state = self._get_new_state(actions)
        rew = self._get_reward(new_state)
        return rew

    def cosine_similarity(self,x, y, norm=False):
        """ 计算两个向量x和y的余弦相似度 """
        assert len(x) == len(y), "len(x) != len(y)"
        zero_list = [0] * len(x)
        if x == zero_list or y == zero_list:
            return float(1) if x == y else float(0)

        res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
        cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
        return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内

    def _get_reward(self, state=None):
        if state is None:
            state = self._state
        xs, ys, gxs, gys, angles, goal_angles = self.get_pos_gpos(state=state)
        #print("xs",xs,"gxs",gxs)
        dists = np.sqrt(np.square(xs - gxs) + np.square(ys - gys)+np.square(angles- goal_angles))

        #每个agents的奖励是一样的（平均的），这里可以考虑修改
        a=[self.F_X,self.F_Y]
        b=[self.goal[0]-state[0],self.goal[1]-state[1]]
        current_value = -np.mean(dists)+0.3*np.mean(dists)*self.cosine_similarity(a, b)
        #current_value= -dists
        self._last_value = current_value
        reward = current_value
        #print(reward,"rews")
        return reward

    def _clip_actions(self, actions):
        return np.clip(actions, self.action_low, self.action_high)

    def step(self, actions, **kwargs):
        # Book keeping
        actions = self._clip_actions(np.array(actions))
        self._last_state = self._state

        # Get reward
        new_cx, new_cy, new_angle = self._get_new_state(actions)
        rew = self._action_reward(actions)
        #print(rew,"rew")
        #修改rewards
        rews = [rew] * self.num_agents
        #rews= rew

        marginalized_rews = [rew] * self.num_agents
        if "additional_actions" in kwargs.keys():
            acts = kwargs["additional_actions"]
            marginalized_rews = self._get_marginalized_reward(actions, acts=acts)

        # Update state
        self._state = np.array([new_cx, new_cy, new_angle])
        #print(self._state)
        self._step_count += 1
        done = self._step_count > self._max_episode_steps
        dones = [done] * self.num_agents
        obs = self._get_obs()
        self._total_reward += rew
        self._last_actions = actions

        info = {"goal": self.goal, "marginalized_rews": [0] * self.num_agents}
        for i in range(self.num_agents):
            info["marginalized_rews"][i] = np.float32(marginalized_rews[i])
        return obs, rews, dones, info

    def _get_marginalized_reward(self, actions, acts=None):
            #Use additional actions to marginalize out current reward

            #Params
            # acs : (n_agent, n_additional, act_dim)
            #

        debug = False
        rew = self._action_reward(actions)
        rews = [rew] * self.num_agents
        assert len(acts) == self.num_agents
        for agent_i, acts_i in enumerate(acts):
            # Each agent
            marginalized_rews = []
            if debug:
                print(f" Agent({agent_i}) old:{rew:.4f}")
            for a_i, act_i in enumerate(acts_i):
                # Each additional action
                new_a = np.array(actions, copy=True)
                new_a[agent_i] = act_i
                a_value = self._action_reward(new_a)
                if debug:
                    print(
                        f" Agent({agent_i}) Act:{a_i} val:{a_value:.4f}/{len(acts_i)}"
                    )
                marginalized_rews.append(a_value)
            rews[agent_i] -= np.mean(marginalized_rews)
            if debug:
                print(
                    f" Agent({agent_i}) rew:{rews[agent_i]:.4f} mean:{np.mean(marginalized_rews):.4f} std:{np.std(marginalized_rews):.4f}"
                )
        return rews

    def _convert_f_xyr(self, state, actions):
        """
        Return
        : state : environment state (can use past states)
        : F_x   : joint force along x
        : F_y   : joint force along y
        : F_r   : joint force to rotate the stick
        """
        angles = state[2] + self.get_angle_offsets()
        #print(state[2],self.get_angle_offsets())
        if self.centralized:
            F_xs = actions[:, 0] * np.cos(angles[0]) - actions[:, 1] * np.sin(angles[1])
            F_ys = actions[:, 0] * np.sin(angles[0]) + actions[:, 1] * np.cos(angles[1])
            F_rs = actions[:, 1]
        else:
            F_xs = actions[:, 0] * np.cos(angles + actions[:, 1])
            F_ys = actions[:, 0] * np.sin(angles + actions[:, 1])
            F_rs = actions[:, 0] * np.sin(actions[:, 1])
        for i in range(len(F_rs)):
            F_rs[i]=F_rs[i]*(self.get_r_length()[i])

        F_xs, F_ys, F_rs = F_xs.T, F_ys.T, F_rs.T
        return F_xs, F_ys, F_rs

    def _get_new_state(self, actions):

        cx, cy, angle = self._state
        F_xs, F_ys, F_rs = self._convert_f_xyr(self._state, actions)
        F_x, F_y, F_r = np.mean(F_xs), np.mean(F_ys), np.mean(F_rs)

        self.F_X=F_x
        self.F_Y = F_y
        self.F_R = F_r
        #瞬间刹车，不考虑速度
        new_cx = cx + (F_x / self.mass) * self.dt
        new_cy = cy + (F_y / self.mass) * self.dt
        new_angle = self.regulate_radians(angle + (F_r / self.inertia) * self.dt)

        optimal_action = False  # report optimal rew in paper
        if optimal_action:
            ds = (
                (self.max_robot_force * self.num_agents / self.stick_mass) * self.dt / 2
            )
            gx, gy, ga = self.goal
            dx = gx - cx
            dy = gy - cy
            new_cx = cx + ds * dx / (np.sqrt(dx * dx + dy * dy) + 1e-5)
            new_cy = cy + ds * dy / (np.sqrt(dx * dx + dy * dy) + 1e-5)
            new_angle = ga
        #质心改变
        new_cx+=self.cen_change_x
        new_cy+=self.cen_change_y
        return new_cx, new_cy, new_angle

    def regulate_radians(self,angle):
        # takes all radians to the inverval [-np.pi, np.pi)
        return np.mod(angle + np.pi, 2 * np.pi) - np.pi

    def get_pos_gpos(self, state=None):
        """
        Get current position and goal positions
        """
        state = state if state is not None else self._state
        cx, cy, angle = state
        #print(state,"state")
        gx, gy, goal_angle = self.goal

        angles = self.regulate_radians(angle + self.get_angle_offsets())

        goal_angles = self.regulate_radians(goal_angle + self.get_angle_offsets())

        xs = cx + self.get_r_length() * np.cos(angles)
        ys = cy + self.get_r_length() * np.sin(angles)
        gxs = gx + self.get_r_length() * np.cos(goal_angles)
        gys = gy + self.get_r_length() * np.sin(goal_angles)
        return xs, ys, gxs, gys, angles, goal_angles

    def render(
        self,
        title=None,
        n_frame=10,
        show_before=False,
        task_num=0,
        save_viz=False,
        iteration=0,
        **kargs,
    ):
        t_delta = 0.001
        if self.num_agents == 1:
            colors = ["#000075"]
        elif self.num_agents == 2:
            colors = ["#000075", "#e6194B"]
        elif self.num_agents == 3:
            colors = ["#000075", "#e6194B", "#3cb44b"]
        elif self.num_agents == 4:
            colors = ["#000075", "#e6194B", "#3cb44b", "199737"]
        else:
            colors = ["#000075", "#e6194B", "#3cb44b", "199737","199775"]

        curr_state = self._state
        show_state = self._state
        if n_frame != 1:
            show_state = self._last_state

        for i_frame in range(n_frame):
            # Refresh
            if self._fig is None:
                self._fig = plt.figure(figsize=(13, 13), dpi=80)
            else:
                self._fig.clear()
            ax = plt.gca()

            # Object visualization
            cha=(curr_state - show_state)
            cha[2]=self.regulate_radians(cha[2])
            show_state = show_state + (cha) / n_frame
            xs, ys, gxs, gys, angles, _ = self.get_pos_gpos(show_state)

            if self.num_agents == 2:
                object_plt = plt.Line2D(
                    xs,
                    ys,
                    lw=10.0,
                    ls="-",
                    marker=".",
                    color="grey",
                    markersize=0,
                    markerfacecolor="r",
                    markeredgecolor="r",
                    markerfacecoloralt="k",
                    alpha=0.7,
                )
                ax.add_line(object_plt)
            else:
                object_plt = plt.Polygon(
                    list(map(list, zip(xs, ys))), alpha=0.7, color="grey"
                )
                ax.add_line(object_plt)

            for i, c in zip(range(self.num_agents), colors):
                plt.scatter(xs[i], ys[i], c=c, marker="o", s=140, zorder=10, alpha=0.7)

            #质心位置
            cen_x=self.cen_change_x+show_state[0]
            cen_y = self.cen_change_y + show_state[1]
            plt.scatter(cen_x,cen_y,marker="*",c="b")
            gx, gy, _ = self.goal
            g_cen_x = self.cen_change_x + gx
            g_cen_y = self.cen_change_y + gy
            plt.scatter(g_cen_x, g_cen_y, marker="*", c="y")

            # Before adaptation visualization
            if show_before:
                if self.num_agents == 2:
                    object_plt = plt.Line2D(
                        before_xs,
                        before_ys,
                        lw=10.0,
                        ls="-",
                        marker=".",
                        color="grey",
                        markersize=0,
                        markerfacecolor="r",
                        markeredgecolor="r",
                        markerfacecoloralt="k",
                        alpha=0.35,
                    )
                    ax.add_line(object_plt)
                elif self.num_agents == 3:
                    object_plt = plt.Polygon(
                        list(map(list, zip(before_xs, before_ys))),
                        alpha=0.35,
                        color="grey",
                    )
                    ax.add_line(object_plt)
                for i, c in zip(range(self.num_agents), colors):
                    plt.scatter(
                        before_xs[i],
                        before_ys[i],
                        c=c,
                        marker="o",
                        s=140,
                        zorder=10,
                        alpha=0.35,
                    )

            # Goal visualization
            markers = ["$" + s + "$" for s in "ABCDE"[: self.num_agents]]
            sm_marker = 50
            lg_marker = 80
            #if gxs is not None and gys is not None:
            #    gxs += [gxs[0]]
            #    gys += [gys[0]]
            ax.plot(gxs, gys, ":", lw=2, alpha=1, color="grey")
            if self.num_agents == 2:
                goal_plt = plt.Line2D(
                    gxs,
                    gys,
                    lw=10.0,
                    ls="-",
                    marker=".",
                    color="grey",
                    markersize=0,
                    markerfacecolor="r",
                    markeredgecolor="r",
                    markerfacecoloralt="k",
                    alpha=0.3,
                )
            else:
                goal_plt = plt.Polygon(
                    list(map(list, zip(gxs, gys))), alpha=0.3, color="grey"
                )
            ax.add_line(goal_plt)
            for i, c in zip(range(self.num_agents), colors):
                plt.scatter(
                    gxs[i], gys[i], c=c, marker="o", s=140, zorder=10, alpha=0.3
                )

            # Action visualization
            fs = self._last_actions
            F_xs, F_ys, F_rs = [], [], []
            if fs is not None:
                F_xs, F_ys, F_rs = self._convert_f_xyr(self._last_state, fs)
                fs_const = 0.3
                F_xs *= fs_const
                F_ys *= fs_const
                lengths = np.sqrt(np.square(F_xs) + np.square(F_ys))
                for i in range(self.num_agents):
                    plt.arrow(
                        xs[i],
                        ys[i],
                        F_xs[i],
                        F_ys[i],
                        fc=colors[i],
                        ec="none",
                        alpha=0.7,
                        width=0.06,
                        head_width=0.1,
                        head_length=0.2,
                        zorder=8,
                    )
            sns.despine(offset=10, trim=True)
            #ax.set_xlim([-(PAPER_FIX_DIST + 0.75), (PAPER_FIX_DIST + 0.75)])
            #ax.set_ylim([-(PAPER_FIX_DIST + 0.75), (PAPER_FIX_DIST + 0.75)])
            ax.set_xlim([-13, 13])
            ax.set_ylim([-13, 13])
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.axis("off")
            for item in (
                [ax.title, ax.xaxis.label, ax.yaxis.label]
                + ax.get_xticklabels()
                + ax.get_yticklabels()
            ):
                item.set_fontname("Times New Roman")
            if save_viz:
                if not self._fig_folder:
                    self._fig_folder = "figures/stick-{s}-{date:%Y-%m-%d %H:%M:%S}".format(
                        s=self.num_agents, date=datetime.datetime.now()
                    ).replace(
                        " ", "_"
                    )
                    os.makedirs(self._fig_folder, exist_ok=True)
                self._fig.savefig(
                    os.path.join(
                        self._fig_folder,
                        "stick{}_task{}_itr{}_frame{:04d}.png".format(
                            self.num_agents,
                            task_num,
                            iteration,
                            self._step_count * n_frame + i_frame,
                        ),
                    )
                )
            self._fig.show()
            plt.pause(t_delta / n_frame)

    def record_hist(self, done=False, task_num=0, reward=-1):
        if task_num not in self._hist.keys():
            self._hist[task_num] = dict(score=-np.inf, hist=[])
        xs, ys, gxs, gys, _, _ = self.get_pos_gpos()
        self._hist[task_num]["hist"].append((xs, ys, gxs, gys))
        if done:
            if reward > self._hist[task_num]["score"]:
                self._hist[task_num]["score"] = reward
                self._recordings[task_num] = self._hist[task_num]["hist"]
            self._hist[task_num]["hist"] = []

    def make_figure(self, interpolate=1, after=True):
        assert self._recordings is not None
        palatte = sns.color_palette("deep", len(self._recordings.keys()))
        # colors = ["windows blue", "#34495e", "greyish", "faded green", "dusty purple"]
        # palatte = sns.xkcd_palette(colors)
        if self._fig is None:
            self._fig = plt.figure(figsize=(8, 8), dpi=80)
        else:
            self._fig.clear()
        initial_pos = self._recordings[0][0]
        for task_i, record in self._recordings.items():
            for ts in record[-1::-interpolate]:
                xs, ys, gxs, gys = ts
                self._render_pos(xs, ys, gxs, gys, obj_alpha=0.3, color=palatte[task_i])
            print("task {}".format(task_i))
        sns.despine(offset=10, trim=True)
        self._render_pos(
            initial_pos[0],
            initial_pos[1],
            None,
            None,
            obj_alpha=1.0,
            color="k",
            fixed_pos=True,
        )

        ax = self._fig.axes[0]
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontname("Times New Roman")
        self._fig.show()
        plt.savefig(
            "HeavyObject-{}-{}.png".format(
                self.num_agents, "after" if after else "before"
            ),
            bbox_inches="tight",
        )
        plt.pause(100)

    def _render_pos(self, xs, ys, gxs, gys, obj_alpha=1.0, color=None, fixed_pos=False):
        ax = self._fig.gca()
        w = self.stick_mass
        ax.set_xlim([-(PAPER_FIX_DIST + 0.5), (PAPER_FIX_DIST + 0.5)])
        ax.set_ylim([-(PAPER_FIX_DIST + 0.5), (PAPER_FIX_DIST + 0.5)])
        markers = ["$" + s + "$" for s in "ABC"[: self.num_agents]]
        sm_marker = 50
        lg_marker = 80
        xs += [xs[0]]
        ys += [ys[0]]
        ax.plot(xs, ys, "-", lw=2 + w, alpha=obj_alpha, color=color if color else "k")
        for xi, yi, m in zip(xs, ys, markers):
            plt.scatter(
                xi,
                yi,
                marker=m,
                c=color if color else "k",
                s=lg_marker if fixed_pos else sm_marker,
            )
        if gxs and gys:
            if gxs and gys:
                gxs += [gxs[0]]
                gys += [gys[0]]
            ax.plot(gxs, gys, ":", lw=2 + w, alpha=1, color=color if color else "b")
            for gxi, gyi, m in zip(gxs, gys, markers):
                plt.scatter(gxi, gyi, marker=m, c=color if color else "b", s=lg_marker)
