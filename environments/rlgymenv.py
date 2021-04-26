import logging

import numpy as np
import yaml
from gym import spaces

import policyopt

logging.getLogger('gym.core').addHandler(logging.NullHandler())

import importlib
import os
from collections import namedtuple
from os.path import join as pjoin

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

try:
    import pybullet_envs
    import gym
    import sb3_contrib
except ImportError:
    raise ImportError("Cannot import sb3_contrib")


def get_wrapper_class(hyperparams):
    def get_module_name(wrapper_name):
        return ".".join(wrapper_name.split(".")[:-1])

    def get_class_name(wrapper_name):
        return wrapper_name.split(".")[-1]

    if "env_wrapper" in hyperparams:
        wrapper_name = hyperparams.get("env_wrapper")

        if wrapper_name is None:
            return None

        if not isinstance(wrapper_name, list):
            wrapper_names = [wrapper_name]
        else:
            wrapper_names = wrapper_name

        wrapper_classes = []
        wrapper_kwargs = []
        for wrapper_name in wrapper_names:
            if isinstance(wrapper_name, dict):
                wrapper_dict = wrapper_name
                wrapper_name = list(wrapper_dict.keys())[0]
                kwargs = wrapper_dict[wrapper_name]
            else:
                kwargs = {}
            wrapper_module = importlib.import_module(get_module_name(wrapper_name))
            wrapper_class = getattr(wrapper_module, get_class_name(wrapper_name))
            wrapper_classes.append(wrapper_class)
            wrapper_kwargs.append(kwargs)

        def wrap_env(env):
            for wrapper_class, kwargs in zip(wrapper_classes, wrapper_kwargs):
                env = wrapper_class(env, **kwargs)
            return env

        return wrap_env
    else:
        return None


def create_zoo_env(env_id, stats_dir, hyperparams, should_render=False):
    env_wrapper = get_wrapper_class(hyperparams)

    vec_env_cls = DummyVecEnv
    if "Bullet" in env_id and should_render:
        vec_env_cls = SubprocVecEnv

    env = make_vec_env(
        env_id,
        wrapper_class=env_wrapper,
        vec_env_cls=vec_env_cls
    )

    if stats_dir is not None:
        if hyperparams["normalize"]:
            norm_fpath = pjoin(stats_dir, "vecnormalize.pkl")

            if os.path.exists(norm_fpath):
                env = VecNormalize.load(norm_fpath, env)
                env.training = False
                env.norm_reward = False
            else:
                raise ValueError(f"VecNormalize stats {norm_fpath} not found")

    max_episode_steps = gym.make(env_id).spec.max_episode_steps
    Spec = namedtuple("Spec", ["max_episode_steps"])
    env.spec = Spec(max_episode_steps=max_episode_steps)

    return env


def load_saved_hyperparams(stats_path, norm_reward=False):
    config_fpath = pjoin(stats_path, "config.yml")

    with open(config_fpath, "r") as f:
        hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)
    hyperparams["normalize"] = hyperparams.get("normalize", False)

    if hyperparams["normalize"]:
        normalize_kwargs = {"norm_obs": hyperparams["normalize"], "norm_reward": norm_reward}
        hyperparams["normalize_kwargs"] = normalize_kwargs

    return hyperparams


class RLGymSim(policyopt.Simulation):
    def __init__(self, env_name):
        expert_dir = pjoin("/cluster/home/fpirovan/topographic-nn/environments", "experts", env_name)
        # expert_dir = pjoin("/Users/fedepiro/Projects/topographic-nn/environments", "experts", env_name)
        stats_dir = pjoin(expert_dir, env_name)
        hyperparams = load_saved_hyperparams(stats_dir)
        self.env = create_zoo_env(env_name, stats_dir, hyperparams)
        self.env.viewer = None
        self.action_space = self.env.action_space
        self.curr_obs = self.env.reset()
        self.is_done = False

    def step(self, action):
        if isinstance(self.action_space, spaces.Discrete):
            # We encode actions in finite spaces as an integer inside a length-1 array
            # but Gym wants the integer itself
            assert action.ndim == 1 and action.size == 1 and action.dtype in (np.int32, np.int64)
            action = action[0]
        else:
            assert action.ndim == 1 and action.dtype == np.float64

        self.curr_obs, reward, self.is_done, _ = self.env.step(action[None, :])
        return reward

    @property
    def obs(self):
        return self.curr_obs.copy()

    @property
    def done(self):
        return self.is_done

    def draw(self, track_body_name='torso'):
        self.env.render()
        if track_body_name is not None and track_body_name in self.env.model.body_names:
            self.env.viewer.cam.trackbodyid = self.env.model.body_names.index(track_body_name)

    def __del__(self):
        if self.env.viewer:
            self.env.viewer.finish()

    def reset(self):
        self.curr_obs = self.env.reset()
        self.is_done = False


def _convert_space(space):
    '''Converts a rl-gym space to our own space representation'''
    if isinstance(space, spaces.Box):
        assert space.low.ndim == 1 and space.low.shape[0] >= 1
        return policyopt.ContinuousSpace(dim=space.low.shape[0])
    elif isinstance(space, spaces.Discrete):
        return policyopt.FiniteSpace(size=space.n)
    raise NotImplementedError(space)


class RLGymMDP(policyopt.MDP):
    def __init__(self, env_name):
        print('Gym version:', gym.version.VERSION)
        self.env_name = env_name

        tmpsim = self.new_sim()
        self._obs_space = _convert_space(tmpsim.env.observation_space)
        self._action_space = _convert_space(tmpsim.env.action_space)
        self.env_spec = tmpsim.env.spec
        self.gym_env = tmpsim.env

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._action_space

    def new_sim(self, init_state=None):
        assert init_state is None
        return RLGymSim(self.env_name)
