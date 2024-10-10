import quad
import time
import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import torch as th
import os
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps


render_mode = 'headless'

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[256, 256, 16], vf=[256, 256, 16])) 

batch_size = 32*16 
lr_rate = 0.00033 
n_steps=1024
gamma = 0.97 
max_grad_norm = 0.7 
vf_coef = 0.3
ent_coef = 2.001e-05 

dir_weights = "./weights.zip"

def make_env(env_id: str,render_mode, rank: int=0, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gymnasium.Env:
        env = gymnasium.make('CustomQuad-v1', render_mode=render_mode)
        env.reset(seed=seed + rank)
        env = Monitor(env)
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    env=SubprocVecEnv([make_env('CustomQuad-v1',render_mode='human',seed=15)])
    start_time = time.time()
    model = PPO.load(dir_weights,env, learning_rate=lr_rate, n_epochs=10, gamma=gamma, max_grad_norm=max_grad_norm, vf_coef=vf_coef, ent_coef= ent_coef, verbose=1, batch_size=batch_size, policy_kwargs=policy_kwargs, n_steps=n_steps)
    obs = env.reset()
    while True:
        action, _states = model.predict(obs,deterministic=True)
        obs, rewards, dones, info = env.step(action)