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
from callback import CustomCallback
# log_dir = "./logs/"
# new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

render_mode = 'headless'

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[256, 256, 16], vf=[256, 256, 16]))

batch_size = 32*8 # 32 is the best
lr_rate = 0.0003 # didnt try it yet 0.00033
n_steps=1024
gamma = 0.97 # Low is better (0.97)
max_grad_norm = 0.7 # was 0.7
vf_coef = 0.3 # was 0.5, 0.3
ent_coef = 0.0 # 0.0 is better
dir_weights ="./saves/save68_6/rl_model_204800_steps.zip"
# dir_weights ="./saves/save71/rl_model_3276800_steps.zip"

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
        env = gymnasium.make('CustomQuad-v0', render_mode=render_mode)
        env.reset(seed=seed + rank)
        env = Monitor(env)
        return env

    set_random_seed(seed)
    return _init

# if render_mode == 'headless':
#     num_core = 12
#     episodes = 60
# if render_mode == 'human':
#     num_core = 1
#     episodes = 1

if __name__ == "__main__":
    # env = SubprocVecEnv([make_env('CustomQuad-v0',render_mode='headless',seed=i) for i in range(15)]+[make_env('CustomQuad-v0',render_mode='human',seed=15)])
    env = SubprocVecEnv([make_env('CustomQuad-v0',render_mode='headless',seed=i) for i in range(8)])
    # env=SubprocVecEnv([make_env('CustomQuad-v0',render_mode='human',seed=15)])
    start_time = time.time()

    # env.env_method('custom_function')

    # , tensorboard_log="./PPO"
    # model = PPO.load(dir_weights, env, learning_rate=lr_rate, n_epochs=10, gamma=gamma, max_grad_norm=max_grad_norm, vf_coef=vf_coef, ent_coef= ent_coef, verbose=1, batch_size=batch_size, policy_kwargs=policy_kwargs, n_steps=n_steps,tensorboard_log="./PPO/saves/save73")
    # model.set_logger(new_logger)
    # , tensorboard_log="./PPO/saves/save13"
    model = PPO("MlpPolicy", env, learning_rate=lr_rate, n_epochs=10, gamma=gamma, max_grad_norm=max_grad_norm, vf_coef=vf_coef, ent_coef= ent_coef, verbose=1, batch_size=batch_size, policy_kwargs=policy_kwargs, n_steps=n_steps,tensorboard_log="./PPO/saves/save101")
    # model = PPO("MlpPolicy", env, learning_rate=lr_rate,ent_coef=ent_coef, verbose=1, policy_kwargs=policy_kwargs, n_steps=n_steps, tensorboard_log="./PPO/saves/save68", vf_coef=vf_coef, gamma=gamma)
    checkpoint_on_event = CheckpointCallback(save_freq=1, save_path="./saves/save101",name_prefix = "rl_model")
    event_callback = EveryNTimesteps(n_steps=1024*8*25, callback=checkpoint_on_event)
    # event_callback = EveryNTimesteps(n_steps=2048*16*10, callback=checkpoint_on_event)
    # callback_on_best=CustomCallback()
    # eval_callback = EvalCallback(env,eval_freq=2048)
    model.learn(total_timesteps=1024*400*8, callback=event_callback)
    print(f"--- {time.time() - start_time} seconds ---")
    # os.system('shutdown -s -t 0')