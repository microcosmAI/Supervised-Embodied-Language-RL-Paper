import time
from dynamics import *
from MuJoCo_Gym.mujoco_rl import MuJoCoRL
from MuJoCo_Gym.wrappers import GymnasiumWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.experimental.wrappers import NormalizeObservationV0
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv 
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        for env in self.training_env.envs:
            if isinstance(env.env.env.environment.environment_dynamics[3], Accuracy):
                accuacies = env.env.env.environment.environment_dynamics[3].accuracies
                variances = env.env.env.environment.environment_dynamics[3].variances
                sendAccuracies = env.env.env.environment.environment_dynamics[3].sendAccuracies
                if len(accuacies) > 50:
                    value = sum(accuacies[-50:]) / 50
                    self.logger.record("accuracy", value)

                    variance = 1 - abs(sum(variances[-50:]) / 50)
                    self.logger.record("variance", variance)
                if len(sendAccuracies) > 16000:
                    value = sum(sendAccuracies[-16000:]) / 16000
                    self.logger.record("send accuracy", value)
        return True
    

xml_files = ["levels/" + file for file in os.listdir("levels/")][0]
window = 5
learning_rate = 1e-6
network = [256, 128]
batch_size = 32
device = "cuda"
timesteps = 4000000



agents = ["sender"]
config_dict = {"xmlPath":xml_files, "agents":agents, "rewardFunctions":[collision_reward, target_reward], "doneFunctions":[target_done, border_done], "skipFrames":5, "environmentDynamics":[Image, Reward, Communication, Accuracy], "freeJoint":True, "renderMode":False, "maxSteps":512, "agentCameras":True}
policy_kwargs = dict(
                net_arch=dict(pi=network, vf=network),
)
# env = MuJoCoRL(config_dict=config_dict)
# env = GymnasiumWrapper(env, "sender")
# env = NormalizeObservationV0(FrameStack(env, window))
# name = "PPO Sender"
# model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, batch_size=batch_size, device=device, tensorboard_log="./results/", learning_rate=learning_rate, stats_window_size=200)
# model.learn(timesteps, tb_log_name=name, progress_bar=True, callback=TensorboardCallback())
# model.save("models/Sender" + str(int(time.time())))



# agents = ["receiver"]
# config_dict = {"xmlPath":xml_files, "agents":agents, "rewardFunctions":[collision_reward, target_reward, turn_reward], "doneFunctions":[target_done, border_done, turn_done], "skipFrames":5, "environmentDynamics":[Image, Reward, Communication, Accuracy, Reward], "freeJoint":False, "renderMode":False, "maxSteps":1024, "agentCameras":True}
# policy_kwargs = dict(
#                 net_arch=dict(pi=network, vf=network),
# )
# envs = []

def createEnv():
    env = MuJoCoRL(config_dict=config_dict)
    env = GymnasiumWrapper(env, "sender")
    env = NormalizeObservationV0(FrameStack(env, window))
    return env

envs = [lambda: createEnv() for i in range(6)]
envs = DummyVecEnv(envs)
timesteps = 3000000
name = "PPO Sender"
model = PPO("MlpPolicy", envs, policy_kwargs=policy_kwargs, verbose=1, batch_size=batch_size, device=device, tensorboard_log="./results/", learning_rate=learning_rate, stats_window_size=200)
model.learn(timesteps, tb_log_name=name, progress_bar=True, callback=TensorboardCallback())
# model.save("models/Receiver" + str(int(time.time())))