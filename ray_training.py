import time
from dynamics import *
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray import air
from ray import tune
from MuJoCo_Gym.mujoco_rl import MuJoCoRL
import os
import ray

ray.init(num_gpus=1)
xml_files = ["levels/" + file for file in os.listdir("levels/")]
# xml_files = "EmbodiedReference/mujoco/Model1.xml"
agents = ["sender", "receiver"]
config_dict = {"xmlPath":xml_files, "agents":agents, "rewardFunctions":[collision_reward, target_reward], "doneFunctions":[target_done, border_done], "skipFrames":60, "environmentDynamics":[Image, Communication, Reward, Accuracy], "freeJoint":True, "renderMode":False, "maxSteps":512, "agentCameras":True}
algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=4)
    .resources(num_gpus=1)
    .framework("tf2")
    .training(lr=1e-6)
    .environment(env=MuJoCoRL, env_config=config_dict)
#     .build()
)
algo["model"]["fcnet_hiddens"] = [256, 256]
# algo.build()
tune.Tuner(  
    "PPO",
    run_config=air.RunConfig(stop={"episode_reward_mean": 5}),
    param_space=algo.to_dict(),
).fit()