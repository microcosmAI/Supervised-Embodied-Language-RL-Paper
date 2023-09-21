from MuJoCo_Gym.mujoco_rl import MuJoCoRL
from MuJoCo_Gym.wrappers import GymnasiumWrapper, GymWrapper
import os
from dynamics import *


# Experiment settings
exp_name = os.path.basename(__file__).rstrip(".py")
xml_files = ["levels_ants/" + file for file in os.listdir("levels_ants/")]
agents = ["receiver"]
learning_rate = 2e-5
seed = 1
total_timesteps = 10000000
# total_timesteps = 10000
torch_deterministic = True
cuda = True
mps = False
track = False
wandb_project_name = "ppo-implementation-details"
wandb_entity = None
capture_video = False

# Algorithm-specific arguments
num_envs = 7
num_steps = 2048
anneal_lr = False
gae = True
gamma = 0.99
gae_lambda = 0.95
num_minibatches = 32
update_epochs = 10
norm_adv = True
clip_coef = 0.2
clip_vloss = True
ent_coef = 0.0
vf_coef = 0.5
max_grad_norm = 0.5
target_kl = None

config_dict = {"xmlPath":xml_files, "agents":agents, "rewardFunctions":[collision_reward, target_reward, turn_reward], "doneFunctions":[target_done, border_done, turn_done], "skipFrames":5, "environmentDynamics":[Image, Communication, Accuracy, Reward], "freeJoint":False, "renderMode":False, "maxSteps":1024, "agentCameras":True, "tensorboard_writer":None}

env = MuJoCoRL(config_dict=config_dict)
env = GymWrapper(env, "receiver")

env.step(env.action_space.sample())
env.environment.export_json()