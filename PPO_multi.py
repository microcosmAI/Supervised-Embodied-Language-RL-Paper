from MuJoCo_Gym.mujoco_rl import MuJoCoRL
from MuJoCo_Gym.wrappers import GymnasiumWrapper, GymWrapper
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.experimental.wrappers import NormalizeObservationV0
from dynamics import *
import argparse
import os
import random
import time
from distutils.util import strtobool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from wrappers.record_episode_statistics import RecordEpisodeStatistics
from wrappers.frame_stack import FrameStack
from wrappers.normalizeObservation import NormalizeObservation
from wrappers.normalizeRewards import NormalizeReward

from progressbar import progressbar


def make_env(config_dict):
    def thunk():
        window = 5
        env = MuJoCoRL(config_dict=config_dict)
        # env = GymWrapper(env, "receiver")
        # env = FrameStack(env, 4)
        env = NormalizeObservation(env)
        env = NormalizeReward(env)
        # env = RecordEpisodeStatistics(env)
        # env = gym.wrappers.ClipAction(env)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.NormalizeReward(env)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
class Buffer():

    def __init__(self, num_steps, envs, num_envs, device):
        self.obs = torch.zeros((num_steps, num_envs) + envs.observation_space.shape).to(device)
        self.actions = torch.zeros((num_steps, num_envs) + envs.action_space.shape).to(device)
        self.logprobs = torch.zeros((num_steps, num_envs)).to(device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(device)
        self.dones = torch.zeros((num_steps, num_envs)).to(device)
        self.values = torch.zeros((num_steps, num_envs)).to(device)

def update_agent(agent, buffer, optimizer, next_obs, next_done, env, batch_size, update_epochs, minibatch_size, clip_coef, vf_coef, ent_coef, max_grad_norm, target_kl, clip_vloss, norm_adv, gae_lambda, gae, gamma):

    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        if gae:
            advantages = torch.zeros_like(buffer.rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - buffer.dones[t + 1]
                    nextvalues = buffer.values[t + 1]
                delta = buffer.rewards[t] + gamma * nextvalues * nextnonterminal - buffer.values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + buffer.values
        else:
            returns = torch.zeros_like(buffer.rewards).to(device)
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = next_value
                else:
                    nextnonterminal = 1.0 - buffer.dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = buffer.rewards[t] + gamma * nextnonterminal * next_return
            advantages = returns - buffer.values

    # flatten the batch
    b_obs = buffer.obs.reshape((-1,) + env.observation_space.shape)
    b_logprobs = buffer.logprobs.reshape(-1)
    b_actions = buffer.actions.reshape((-1,) + env.action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = buffer.values.reshape(-1)

    # Optimizing the policy and value network
    b_inds = np.arange(batch_size)
    clipfracs = []
    for epoch in range(update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

        if target_kl is not None:
            if approx_kl > target_kl:
                break

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

def initialize_agent(env, device, learning_rate):
    agent = Agent(env).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    return agent, optimizer

def get_action_and_update_buffer(agent, obs, buffer, step):
    with torch.no_grad():
        action, logprob, _, value = agent.get_action_and_value(obs)
        buffer.values[step] = value.flatten()
    buffer.actions[step] = action
    buffer.logprobs[step] = logprob
    return action

def reset_environment(env, device):
    next_obs, infos = env.reset()
    next_obs = {k: torch.Tensor(v).unsqueeze(0).to(device) for k, v in next_obs.items()}
    return next_obs, infos

if __name__ == "__main__":

    # Experiment settings
    # exp_name = os.path.basename(__file__).rstrip(".py")
    exp_name = "Sender box"
    # xml_files = "levels/Model1.xml"
    xml_files = ["levels/" + file for file in os.listdir("levels/")]
    # xml_files = ["levels_obstacles/" + file for file in os.listdir("levels_obstacles/")]
    agents = ["sender", "receiver"]
    # agents = ["sender"]
    learning_rate = 1e-5
    seed = 1
    # total_timesteps = 20000000
    total_timesteps = 2000000
    torch_deterministic = True
    cuda = False
    mps = False
    track = False
    wandb_project_name = "ppo-implementation-details"
    wandb_entity = None
    capture_video = False

    # Algorithm-specific arguments
    num_envs = 1
    num_steps = 2048
    anneal_lr = True
    gae = True
    gamma = 0.99
    gae_lambda = 0.95
    num_minibatches = 128
    update_epochs = 10
    norm_adv = True
    clip_coef = 0.2
    clip_vloss = True
    ent_coef = 0.0
    vf_coef = 0.5
    max_grad_norm = 0.5
    target_kl = None
    store_freq = 20

    # Calculate derived variables
    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)

    torch.set_default_dtype(torch.float32)

    run_name = f"{exp_name}__{seed}__{int(time.time())}"
    if track:
        import wandb

        wandb.init(
            project=wandb_project_name,
            entity=wandb_entity,
            sync_tensorboard=True,
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")

    writer.add_text("environment/level_number", str(len(xml_files)), 0)
    writer.add_text("environment/agents", ', '.join(agents), 0)
    writer.add_text("hyperparameters/learning_rate", str(learning_rate), 0)
    writer.add_text("hyperparameters/network_size", ', '.join(str(e) for e in [512, 256]), 0)
    writer.add_text("hyperparameters/batch", str(minibatch_size), 0)

    config_dict = {"xmlPath":xml_files, 
                   "agents":agents, 
                   "rewardFunctions":[collision_reward, target_reward], 
                   "doneFunctions":[target_done, border_done], 
                   "skipFrames":5,
                   "environmentDynamics":[Image, Reward, Communication, Accuracy],
                   "freeJoint":True,
                   "renderMode":False,
                   "maxSteps":1024,
                   "agentCameras":True}
    # config_dict = {"xmlPath":xml_files, "agents":agents, "rewardFunctions":[collision_reward, target_reward, turn_reward], "doneFunctions":[target_done, border_done, turn_done], "skipFrames":1, "environmentDynamics":[Image, Communication, Accuracy, Reward], "freeJoint":False, "renderMode":True, "maxSteps":2000, "agentCameras":True, "tensorboard_writer":None}

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    # device = torch.device("mps" if torch.backends.mps.is_available() and mps else "cpu")

    # env = make_env(config_dict)()
    # env.reset()
    # env.step(env.action_space.sample())

    env = make_env(config_dict)()
    obs, infos = env.reset()

    sender, sender_optimizer = initialize_agent(env, device, learning_rate)
    receiver, receiver_optimizer = initialize_agent(env, device, learning_rate)

    buffer_sender = Buffer(num_steps, env, num_envs, device)
    buffer_receiver = Buffer(num_steps, env, num_envs, device)

    global_step = 0
    start_time = time.time()
    next_obs, infos = reset_environment(env, device)

    next_done = {"sender": torch.zeros(num_envs).to(device), "receiver": torch.zeros(num_envs).to(device)}

    num_updates = total_timesteps // batch_size
    train_start = time.time()

    epoch_lengths = []
    current_length = 0

    for update in progressbar(range(1, num_updates + 1), redirect_stdout=True):
    # for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * learning_rate
            sender_optimizer.param_groups[0]["lr"] = lrnow
            receiver_optimizer.param_groups[0]["lr"] = lrnow
        
        epoch_rewards = {"sender":0, "receiver":0}
        current_rewards = {"sender":[], "receiver":[]}
        variances = {"sender":[], "receiver":[]}
        epoch_runs = 0
        episode_accuracies = 0
        episode_sendAccuracies = 0
        for step in range(0, num_steps):
            global_step += 1 * num_envs
            current_length += 1
            buffer_sender.obs[step] = next_obs["sender"]
            buffer_receiver.obs[step] = next_obs["receiver"]


            sender_action = get_action_and_update_buffer(sender, next_obs["sender"], buffer_sender, step)
            receiver_action = get_action_and_update_buffer(receiver, next_obs["receiver"], buffer_receiver, step)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, info = env.step({"sender": sender_action.cpu().numpy()[0], "receiver": receiver_action.cpu().numpy()[0]})
            current_rewards["sender"].append(reward["sender"])
            current_rewards["receiver"].append(reward["receiver"])
            next_obs = {"sender": torch.Tensor(next_obs["sender"]).unsqueeze(0).to(device), "receiver": torch.Tensor(next_obs["receiver"]).unsqueeze(0).to(device)}

            if terminations["sender"] or terminations["receiver"] or truncations["sender"] or truncations["receiver"]:
                next_obs, infos = reset_environment(env, device)
                epoch_rewards["sender"] += sum(current_rewards["sender"])
                epoch_rewards["receiver"] += sum(current_rewards["receiver"])

                epoch_lengths.append(current_length)
                current_length = 0

                dynamic = env.env.env.environment_dynamics[3]

                if len(dynamic.sendAccuracies) > 512:
                    episode_sendAccuracies = sum(dynamic.sendAccuracies[-512:]) / 512
                    del dynamic.sendAccuracies[:-513]
                    writer.add_scalar("charts/sender/accuracies", episode_sendAccuracies, global_step)

                if len(dynamic.accuracies) > 4:
                    window = min(15, len(dynamic.accuracies))
                    episode_accuracies = sum(dynamic.accuracies[-1 * window:]) / window
                    writer.add_scalar("charts/receiver/accuracies", episode_accuracies, global_step)
                    if window == 15:
                        del dynamic.accuracies[:-16]

                if len(dynamic.variances) > 4:
                    window = min(15, len(dynamic.variances))
                    current_variance = sum(dynamic.variances[-1 * window:]) / window
                    writer.add_scalar("charts/receiver_variance", current_variance, global_step)
                    if window == 15:
                        del dynamic.variances[:-16]

                if len(epoch_lengths) > 3:
                    window = min(10, len(epoch_lengths))
                    epoch_length = sum(epoch_lengths[-1 * window:]) / window
                    writer.add_scalar("charts/episodic_length", epoch_length, global_step)
                    if window == 10:
                        del epoch_lengths[:-11]
                epoch_runs += 1
            
            buffer_sender.rewards[step] = torch.tensor(reward["sender"]).to(device).view(-1)
            buffer_receiver.rewards[step] = torch.tensor(reward["receiver"]).to(device).view(-1)
            next_done = {"sender": torch.Tensor([terminations["sender"]]).to(device), "receiver": torch.Tensor([terminations["receiver"]]).to(device)}
        if update % store_freq == 0:
            torch.save(sender, "models/model" + str(start_time) + ".pth")
            torch.save(receiver, "models/model" + str(start_time) + ".pth")

        update_agent(sender, buffer_sender, sender_optimizer, next_obs["sender"], next_done["sender"], env, batch_size, update_epochs, minibatch_size, clip_coef, vf_coef, ent_coef, max_grad_norm, target_kl, clip_vloss, norm_adv, gae_lambda, gae, gamma)
        update_agent(receiver, buffer_receiver, receiver_optimizer, next_obs["receiver"], next_done["receiver"], env, batch_size, update_epochs, minibatch_size, clip_coef, vf_coef, ent_coef, max_grad_norm, target_kl, clip_vloss, norm_adv, gae_lambda, gae, gamma)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", sender_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/sender/episodic_return", epoch_rewards["sender"] / epoch_runs, global_step)
        writer.add_scalar("charts/receiver/episodic_return", epoch_rewards["receiver"] / epoch_runs, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)), "Average Reward:", epoch_rewards["sender"] / epoch_runs)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    writer.close()