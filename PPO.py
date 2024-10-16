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
from distributions import StateDependentNoiseDistribution

from wrappers.record_episode_statistics import RecordEpisodeStatistics
from wrappers.frame_stack import FrameStack
from wrappers.normalizeObservation import NormalizeObservation
from wrappers.normalizeRewards import NormalizeReward

from progressbar import progressbar


def make_env(config_dict):
    window = 5
    env = MuJoCoRL(config_dict=config_dict)
    # env = GymWrapper(env, "receiver")
    # env = FrameStack(env, 4)
    env = NormalizeObservation(env)
    env = NormalizeReward(env)
    env = GymWrapper(env, "sender")
    env = RecordEpisodeStatistics(env)
    return env


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

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
        self.obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
        self.actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
        self.logprobs = torch.zeros((num_steps, num_envs)).to(device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(device)
        self.dones = torch.zeros((num_steps, num_envs)).to(device)
        self.values = torch.zeros((num_steps, num_envs)).to(device)

def update_agent(agent, buffer, optimizer, batch_size, update_epochs, minibatch_size, clip_coef, vf_coef, ent_coef, max_grad_norm, target_kl, clip_vloss, norm_adv, gae_lambda, gae, gamma):

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
    b_obs = buffer.obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = buffer.logprobs.reshape(-1)
    b_actions = buffer.actions.reshape((-1,) + envs.single_action_space.shape)
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
            # logratio = b_logprobs[mb_inds] - newlogprob
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
            # loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
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

    send_accuracies = []
    for env in envs.envs:
        env_dynamic = env.env.environment.env.env.environment_dynamics[3]
        if isinstance(env_dynamic, Accuracy):
            send_accuracies.append(env_dynamic.sendAccuracies)
    send_accuracies = [item for sublist in send_accuracies for item in sublist]
    if len(send_accuracies) > 0 and len(send_accuracies) > 16000:
        episode_sendAccuracies = sum(send_accuracies[-16000:]) / 16000
    else:
        episode_sendAccuracies = 0

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)
    writer.add_scalar("charts/episodic_return", epoch_rewards / epoch_runs, global_step)
    writer.add_scalar("charts/episodic_length", epoch_lengths / epoch_runs, global_step)
    writer.add_scalar("charts/accuracies", episode_accuracies / epoch_runs, global_step)
    writer.add_scalar("charts/send_accuracies", episode_sendAccuracies, global_step)
    print("SPS:", int(global_step / (time.time() - start_time)), "Average Reward:", epoch_rewards / epoch_runs)
    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


if __name__ == "__main__":

    # Experiment settings
    # exp_name = os.path.basename(__file__).rstrip(".py")
    exp_name = "Sender box"
    # xml_files = "levels_obstacles/Model1.xml"
    xml_files = ["levels/" + file for file in os.listdir("levels/")]
    # xml_files = ["levels_obstacles/" + file for file in os.listdir("levels_obstacles/")]
    agents = ["sender"]
    # agents = ["sender"]
    learning_rate = 3e-4
    seed = 1
    # total_timesteps = 20000000
    total_timesteps = 2000000
    torch_deterministic = True
    cuda = True
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
    num_minibatches = 32
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

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [lambda: make_env(config_dict) for i in range(num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    # agent = torch.load("models/model1695939146.0011158.pth")
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    buffer = Buffer(num_steps, envs, num_envs, device)

    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(num_envs).to(device)
    num_updates = total_timesteps // batch_size
    train_start = time.time()

    for update in progressbar(range(1, num_updates + 1), redirect_stdout=True):
        # Annealing the rate if instructed to do so.
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        epoch_rewards = 0
        epoch_lengths = 0
        epoch_runs = 0
        episode_accuracies = 0
        episode_sendAccuracies = 0
        for step in range(0, num_steps):
            global_step += 1 * num_envs
            buffer.obs[step] = next_obs
            buffer.dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                buffer.values[step] = value.flatten()
            buffer.actions[step] = action
            buffer.logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            buffer.rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for i, item in enumerate(info):
                if "episode" in item.keys():
                    epoch_rewards += item['episode']['r']
                    epoch_lengths += item["episode"]["l"]
                    # episode_accuracies += item["episode"]["a"]
                    # episode_sendAccuracies += item["episode"]["sa"]
                    epoch_runs += 1
                    break
        if update % store_freq == 0:
            torch.save(agent, "models/model" + str(start_time) + ".pth")

        update_agent(agent, buffer, optimizer, batch_size, update_epochs, minibatch_size, clip_coef, vf_coef, ent_coef, max_grad_norm, target_kl, clip_vloss, norm_adv, gae_lambda, gae, gamma)

    envs.close()
    writer.close()