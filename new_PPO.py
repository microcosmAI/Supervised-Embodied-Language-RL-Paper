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

# from wrappers.record_episode_statistics import RecordEpisodeStatistics
from wrappers.frame_stack import FrameStack
from wrappers.normalizeObservation import NormalizeObservation
from wrappers.normalizeRewards import NormalizeReward

from dynamics import *
from MuJoCo_Gym.mujoco_rl import MuJoCoRL
from MuJoCo_Gym.wrappers import GymnasiumWrapper, GymWrapper

def parse_args():
    args = {
        "exp_name": os.path.basename(__file__).rstrip(".py"),
        "learning_rate": 3e-4,
        "seed": 1,
        "total_timesteps": 2000000,
        "torch_deterministic": True,
        "cuda": True,
        "track": False,
        "wandb_project_name": "ppo-implementation-details",
        "wandb_entity": None,
        "capture_video": False,
        "num_envs": 1,
        "num_steps": 2048,
        "anneal_lr": True,
        "gae": True,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "num_minibatches": 32,
        "update_epochs": 10,
        "norm_adv": True,
        "clip_coef": 0.2,
        "clip_vloss": True,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
    }
    args["batch_size"] = int(args["num_envs"] * args["num_steps"])
    args["minibatch_size"] = int(args["batch_size"] // args["num_minibatches"])
    return args

def make_env(config_dict):
    window = 5
    env = MuJoCoRL(config_dict=config_dict)
    # env = GymWrapper(env, "receiver")
    # env = FrameStack(env, 4)
    env = NormalizeObservation(env)
    env = NormalizeReward(env)
    env = GymWrapper(env, "sender")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

args = parse_args()

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
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


if __name__ == "__main__":
    args = parse_args()
    run_name = f"TEST__{args['exp_name']}__{args['seed']}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in args.items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.backends.cudnn.deterministic = args['torch_deterministic']

    device = torch.device("cuda" if torch.cuda.is_available() and args['cuda'] else "cpu")

    xml_files = ["levels/" + file for file in os.listdir("levels/")]
    # xml_files = ["levels_obstacles/" + file for file in os.listdir("levels_obstacles/")]
    agents = ["sender"]

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

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [lambda: make_env(config_dict) for i in range(args['num_envs'])]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args['learning_rate'], eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args['num_steps'], args['num_envs']) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args['num_steps'], args['num_envs']) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args['num_steps'], args['num_envs'])).to(device)
    rewards = torch.zeros((args['num_steps'], args['num_envs'])).to(device)
    dones = torch.zeros((args['num_steps'], args['num_envs'])).to(device)
    values = torch.zeros((args['num_steps'], args['num_envs'])).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args['num_envs']).to(device)
    num_updates = args['total_timesteps'] // args['batch_size']

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args['anneal_lr']:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args['learning_rate']
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args['num_steps']):
            global_step += 1 * args['num_envs']
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args['gae']:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args['num_steps'])):
                    if t == args['num_steps'] - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args['gamma'] * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args['gamma'] * args['gae_lambda'] * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args['num_steps'])):
                    if t == args['num_steps'] - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args['gamma'] * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args['batch_size'])
        clipfracs = []
        for epoch in range(args['update_epochs']):
            np.random.shuffle(b_inds)
            for start in range(0, args['batch_size'], args['minibatch_size']):
                end = start + args['minibatch_size']
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args['clip_coef']).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args['norm_adv']:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args['clip_coef'], 1 + args['clip_coef'])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args['clip_vloss']:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args['clip_coef'],
                        args['clip_coef'],
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args['ent_coef'] * entropy_loss + v_loss * args['vf_coef']

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args['max_grad_norm'])
                optimizer.step()

            if args['target_kl'] is not None:
                if approx_kl > args['target_kl']:
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
        writer.add_scalar("charts/send_accuracies", episode_sendAccuracies, global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()