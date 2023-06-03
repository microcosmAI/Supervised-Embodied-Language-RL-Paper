from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.n_episodes = 10  # Change this value as per your requirement

    def _on_step(self):
        # Calculate and store the episode reward and length
        if self.n_episodes > 0 and self.num_timesteps > 0 and self.num_timesteps % self.n_episodes == 0:
            mean_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            mean_length = sum(self.episode_lengths) / len(self.episode_lengths)
            self.episode_rewards = []
            self.episode_lengths = []
            self.logger.record("train/mean_reward", mean_reward)  # Log mean reward to TensorBoard
            self.logger.record("train/mean_length", mean_length)  # Log mean episode length to TensorBoard

        return True

    def _on_rollout_end(self) -> None:
        if self.n_episodes > 0 and self.num_timesteps > 0 and self.num_timesteps % self.n_episodes == 0:
            self.episode_lengths.append(self.num_timesteps - self._last_observation)