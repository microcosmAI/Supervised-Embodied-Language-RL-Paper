import time
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue, Event
from pettingzoo.butterfly import pistonball_v6


class ParallelEnvWrapper:
    def __init__(self, env_create_fn, num_envs):
        self.env_create_fn = env_create_fn
        self.num_envs = num_envs
        self.action_queues = [Queue() for _ in range(num_envs)]
        self.result_queues = [Queue() for _ in range(num_envs)]
        self.done_event = Event()

        # Create and delete a temporary environment to get the observation space
        temp_env = env_create_fn()
        sample_agent = next(iter(temp_env.possible_agents))  # Get any agent's ID
        self.single_observation_space = temp_env.observation_space(sample_agent)
        self.single_action_space = temp_env.action_space(sample_agent)
        self.possible_agents = temp_env.possible_agents
        temp_env.close()

        self.processes = [
            Process(
                target=self.env_worker,
                args=(
                    env_create_fn,
                    self.action_queues[i],
                    self.result_queues[i],
                ),
            )
            for i in range(num_envs)
        ]

    @staticmethod
    def env_worker(env_create_fn, action_queue, result_queue):
        env = env_create_fn()  # Create the environment in the worker process
        env.reset()

        while True:
            action_or_signal = action_queue.get()

            if action_or_signal == "quit":
                break
            if action_or_signal == "reset":
                # Reset the environment and send initial observation
                observations = env.reset()
                result_queue.put((observations, {}, {}, {}, {}))
            else:
                # Process the action
                observations, rewards, terminations, truncations, infos = env.step(
                    action_or_signal
                )
                result_queue.put(
                    (observations, rewards, terminations, truncations, infos)
                )

            if all(terminations.values()) or all(truncations.values()):
                env.reset()

    def start(self):
        for process in self.processes:
            process.start()

    def step(self, actions):
        for action_queue, action in zip(self.action_queues, actions):
            action_queue.put(action)

        results = tuple(
            ([], [], [], [], [])
        )  # observations, rewards, terminations, truncations, infos
        # Shuffle results in corresponding lists
        for result_queue in self.result_queues:
            for result_list, result_item in zip(results, result_queue.get()):
                result_list.append(result_item)
        return results

    def reset(self):
        # Send a reset signal to each environment
        for action_queue in self.action_queues:
            action_queue.put("reset")

        # Collect initial observations from each environment
        initial_observations = []
        for result_queue in self.result_queues:
            initial_observations.append(result_queue.get())

        return initial_observations

    def close(self):
        for action_queue in self.action_queues:
            action_queue.put("quit")
        for process in self.processes:
            process.join()


# Create a function that returns a new environment instance
def env_create_fn():
    return pistonball_v6.parallel_env(
        render_mode=None,
        max_cycles=100,
    )

# Function to benchmark a given number of environments
def benchmark(num_envs, num_episodes, max_steps):
    envs = ParallelEnvWrapper(env_create_fn, num_envs)
    envs.start()

    start_time = time.time()

    for episode in range(num_episodes):
        for step in range(max_steps):
            actions_list = [
                {agent: envs.single_action_space.sample() for agent in envs.possible_agents}
                for _ in range(envs.num_envs)
            ]
            envs.step(actions_list)
            time.sleep(0.01)  # Simulating the time taken for action sampling

        envs.reset()

    total_time = time.time() - start_time
    envs.close()

    samples_generated = num_envs * num_episodes * max_steps
    samples_per_second = samples_generated / total_time
    return samples_per_second

if __name__ == "__main__":
    # Parameters for the benchmark
    num_episodes = 10
    max_steps = 50
    env_counts = [1, 2, 3, 4]  # Different number of environments to test

    # Collecting benchmark data
    results = [benchmark(env_count, num_episodes, max_steps) for env_count in env_counts]

    plt.plot(env_counts, results, marker='o')
    plt.xlabel('Number of Vectorized Environments')
    plt.xticks(env_counts)
    plt.ylabel('Samples per Second')
    plt.ylim(bottom=0)
    plt.title('Impact of Vectorization on Samples Generated per Second')
    plt.grid(True)
    plt.show()