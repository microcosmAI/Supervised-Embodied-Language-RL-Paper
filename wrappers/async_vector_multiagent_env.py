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
        render_mode="human",
        max_cycles=50,
    )


# Test run envs until all done
if __name__ == "__main__":
    num_envs = 2
    num_episodes = 3
    max_steps = 30

    envs = ParallelEnvWrapper(env_create_fn, num_envs)
    envs.start()

    for episode in range(num_episodes):
        for step in range(max_steps):
            actions_list = [
                {agent: envs.single_action_space.sample() for agent in envs.possible_agents}
                for _ in range(envs.num_envs)
            ]
            # Perform a step in each environment with the new actions
            observations, rewards, terminations, truncations, infos = envs.step(
                actions_list
            )

        initial_observations = envs.reset()
        print("Reset")

    envs.close()
