from multiprocessing import Process, Queue, Event
from pettingzoo.butterfly import pistonball_v6


# Create a function that returns a new environment instance
def env_create_fn():
    return pistonball_v6.parallel_env(
        render_mode="human",
        max_cycles=50,
    )


def env_worker(env_create_fn, action_queue, result_queue, done_event):
    env = env_create_fn()  # Create the environment in the worker process
    observations = env.reset()
    while not done_event.is_set():
        actions = action_queue.get()
        observations, rewards, terminations, truncations, infos = env.step(actions)
        result_queue.put((observations, rewards, terminations, truncations, infos))

        if all(terminations.values()) or all(truncations.values()):
            done_event.set()
    env.close()


def main():
    num_envs = 2  # Number of parallel environments
    action_queues = [Queue() for _ in range(num_envs)]
    result_queues = [Queue() for _ in range(num_envs)]
    done_event = Event()

    # Start processes
    processes = [
        Process(
            target=env_worker,
            args=(env_create_fn, action_queues[i], result_queues[i], done_event),
        )
        for i in range(num_envs)
    ]
    for p in processes:
        p.start()

    # Initialize actions and send them to the worker processes
    for action_queue in action_queues:
        env = env_create_fn()  # Create a temporary environment to get action spaces
        actions = {
            agent: env.action_space(agent).sample() for agent in env.possible_agents
        }
        action_queue.put(actions)

    # Main loop for sending actions and receiving results
    while not done_event.is_set():
        for i, result_queue in enumerate(result_queues):
            observations, rewards, terminations, truncations, infos = result_queue.get()

            if not all(terminations.values()):
                env = env_create_fn() # Create a temporary environment to get action spaces
                actions = {
                    agent: env.action_space(agent).sample() for agent in env.possible_agents
                }
                action_queues[i].put(actions)

    # Wait for all processes to finish
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
