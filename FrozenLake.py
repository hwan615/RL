import gym

env = gym.make('FrozenLake-v1', render_mode="human")
env.action_space.seed(42)
observation, info = env.reset(seed=40)
print(observation)
print(info)

for _ in range(100):
    action = env.action_space.sample() 
    a = env.step(action)
    print(a)

env.close()
