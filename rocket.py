import gymnasium as gym
env = gym.make("LunarLander-v3")
model = PPO("MlpPolicy", env)
