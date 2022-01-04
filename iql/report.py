import pandas as pd

env_name = "walker2d-medium-expert-v2"
seed = 0

data = pd.read_csv(f"tmp/{env_name}/{seed}.txt", sep=' ', header=None, names=["step", "reward"])
print(f"{env_name} reward = {data.iloc[-10:, 1].mean():.2f}")

