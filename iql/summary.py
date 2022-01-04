import os
import pandas as pd

files = os.listdir("tmp")
for f in files:
    df = pd.read_csv(f"tmp/{f}/0.txt", sep=" ", header=None, names=["step", "reward"])
    reward = df["reward"].iloc[-10:].mean()
    print(f"{f}:\t{reward:.2f}")

