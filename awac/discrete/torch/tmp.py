import os
import pandas as pd


def summarize_cql():
    fnames = os.listdir("logs/online_cql")
    res = []
    for fname in fnames:
        fields = fname.split("_")
        cql_alpha = float(fields[2][1:])
        epsilon = float(fields[3][1:])
        new_buffer = float(fields[4][2])
        df = pd.read_csv(f"logs/online_cql/{fname}", index_col=0)
        reward = df["reward"].iloc[-3:].mean()
        res.append((reward, cql_alpha, epsilon, new_buffer))
    res_df = pd.DataFrame(res, columns=("reward", "cql_alpha", "epsilon", "new_buffer"))
    res_df.to_csv("cql.csv")


def summarize_dqn():
    fnames = os.listdir("logs/online_dqn")
    res = []
    for fname in fnames:
        fields = fname.split("_")
        epsilon = float(fields[3][1:])
        new_buffer = float(fields[4][2])
        df = pd.read_csv(f"logs/online_dqn/{fname}", index_col=0)
        reward = df["reward"].iloc[-3:].mean()
        res.append((reward, epsilon, new_buffer))
    res_df = pd.DataFrame(res, columns=("reward", "epsilon", "new_buffer"))
    res_df.to_csv("dqn.csv")
