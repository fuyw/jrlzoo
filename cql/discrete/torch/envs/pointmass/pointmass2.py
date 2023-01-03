import gym
import pickle
import scipy.sparse.csgraph

import networkx as nx
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from collections import deque


WALLS = {
    "Small":
    np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    "Cross":
    np.array([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0],
              [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0]]),
    "FourRooms":
    np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]),
    "Spiral5x5":
    np.array([[0, 0, 0, 0, 0], [0, 1, 1, 1, 1], [0, 1, 0, 0, 1],
              [0, 1, 1, 0, 1], [0, 0, 0, 0, 1]]),
    "Spiral7x7":
    np.array([[1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0],
              [1, 0, 1, 1, 1, 1, 0], [1, 0, 1, 0, 0, 1, 0],
              [1, 0, 1, 1, 0, 1, 0], [1, 0, 0, 0, 0, 1, 0],
              [1, 1, 1, 1, 1, 1, 0]]),
    "Spiral9x9":
    np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1],
              [0, 1, 0, 0, 0, 0, 0, 0, 1], [0, 1, 0, 1, 1, 1, 1, 0, 1],
              [0, 1, 0, 1, 0, 0, 1, 0, 1], [0, 1, 0, 1, 1, 0, 1, 0, 1],
              [0, 1, 0, 0, 0, 0, 1, 0, 1], [0, 1, 1, 1, 1, 1, 1, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 1]]),
    "Spiral11x11":
    np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
              [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
              [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
              [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
              [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
              [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
              [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
              [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]),
    "Maze5x5":
    np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0],
              [1, 1, 1, 1, 0], [1, 1, 1, 1, 0]]),
    "Maze6x6":
    np.array([[0, 0, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0], [0, 0, 1, 0, 1, 1],
              [0, 1, 1, 0, 0, 1], [0, 0, 1, 1, 0, 1], [1, 0, 0, 0, 0, 1]]),
    "Maze11x11":
    np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
              [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
              [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
              [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
              [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
              [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
              [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
              [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
              [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    "Tunnel":
    np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
              [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
              [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
              [0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
              [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
              [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
              [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
              [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0],
              [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
              [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
              [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
              [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
              [0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
              [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
              [0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
              [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0],
              [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
               0]]),
    "U":
    np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 1, 0],
              [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0]]),
    "Tree":
    np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
    ]),
    "UMulti":
    np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]),
    "FlyTrapSmall":
    np.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    ]),
    "FlyTrapBig":
    np.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    ]),
    "Galton":
    np.array([
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0
        ],
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0
        ],
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0
        ],
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0
        ],
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0
        ],
        [
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0
        ],
        [
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 0, 0, 0
        ],
        [
            0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1,
            1, 0, 0, 0, 0, 0, 0
        ],
        [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0
        ],
        [
            0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1,
            0, 0, 0, 0, 0, 1, 0
        ],
        [
            0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1,
            0, 0, 0, 0, 0, 1, 0
        ],
        [
            0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
            1, 1, 1, 1, 1, 1, 0
        ],
        [
            0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
            0, 0, 1, 0, 0, 0, 0
        ],
    ]),
}


ACT_DICT = {
    0: [0., 0.],
    1: [0., -1.],
    2: [0., 1.],
    3: [-1., 0.],
    4: [1., 0.],
}


def resize_walls(walls, factor):
    """Increase the environment by rescaling.
  
    Args:
        walls: 0/1 array indicating obstacle locations.
        factor: (int) factor by which to rescale the environment.
    """
    (height, width) = walls.shape
    row_indices = np.array([i for i in range(height) for _ in range(factor)])
    col_indices = np.array([i for i in range(width) for _ in range(factor)])
    walls = walls[row_indices]
    walls = walls[:, col_indices]
    assert walls.shape == (factor * height, factor * width)
    return walls


class Pointmass2(gym.Env):
    """Abstract class for 2D navigation environments."""
    def __init__(self,
                 difficulty=0,
                 dense_reward=False,
                 action_noise=0.5):
        """Initialize the point environment.

        Args:
            walls: (str) name of one of the maps defined above.
            resize_factor: (int) Scale the map by this factor.
            action_noise: (float) Standard deviation of noise to add to actions.
                          Use 0 to add no noise.
        """
        self.fig = plt.figure()

        self.action_dim = self.ac_dim = 2
        self.observation_dim = self.obs_dim = 2
        self.env_name = "pointmass"
        self.is_gym = True

        if difficulty == 0:
            walls = "Maze5x5"
            resize_factor = 2
            self.fixed_start = np.array([0.5, 0.5]) * resize_factor
            self.fixed_goal = np.array([4.5, 4.5]) * resize_factor
            self._max_episode_steps = 50

        elif difficulty == 1:
            walls = "Maze6x6"
            resize_factor = 1
            self.fixed_start = np.array([0.5, 0.5]) * resize_factor
            self.fixed_goal = np.array([1.5, 5.5]) * resize_factor
            self._max_episode_steps = 150

        elif difficulty == 2:
            walls = "FourRooms"
            resize_factor = 2
            self.fixed_start = np.array([1.0, 1.0]) * resize_factor
            self.fixed_goal1 = np.array([10.0, 10.0]) * resize_factor
            self.fixed_goal2 = np.array([1.0, 8.0]) * resize_factor
            self._max_episode_steps = 100
 
        else:
            print("Invalid difficulty setting")

        if resize_factor > 1:
            self._walls = resize_walls(WALLS[walls], resize_factor)
        else:
            self._walls = WALLS[walls]

        (height, width) = self._walls.shape
        self._apsp = self._compute_apsp(self._walls)

        self._height = height
        self._width = width
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self._height, self._width]),
            dtype=np.float64)

        self.dense_reward = dense_reward
        self.num_actions = 5
        self.epsilon = resize_factor
        self.action_noise = action_noise

        self.obs_vec = []
        self.obs_queue = deque(maxlen=20)
        self.difficulty = difficulty

        self.num_runs = 0
        self.reset()

    def reset(self):
        self.timesteps_left = self._max_episode_steps
        if len(self.obs_vec) > 0:
            self.obs_queue.append(self.obs_vec) 
        self.obs_vec = [self._normalize_obs(self.fixed_start.copy())]
        self.state = self.fixed_start.copy()
        self.num_runs += 1
        return self._normalize_obs(self.state.copy())

    def simulate_step(self, state, action):
        num_substeps = 10
        dt = 1.0 / num_substeps
        num_axis = len(action)
        for _ in np.linspace(0, 1, num_substeps):
            for axis in range(num_axis):
                new_state = state.copy()
                new_state[axis] += dt * action[axis]

                if not self._is_blocked(new_state):
                    state = new_state
        return state

    def _discretize_state(self, state, resolution=1.0):
        (i, j) = np.floor(resolution * state).astype(np.int)
        # Round down to the nearest cell if at the boundary.
        if i == self._height:
            i -= 1
        if j == self._width:
            j -= 1
        return (i, j)

    def _normalize_obs(self, obs):
        return np.array(
            [obs[0] / float(self._height), obs[1] / float(self._width)])

    def _unnormalize_obs(self, obs):
        return np.array(
            [obs[0] * float(self._height), obs[1] * float(self._width)])

    def _is_blocked(self, state):
        if not self.observation_space.contains(state):
            return True
        (i, j) = self._discretize_state(state)
        return (self._walls[i, j] == 1)

    def step(self, action):
        self.timesteps_left -= 1

        if isinstance(action, np.ndarray):
            action = action.item()

        action = np.array(ACT_DICT[action])
        action = np.random.normal(action, self.action_noise)
        self.state = self.simulate_step(self.state, action)

        dist1 = np.linalg.norm(self.state - self.fixed_goal1)
        dist2 = np.linalg.norm(self.state - self.fixed_goal2)
        dist = min(dist1, dist2)
        done = (dist < self.epsilon) or (self.timesteps_left == 0)
        ns = self._normalize_obs(self.state.copy())
        self.obs_vec.append(ns.copy())

        if self.dense_reward:
            reward = -dist
        else:
            reward = int(dist < self.epsilon) - 1

        return ns, reward, done, {}

    @property
    def walls(self):
        return self._walls

    @property
    def goal1(self):
        return self._normalize_obs(self.fixed_goal1.copy())

    @property
    def goal2(self):
        return self._normalize_obs(self.fixed_goal2.copy())

    def _compute_apsp(self, walls):
        (height, width) = walls.shape
        g = nx.Graph()
        # Add all the nodes
        for i in range(height):
            for j in range(width):
                if walls[i, j] == 0:
                    g.add_node((i, j))

        # Add all the edges
        for i in range(height):
            for j in range(width):
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == dj == 0: continue  # Don"t add self loops
                        if i + di < 0 or i + di > height - 1:
                            continue  # No cell here
                        if j + dj < 0 or j + dj > width - 1:
                            continue  # No cell here
                        if walls[i, j] == 1:
                            continue  # Don"t add edges to walls
                        if walls[i + di, j + dj] == 1:
                            continue  # Don"t add edges to walls
                        g.add_edge((i, j), (i + di, j + dj))

        # dist[i, j, k, l] is path from (i, j) -> (k, l)
        dist = np.full((height, width, height, width), np.float("inf"))
        for ((i1, j1), dist_dict) in nx.shortest_path_length(g):
            for ((i2, j2), d) in dist_dict.items():
                dist[i1, j1, i2, j2] = d

        return dist

    def save_trajectories(self):
        obs_vecs = [np.array(i) for i in self.obs_queue]
        return obs_vecs

    def plot_trajectories(self, trajs1, trajs2, fname):
        self.plot_walls()
        obs_vecs1 = trajs1
        obs_vecs2 = trajs2
        goal1, goal2 = self.goal1, self.goal2

        for obs_vec in obs_vecs1:
            plt.plot(obs_vec[1:, 0], obs_vec[1:, 1], "m-o", alpha=0.05)

        for obs_vec in obs_vecs2:
            plt.plot(obs_vec[1:, 0], obs_vec[1:, 1], "b-o", alpha=0.05)

        plt.plot([0.09090909], [0.09090909], "m-o", label="DQN", alpha=0.3)
        plt.plot([0.09090909], [0.09090909], "b-o", label="CQL", alpha=0.3)

        plt.scatter([obs_vec[0, 0]], [obs_vec[0, 1]],
                    marker="+",
                    color="red",
                    s=100,
                    label="start")
        plt.scatter([goal1[0]], [goal1[1]],
                    marker="*",
                    color="green",
                    s=100,
                    label="goal1")
        plt.scatter([goal2[0]], [goal2[1]],
                    marker="*",
                    color="green",
                    s=100,
                    label="goal2")
        plt.legend(loc="lower right")
        plt.savefig(f"{fname}.png", dpi=360)
        plt.close()

    def plot_trajectory(self, fname):
        self.plot_walls()
        obs_vec, goal1, goal2 = np.array(self.obs_vec), self.goal1, self.goal2
        plt.plot(obs_vec[:, 0], obs_vec[:, 1], "b-o", alpha=0.3)
        plt.scatter([obs_vec[0, 0]], [obs_vec[0, 1]],
                    marker="+",
                    color="red",
                    s=100,
                    label="start")
        plt.scatter([obs_vec[-1, 0]], [obs_vec[-1, 1]],
                    marker="+",
                    color="orange",
                    s=100,
                    label="end")
        plt.scatter([goal1[0]], [goal1[1]],
                    marker="*",
                    color="green",
                    s=100,
                    label="goal1")
        plt.scatter([goal2[0]], [goal2[1]],
                    marker="*",
                    color="green",
                    s=100,
                    label="goal2")
        plt.legend(loc="lower right")
        plt.savefig(f"{fname}.png", dpi=360)
        plt.close()

    def plot_walls(self, walls=None):
        if walls is None:
            walls = self._walls.T
        (height, width) = walls.shape
        for (i, j) in zip(*np.where(walls)):
            x = np.array([j, j + 1]) / float(width)
            y0 = np.array([i, i]) / float(height)
            y1 = np.array([i + 1, i + 1]) / float(height)
            plt.fill_between(x, y0, y1, color="grey")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xticks([])
        plt.yticks([])


def refresh_path():
    path = dict()
    path["observations"] = []
    path["actions"] = []
    path["next_observations"] = []
    path["terminals"] = []
    path["rewards"] = []
    return path
