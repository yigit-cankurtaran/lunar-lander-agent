import numpy as np
import os

def graph():
    if not os.path.exists("logs/evaluations.npz"):
        return Exception("no log file, run training")
    else:
        print("hi")


if __name__ == "__main__":
    graph()
