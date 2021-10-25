"""Test code for early stopper.

- Auther: Haneol Kim
- Contect: hekim@jmarple.ai
"""
import random

from scripts.utils.torch_utils import EarlyStopping


def test_early_stopper(p: float = 0.5):
    if random.random() > p:
        return

    stopper = EarlyStopping(patience=10)

    for i in range(20):
        is_stop = stopper(i, -i)
        print(f"Epoch: {i}, stop: {is_stop}")


if __name__ == "__main__":
    test_early_stopper()
