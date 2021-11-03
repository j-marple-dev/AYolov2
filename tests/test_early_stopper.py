"""Test code for early stopper.

- Auther: Haneol Kim
- Contect: hekim@jmarple.ai
"""
import random

from scripts.utils.constants import probably_run
from scripts.utils.torch_utils import EarlyStopping


@probably_run()
def test_early_stopper(p: float = 0.5):
    stopper = EarlyStopping(patience=10)

    for i in range(20):
        is_stop = stopper(i, -i)
        print(f"Epoch: {i}, stop: {is_stop}")


if __name__ == "__main__":
    test_early_stopper()
