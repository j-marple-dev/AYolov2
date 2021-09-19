"""This module demonstrates how to construct python project.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""


def demo_generate_world(n_repeat: int) -> str:
    """Generate string that 'world' string concatenated n_repeat times.

    Args:
        n_repeat: Number of repeatition.

    Returns:
        n_repeat time concatenated 'world'
    """
    return "world" * n_repeat
