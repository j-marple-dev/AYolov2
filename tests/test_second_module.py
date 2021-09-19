"""Unit testing for second_module/second_module.py.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""

from scripts.second_module.second_module import demo_generate_world


def test_demo_generate_world() -> None:
    """Unit test for demo_generate_world function."""

    result1 = demo_generate_world(1)
    result2 = demo_generate_world(2)
    result3 = demo_generate_world(3)

    assert result1 == "world"
    assert result2 == "world" * 2
    assert result3 == "world" * 3


if __name__ == "__main__":
    test_demo_generate_world()
