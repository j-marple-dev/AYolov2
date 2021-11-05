import os

from create_swa_model import create_swa_model


def test_create_swa_model(
    force: bool = False,
    model_dir: str = "tests/res/weights",
    model_name: str = "swa.pt",
    best_num: int = 1,
) -> None:
    if not force:
        return

    create_swa_model(model_dir=model_dir, model_name=model_name, best_num=best_num)
    model_path = os.path.join(model_dir, model_name)
    assert os.path.isfile(model_path)
    os.remove(model_path)


if __name__ == "__main__":
    test_create_swa_model()
