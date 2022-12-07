import torch
from monai.networks import one_hot
from torch.testing import assert_close

from segmantic.transforms.ensemble import SelectBestEnsembled


def test_SelectBestEnsembled():
    # label (e.g. "argmax") data
    input = {
        "pred0": torch.ones(1, 3, 1, 1),
        "pred1": torch.tensor([2, 0, 2]).reshape(1, 3, 1, 1),
        "pred2": torch.tensor([2, 1, 0]).reshape(1, 3, 1, 1),
    }
    expected_value = torch.tensor([2, 1, 0], dtype=torch.float32).reshape(1, 3, 1, 1)
    if torch.cuda.is_available():
        for k in input.keys():
            input[k] = input[k].to(torch.device("cuda:0"))
        expected_value = expected_value.to(torch.device("cuda:0"))

    tr = SelectBestEnsembled(
        keys=["pred0", "pred1", "pred2"],
        output_key="output",
        label_model_dict={1: 0, 2: 1, 0: 2},
    )
    result = tr(input)
    assert_close(result["output"], expected_value)

    # One-Hot data
    input = {k: one_hot(i, num_classes=3, dim=0) for k, i in input.items()}
    expected_value = one_hot(expected_value, num_classes=3, dim=0)

    tr = SelectBestEnsembled(
        keys=["pred0", "pred1", "pred2"],
        output_key="output",
        label_model_dict={1: 0, 2: 1, 0: 2},
    )
    result = tr(input)
    assert_close(result["output"], expected_value)


if __name__ == "__main__":
    test_SelectBestEnsembled()
