from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk
import typer

SKULL_ID = 1
VERT_ID = 2


def evaluate_segmentations(
    input_dir: Path, reference_dir: Path, output_file: Path, input_glob: str = ".nii.gz"
):

    header = (
        "name",
        "manufacturer",
        "tesla",
        "sex",
        "dice (1)",
        "fn (1)",
        "fp (1)",
        "hausdorff (1)",
        "mean hausdorff (1)",
        "dice (2)",
        "fn (2)",
        "fp (2)",
        "hausdorff (2)",
        "mean hausdorff (2)",
    )
    stats: list[tuple] = []

    for input_file in input_dir.glob(input_glob):
        name = input_file.name.replace(".nii.gz", "")
        ref_file = reference_dir / input_file.name
        if not ref_file.exists():
            continue

        labels = sitk.ReadImage(input_file, sitk.sitkUInt8)
        ref = sitk.ReadImage(ref_file, sitk.sitkUInt8)

        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
        overlap_measures_filter.Execute(ref, labels)

        hausdorff_filter = sitk.HausdorffDistanceImageFilter()

        vals: list[float] = []
        for id in (SKULL_ID, VERT_ID):

            dice = overlap_measures_filter.GetDiceCoefficient(id)
            false_neg = overlap_measures_filter.GetFalseNegativeError(id)
            false_pos = overlap_measures_filter.GetFalsePositiveError(id)

            hausdorff_filter.Execute(ref == id, labels == id)
            hausdorff = hausdorff_filter.GetHausdorffDistance()
            mean_hausdorff = hausdorff_filter.GetAverageHausdorffDistance()
            vals.extend([dice, false_neg, false_pos, hausdorff, mean_hausdorff])

        name, manufacturer, tesla, age, sex = name.split("_")
        current_stats = (name, manufacturer, tesla, sex, *vals)
        stats.append(current_stats)

    if len(stats) == 0:
        raise RuntimeError("No pairs found. Nothing to evaluate")

    with open(output_file, "w") as f:
        print(", ".join(header), file=f)
        for v in stats:
            print(", ".join(v), file=f)


if __name__ == "":
    typer.run(evaluate_segmentations)
