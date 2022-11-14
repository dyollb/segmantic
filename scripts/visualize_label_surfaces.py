from pathlib import Path
from typing import Dict, List

import itk
import numpy as np
import typer
from monai.utils.module import optional_import

from segmantic.image.labels import load_tissue_list

vtk, _ = optional_import("vtk")


def extract_surfaces(
    file_path: Path,
    output_dir: Path,
    tissuelist_path: Path,
    selected_tissues: List[int] = [],
):
    image = itk.imread(f"{file_path}", pixel_type=itk.US)

    tissues: Dict[int, str] = {}
    if tissuelist_path.exists():
        name_id_map = load_tissue_list(tissuelist_path)
        tissues = {id: name for name, id in name_id_map.items()}

    max_label = np.max(itk.array_view_from_image(image))
    if len(selected_tissues) == 0:
        selected_tissues = [id for id in range(1, max_label)]

    vtk_image = itk.vtk_image_from_image(image)
    contouring = vtk.vtkDiscreteFlyingEdges3D()
    contouring.SetInputData(vtk_image)
    for i, label in enumerate(selected_tissues):
        contouring.SetValue(i, label)
    contouring.Update()

    surfaces = []
    for label in selected_tissues:
        name = tissues[label] if label in tissues else f"label_{label:003}"
        print(f"Processing label {label:3d} : {name}")

        threshold = vtk.vtkThreshold()
        threshold.SetInputData(contouring.GetOutput())
        threshold.SetLowerThreshold(label)
        threshold.SetUpperThreshold(label)
        threshold.Update()

        surf = vtk.vtkPolyData()
        surf.SetPoints(threshold.GetOutput().GetPoints())
        surf.SetPolys(threshold.GetOutput().GetCells())

        if surf.GetNumberOfCells() > 0:
            decimate = vtk.vtkDecimatePro()
            decimate.SetInputData(surf)
            decimate.SetTargetReduction(0.8)
            decimate.PreserveTopologyOn()
            decimate.BoundaryVertexDeletionOff()
            decimate.Update()

            surfaces.append(decimate.GetOutput())

            if output_dir:
                writer = vtk.vtkPLYWriter()
                writer.SetInputData(surfaces[-1])
                writer.SetFileTypeToBinary()
                writer.SetFileName(f"{output_dir / name}.ply")
                writer.Update()


if __name__ == "__main__":
    typer.run(extract_surfaces)
