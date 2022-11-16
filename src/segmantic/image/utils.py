from typing import TYPE_CHECKING

import numpy as np
import SimpleITK as sitk
from monai.utils.module import optional_import

if TYPE_CHECKING:
    import vtk
else:
    vtk, _ = optional_import("vtk")


def array_view_reverse_ordering(x: np.ndarray) -> np.ndarray:
    return x.transpose(np.flip(np.arange(len(x.shape))))


def vtk_image_from_image(
    image: sitk.Image, array_name: str = "Scalars"
) -> "vtk.vtkImageData":
    """Convert a SimpleITK.Image to a vtk.vtkImageData."""
    from vtk.util.numpy_support import numpy_to_vtk

    array = sitk.GetArrayFromImage(image)

    vtk_image = vtk.vtkImageData()
    data_array = numpy_to_vtk(array.reshape(-1))
    data_array.SetNumberOfComponents(image.GetNumberOfComponentsPerPixel())
    data_array.SetName(array_name)
    # Always set Scalars for (future?) multi-component volume rendering
    vtk_image.GetPointData().SetScalars(data_array)
    dim = image.GetDimension()
    spacing = [1.0] * 3
    spacing[:dim] = image.GetSpacing()
    vtk_image.SetSpacing(spacing)
    origin = [0.0] * 3
    origin[:dim] = image.GetOrigin()
    vtk_image.SetOrigin(origin)
    dims = [1] * 3
    dims[:dim] = image.GetSize()
    vtk_image.SetDimensions(dims)
    m3 = vtk.vtkMatrix3x3()
    dc = image.GetDirection()
    for r in range(3):
        m3.SetElement(r, 0, dc[r * 3 + 0])
        m3.SetElement(r, 1, dc[r * 3 + 1])
        m3.SetElement(r, 2, dc[r * 3 + 2])
    vtk_image.SetDirectionMatrix(m3)
    if image.GetDimension() == 3:
        pixel_type: str = image.GetPixelIDTypeAsString()
        if pixel_type.startswith("sitkVector"):
            vtk_image.GetPointData().SetVectors(data_array)
    return vtk_image
