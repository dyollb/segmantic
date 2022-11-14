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
    l_image: sitk.Image, array_name: str = "Scalars"
) -> "vtk.vtkImageData":
    """Convert a SimpleITK.Image to a vtk.vtkImageData."""
    from vtk.util.numpy_support import numpy_to_vtk

    array = sitk.GetArrayFromImage(l_image)

    vtk_image = vtk.vtkImageData()
    data_array = numpy_to_vtk(array.reshape(-1))
    data_array.SetNumberOfComponents(l_image.GetNumberOfComponentsPerPixel())
    data_array.SetName(array_name)
    # Always set Scalars for (future?) multi-component volume rendering
    vtk_image.GetPointData().SetScalars(data_array)
    dim = l_image.GetDimension()
    l_spacing = [1.0] * 3
    l_spacing[:dim] = l_image.GetSpacing()
    vtk_image.SetSpacing(l_spacing)
    l_origin = [0.0] * 3
    l_origin[:dim] = l_image.GetOrigin()
    vtk_image.SetOrigin(l_origin)
    dims = [1] * 3
    dims[:dim] = l_image.GetSize()
    vtk_image.SetDimensions(dims)
    # Todo: Add Direction with VTK 9
    if l_image.GetDimension() == 3:
        pixel_type: str = l_image.GetPixelIDTypeAsString()
        if pixel_type.startswith("sitkVector"):
            vtk_image.GetPointData().SetVectors(data_array)
        # if PixelType == itk.Vector:
        #     vtk_image.GetPointData().SetVectors(data_array)
        # elif PixelType == itk.CovariantVector:
        #     vtk_image.GetPointData().SetVectors(data_array)
        # elif PixelType == itk.SymmetricSecondRankTensor:
        #     vtk_image.GetPointData().SetTensors(data_array)
        # elif PixelType == itk.DiffusionTensor3D:
        #     vtk_image.GetPointData().SetTensors(data_array)
    return vtk_image
