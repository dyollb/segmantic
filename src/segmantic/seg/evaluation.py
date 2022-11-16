from typing import Dict

import numpy as np
import SimpleITK as sitk


def hausdorff_surface_distance(
    y_pred: sitk.Image, y_ref: sitk.Image
) -> Dict[str, float]:
    """Compute symmetric surface distances between two binary masks

    Args:
        y_pred: predicted segmentation
        y_ref: reference segmentation

    Returns:
        Dict[str, float]: keys are 'mean', 'median', 'std', 'max'
    """
    seg_surface = sitk.BinaryContour(y_pred, foregroundValue=1)
    ref_surface = sitk.BinaryContour(y_ref, foregroundValue=1)

    seg_distance = sitk.SignedMaurerDistanceMap(
        y_pred, useImageSpacing=True, squaredDistance=False, insideIsPositive=False
    )
    ref_distance = sitk.SignedMaurerDistanceMap(
        y_ref, useImageSpacing=True, squaredDistance=False, insideIsPositive=False
    )

    # get distance at contour of foreground label
    ref2seg_distance = np.multiply(ref_surface, seg_distance)
    ref2seg_distance = ref2seg_distance[np.nonzero(ref_surface)]

    seg2ref_distance = np.multiply(seg_surface, ref_distance)
    seg2ref_distance = seg2ref_distance[np.nonzero(seg_surface)]

    # compute statistics on symmetric distances (both directions)
    all_surface_distances = np.concatenate(
        (ref2seg_distance, seg2ref_distance), axis=None
    )
    all_surface_distances = np.abs(all_surface_distances)

    mean_surface_distance = np.mean(all_surface_distances)
    median_surface_distance = np.median(all_surface_distances)
    std_surface_distance = np.std(all_surface_distances)
    max_surface_distance = np.max(all_surface_distances)
    return {
        "mean": mean_surface_distance,
        "median": median_surface_distance,
        "std": std_surface_distance,
        "max": max_surface_distance,
    }


def hausdorff_pointwise_distance(
    y_pred: sitk.Image, y_ref: sitk.Image
) -> Dict[str, float]:
    """Compute symmetric point-wise distances between two binary masks

    Args:
        y_pred: predicted segmentation
        y_ref: reference segmentation

    Returns:
        Dict[str, float]: keys are 'mean', 'median', 'std', 'max'
    """
    seg_distance = sitk.SignedMaurerDistanceMap(
        y_pred, useImageSpacing=True, squaredDistance=False, insideIsPositive=False
    )
    ref_distance = sitk.SignedMaurerDistanceMap(
        y_ref, useImageSpacing=True, squaredDistance=False, insideIsPositive=False
    )

    def to_np(img: sitk.Image):
        return sitk.GetArrayViewFromImage(img)

    # get distance inside foreground label
    ref2seg_distance = to_np(seg_distance)[np.nonzero(to_np(y_ref))]
    seg2ref_distance = to_np(ref_distance)[np.nonzero(to_np(y_pred))]

    # compute statistics on symmetric distances (both directions)
    all_surface_distances = np.concatenate(
        (ref2seg_distance, seg2ref_distance), axis=None
    )
    all_surface_distances[all_surface_distances <= 0.0] = 0.0

    mean_surface_distance = np.mean(all_surface_distances)
    median_surface_distance = np.median(all_surface_distances)
    std_surface_distance = np.std(all_surface_distances)
    max_surface_distance = np.max(all_surface_distances)
    return {
        "mean": mean_surface_distance,
        "median": median_surface_distance,
        "std": std_surface_distance,
        "max": max_surface_distance,
    }


def confusion_matrix(num_classes: int, y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute confusion matrix similar to sklearn.metrics.confusion_matrix

    Args:
        num_classes (int): Number of labels including '0', i.e. max(y)+1
        y_pred (np.ndarray): Predicted labels
        y (np.ndarray): True labels

    Returns:
        np.ndarray: Dimension num_classes x num_classes
    """
    try:
        from numba import njit

        @njit
        def _compute_confusion(
            num_classes: int, y_pred: np.ndarray, y: np.ndarray
        ) -> np.ndarray:
            cm = np.zeros((num_classes, num_classes))
            for i in range(num_classes):
                cm[y[i], y_pred[i]] += 1
            return cm

        return _compute_confusion(num_classes, y_pred, y)  # type: ignore
    except ImportError:
        # fall back to naive approach
        cm = np.zeros((num_classes, num_classes))
        for t, p in zip(y_pred, y):
            cm[t, p] += 1
    return cm
