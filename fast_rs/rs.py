from typing import Union

import numpy as np
import numpy.typing as npt


from scipy.ndimage import labeled_comprehension
from skimage.measure import label
import localthickness as lt

from . import util


def rs_calulate(mask: npt.NDArray[bool], lt_scale: float = 0.5) -> npt.NDArray[float]:
    """Calculate the roundness and sphericity of objects in a 2D or 3D binary mask using the local thickness-based approximations.
    Arguments:
        mask: npt.NDArray[float]
            Binary mask of the objects.
        lt_scale: float
            Scale of the local thickness calculation (0,1> range.
    Returns:
        roundness_vec: (n) npt.NDArray[float]
            Roundness of the objects.
        sphericity_vec: (n) npt.NDArray[float]
            Sphericity of the objects.
        label_img: npt.NDArray[int]
            Labeled image of the objects. Label indices correspond to the position indices in roundness_vec and sphericity_vec.
    """

    # Check if mask is 2D or 3D
    assert mask.ndim in [2, 3], f"Mask must be 2D or 3D, but mask.ndim={mask.ndim}"
    # Check if lt_scale is in the correct range
    assert (
        0 < lt_scale <= 1
    ), f"lt_scale must be in the (0,1> range, but lt_scale={lt_scale}"

    label_img = label(mask)

    # Calculate local thickness
    thickness = lt.local_thickness(mask, scale=lt_scale)

    sphericity_vec = sphericity(mask, label_img, lt_scale, thickness)
    roundness_vec = roundness(mask, label_img, lt_scale, thickness)

    return roundness_vec, sphericity_vec, label_img


def roundness(
    mask: npt.NDArray[bool],
    label_img: npt.NDArray[int],
    lt_scale: float = 0.5,
    thickness: Union[None,npt.NDArray[float]] = None,
) -> npt.NDArray[float]:
    """Calculate the roundness of objects in a 2D or 3D binary mask. Approximates corner curvature using mask edge local thickness values.
    Arguments:
        mask: npt.NDArray[float]
            Binary mask of the objects.
        label_img: npt.NDArray[float]
            Labeled image of the objects (eg. from skimage.measure.label).
        lt_scale: float
            Scale of the local thickness calculation (0,1> range.
        thickness: None | npt.NDArray[float]
            Local thickness of the mask, can be passed to accelerate execution if sphericity is also calculated.
    Returns:
        roundness_vec: (n) npt.NDArray[float]
            Roundness of the objects.
    """

    # Check if dimensions of mask and label_img are the same
    assert (
        mask.shape == label_img.shape
    ), f"mask shape {mask.shape} and label_img shape {label_img.shape} must have the same dimensions"
    # Check if mask is 2D or 3D
    assert mask.ndim in [2, 3], f"Mask must be 2D or 3D, but mask.ndim={mask.ndim}"
    # Check if lt_scale is in the correct range
    assert (
        0 < lt_scale <= 1
    ), f"lt_scale must be in the (0,1> range, but lt_scale={lt_scale}"

    # Calculate local thickness if not passed
    if thickness is None:
        thickness = lt.local_thickness(mask, scale=lt_scale)
    else:
        assert (
            mask.shape == thickness.shape
        ), f"mask shape {mask.shape} and thickness shape {thickness.shape} must have the same dimensions"

    # Create edge mask from mask
    edge_mask = util.edge_mask_nd(np.copy(mask), mask.ndim)

    # Remove labels outside of edge mask
    label_edge = np.copy(label_img)
    label_edge[edge_mask == 0] = 0

    # Get mean curvature approximation from local thickness
    mean_thickness = labeled_comprehension(
        thickness, label_edge, np.arange(1, np.max(label_img) + 1), np.mean, float, 0
    )
    # Get radius of the maximum inscribed circle/sphere
    max_thickness = labeled_comprehension(
        thickness, label_edge, np.arange(1, np.max(label_img) + 1), np.max, float, 0
    )

    roundness_vec = mean_thickness / max_thickness

    return roundness_vec


def sphericity(
    mask: npt.NDArray[bool],
    label_img: npt.NDArray[int],
    lt_scale: float = 0.5,
    thickness: Union[None, npt.NDArray[float]] = None,
) -> npt.NDArray[float]:
    """Calculate the sphericity of objects in a 2D or 3D binary mask. Approximates object shape as ellipse in 2D and spheroid in 3D.
    Arguments:
        mask: npt.NDArray[float]
            Binary mask of the objects.
        label_img: npt.NDArray[float]
            Labeled image of the objects (eg. from skimage.measure.label).
        lt_scale: (float)
            Scale of the local thickness calculation (0,1> range.
        thickness: None | npt.NDArray[float]
            Local thickness of the mask, can be passed to accelerate execution if roundness is also calculated.
    Returns:
        sphericity_vec: (n) npt.NDArray[float]
            Sphericity of the objects.
    """

    # Check if dimensions of mask and label_img are the same
    assert (
        mask.shape == label_img.shape
    ), f"mask shape {mask.shape} and label_img shape {label_img.shape} must have the same dimensions"
    # Check if mask is 2D or 3D
    assert mask.ndim in [2, 3], f"Mask must be 2D or 3D, but mask.ndim={mask.ndim}"
    # Check if lt_scale is in the correct range
    assert (
        0 < lt_scale <= 1
    ), f"lt_scale must be in the (0,1> range, but lt_scale={lt_scale}"

    # Calculate local thickness if not passed
    if thickness is None:
        thickness = lt.local_thickness(mask, scale=lt_scale)
    else:
        assert (
            mask.shape == thickness.shape
        ), f"mask shape {mask.shape} and thickness shape {thickness.shape} must have the same dimensions"

    # Collect area/volume of objects
    unique, a_v_vec = np.unique(label_img, return_counts=True)
    unique = unique[1:]
    a_v_vec = a_v_vec[1:]
    # Collect mean thickness of objects
    thick_mean_vec = labeled_comprehension(
        thickness, label_img, unique, np.mean, float, 0
    )

    if mask.ndim == 2:
        # Get perimeter of object (modelled as ellipse)
        obj_perim = util.ellipse_perimeter(thick_mean_vec, a_v_vec)
        # Get perimeter of circle with same area as object
        circle_perim = 2 * np.sqrt(np.pi * a_v_vec)

        sphericity_vec = circle_perim / obj_perim

    elif mask.ndim == 3:
        # Get surface area of object (modelled as spheroid)
        obj_surf_area = util.spheroid_surface_area(thick_mean_vec, a_v_vec)
        # Get surface area of sphere with same volume as object
        sphere_surf_area = np.pi ** (1 / 3) * (6 * a_v_vec) ** (2 / 3)

        sphericity_vec = sphere_surf_area / obj_surf_area

    return sphericity_vec
