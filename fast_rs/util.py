from typing import Union

import numpy as np
import numpy.typing as npt


def ellipse_perimeter(
    sm_axis_r: Union[float, npt.NDArray[np.float64]], area: Union[float, npt.NDArray[np.float64]]
) -> Union[float, npt.NDArray[np.float64]]:
    """Calculate the perimeter of ellipse from area and semi-minor axis radius, using the first Ramanujan approximation. Works also on arrays of ellipses.
    Arguments:
        sm_axis_r: float | npt.NDArray[float]
            Semi-minor axis radius of the ellipse.
        area: float | npt.NDArray[float]
            Area of the ellipse.
    Returns:
        float | npt.NDArray[float]:
            Perimeter of the ellipse.
    """

    h = area / (np.pi * sm_axis_r)

    return np.pi * (
        3 * (h + sm_axis_r) - np.sqrt((3 * h + sm_axis_r) * (h + 3 * sm_axis_r))
    )


def spheroid_surface_area(
    rad_a: npt.NDArray[np.float64], volumes: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Calculate the surface area of spheroids from the equatorial radus and volume. Works on arrays of spheroids.
    Arguments:
        rad_a: npt.NDArray[float]
            Equatorial radius of the spheroids.
        volumes: npt.NDArray[float]
            Volumes of the spheroids.
    Returns:
        npt.NDArray[float]:
            Surface area of the spheroids.
    """

    # Get distance from center to pole
    c_dist = (3 * volumes) / (4 * np.pi * (rad_a**2))

    # Calculate surface area of spheroids
    surface_area = np.where(
        c_dist < rad_a, oblate_sa(rad_a, c_dist), prolate_sa(rad_a, c_dist)
    )

    return surface_area

def oblate_sa(
    rad_a: Union[float, npt.NDArray[np.float64]], c_dist: Union[float, npt.NDArray[np.float64]]
) -> Union[float, npt.NDArray[np.float64]]:
    """Calculate surface area of a oblate spheroid from semi axes size.
    Arguments:
        rad_a: float | npt.NDArray[float]
            Equatorial radius of the spheroid.
        c_dist: float | npt.NDArray[float]
            Distance from center to pole of the spheroid.
    Returns:
        float | npt.NDArray[float]:
            Surface area of the spheroid.
    """

    # Calculate eccentricity values for oblate spheroids
    e_sqr_vals = 1 - (c_dist / rad_a) ** 2
    # Set negative values to np.nan to silence warnings
    e_sqr_vals = np.where(e_sqr_vals < 0, np.nan, e_sqr_vals)

    e_vals = np.sqrt(e_sqr_vals)

    return (
        2 * np.pi * (rad_a**2) * (1 + ((1 - e_sqr_vals) / e_vals) * np.arctanh(e_vals))
    )


def prolate_sa(
    rad_a: Union[float, npt.NDArray[np.float64]], c_dist: Union[float, npt.NDArray[np.float64]]
) -> Union[float, npt.NDArray[np.float64]]:
    """Calculate surface area of a prolate spheroid from semi axes size.
    Arguments:
        rad_a: float | npt.NDArray[float]
            Equatorial radius of the spheroid.
        c_dist: float | npt.NDArray[float]
            Distance from center to pole of the spheroid.
    Returns:
        float | npt.NDArray[float]:
            Surface area of the spheroid.
    """

    # Calculate eccentricity values for prolate spheroids
    e_sqr_vals = 1 - (rad_a / c_dist) ** 2
    # Set negative values to np.nan to silence warnings
    e_sqr_vals = np.where(e_sqr_vals < 0, np.nan, e_sqr_vals)
    
    e_vals = np.sqrt(e_sqr_vals)

    return 2 * np.pi * (rad_a**2) * (1 + c_dist / (rad_a * e_vals) * np.arcsin(e_vals))


def edge_mask_nd(mask: npt.NDArray, ndim: int) -> npt.NDArray:
    """Create edge mask of n-dimensional object.
    Arguments:
        mask: npt.NDArray[float]
            Binary mask of the object.
        ndim: int
            Number of dimensions of the object.
    Returns:
        edge_mask: npt.NDArray[float]
            Edge mask of the object.
    """

    # Initialize edge mask
    edge_mask = np.zeros_like(mask)

    # Iterate over dimensions
    for dim in range(ndim):
        # Take 1-shifted subsets of mask along dimension
        slices1 = [slice(None)] * ndim
        slices2 = [slice(None)] * ndim
        slices1[dim] = slice(1, None)
        slices2[dim] = slice(None, -1)

        # Calculate edge mask along dimension (XOR between 1-shifted subsets)
        edge_mask_dim = mask[tuple(slices1)] ^ mask[tuple(slices2)]

        # Combine edge mask with previous dimensions
        if dim == 0:
            edge_mask[tuple(slices1)] = edge_mask_dim
        else:
            edge_mask[tuple(slices1)] = edge_mask[tuple(slices1)] | edge_mask_dim
        edge_mask[tuple(slices2)] = edge_mask[tuple(slices2)] | edge_mask_dim

    # Remove edge mask outside of object (extra safety)
    edge_mask = edge_mask & mask

    return edge_mask
