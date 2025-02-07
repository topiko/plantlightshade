import numpy as np
import torch


def cast(points_and_angles: np.ndarray, lens: np.ndarray) -> np.ndarray:
    points = points_and_angles[:, :2]
    angles = points_and_angles[:, 2]

    dxs = lens * np.cos(angles)
    dys = lens * np.sin(angles)

    return points + np.vstack([dxs, dys]).T


def vecs_to_point(from_points: torch.tensor, to_point: torch.tensor) -> torch.tensor:
    return to_point - from_points


def get_normals(vecs: torch.tensor) -> torch.tensor:

    vecs_ = vecs / torch.norm(vecs, dim=1).unsqueeze(1)
    return torch.vstack([-vecs_[:, 1], vecs_[:, 0]]).T


def get_projections(curve: torch.tensor, source_point: torch.tensor) -> torch.tensor:
    """
    For a curve get the projected lens of each piece
    where the projection direction is defined by the normal
    from the point on the curve to the source point.
    """
    curve_pieces = curve[:-1] - curve[1:]

    to_source = vecs_to_point(curve, source_point)[:-1]
    normals = get_normals(to_source)

    projections = curve_pieces @ normals.T

    return projections
