"""Single-view point to grasp conversion using only public RGB-D sensors.

This baseline chooses a vertical, fixed-orientation grasp. Students may replace
the point predictor, surface correction or orientation; these are not task rules.
All lengths are metres. UV means (column, row), starting at zero.
"""
import numpy as np

from .open_lift import GraspPose


def project_world(xyz, camera):
    """World point -> pixel and camera-z depth with the supplied calibration."""
    local = np.linalg.solve(np.asarray(camera['camera_to_world']), np.r_[xyz, 1.])[:3]
    if not np.isfinite(local).all() or local[2] <= 0:
        raise ValueError('point must be finite and in front of the camera')
    pixel = np.asarray(camera['K']) @ local
    return pixel[:2] / pixel[2], float(local[2])


def unproject_pixels(uv, depth, camera):
    """Vectorized pixel + camera-z depth -> world coordinates (..., 3)."""
    uv = np.asarray(uv, dtype=float)
    depth = np.asarray(depth, dtype=float)
    if uv.shape[-1:] != (2,) or depth.shape != uv.shape[:-1]:
        raise ValueError('UV/depth shapes must be (...,2) and (...)')
    if not np.isfinite(uv).all() or not np.isfinite(depth).all() or (depth <= 0).any():
        raise ValueError('finite UV and positive depth required')
    rays = np.concatenate([uv, np.ones((*uv.shape[:-1], 1))], axis=-1)
    xyz_camera = (rays @ np.linalg.inv(camera['K']).T) * depth[..., None]
    transform = np.asarray(camera['camera_to_world'])
    return xyz_camera @ transform[:3, :3].T + transform[:3, 3]


def surface_at_pixel(observation, uv):
    """Use the nearest sensor pixel, not an interpolated foreground/background depth."""
    uv = np.asarray(uv, dtype=float)
    depth = observation['depth_m']
    if uv.shape != (2,) or not np.isfinite(uv).all():
        raise ValueError('UV must contain two finite coordinates')
    height, width = depth.shape
    if not (0 <= uv[0] <= width-1 and 0 <= uv[1] <= height-1):
        raise ValueError('predicted pixel is outside the image')
    u, v = np.rint(uv).astype(int)
    if not observation['depth_valid'][v, u] or not np.isfinite(depth[v, u]) or depth[v, u] <= 0:
        raise ValueError('predicted pixel has invalid depth')
    xyz = unproject_pixels(np.array([u, v]), np.asarray(depth[v, u]), observation['camera'])
    return xyz, (u, v)


def local_surface_normal(observation, uv, radius=2, neighbour_distance_m=.012):
    """Fit a small depth plane, rejecting neighbours across a depth discontinuity.

    SVD's last vector is the normal of the least-squares plane. The sign is
    ambiguous, so point it towards the camera. A tiny/degenerate patch falls back
    to the viewing direction, not to any simulator ground-truth normal.
    """
    xyz, (u, v) = surface_at_pixel(observation, uv)
    depth, valid = observation['depth_m'], observation['depth_valid']
    height, width = depth.shape
    yy, xx = np.mgrid[max(0, v-radius):min(height, v+radius+1),
                     max(0, u-radius):min(width, u+radius+1)]
    selected = valid[yy, xx] & np.isfinite(depth[yy, xx]) & (depth[yy, xx] > 0)
    pixels = np.column_stack([xx[selected], yy[selected]])
    points = unproject_pixels(pixels, depth[yy[selected], xx[selected]], observation['camera'])
    points = points[np.linalg.norm(points-xyz, axis=1) < neighbour_distance_m]
    toward_camera = np.asarray(observation['camera']['camera_to_world'])[:3, 3] - xyz
    normal = toward_camera
    if len(points) >= 4:
        _, singular, vectors = np.linalg.svd(points-points.mean(0), full_matrices=False)
        if singular[1] > 1e-6:
            normal = vectors[-1]
            if normal @ toward_camera < 0:
                normal = -normal
    return normal / np.linalg.norm(normal)


def point_to_grasp(observation, uv, surface_offset_m=.004):
    """Public baseline rule: 4 mm inward correction and fixed downward approach.

    A surface point is not the centre between the fingers. This small correction
    is a baseline heuristic, not an assertion that every surface point is valid.
    GraspPose checks the public workspace before any motor command is sent.
    """
    xyz, _ = surface_at_pixel(observation, uv)
    normal = local_surface_normal(observation, uv)
    target = xyz - float(surface_offset_m)*normal
    pose = GraspPose(tuple(target), (0., -1., 0., 0.), .08, .16)
    return {'type': 'grasp', 'position_world': list(pose.position_world),
            'quaternion_xyzw': list(pose.quaternion_xyzw),
            'opening_m': pose.opening_m, 'approach_distance_m': pose.approach_distance_m}
