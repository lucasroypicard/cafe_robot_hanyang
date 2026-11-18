"""Pattern-independent, public-depth grasp orientation for upright mugs.

The robot controller is unchanged. A predicted surface pixel is lifted and
corrected 4 mm using the existing sensor rule. A bounded circle fit to nearby
above-table depth samples estimates the horizontal body centre; no RGB colour,
mask, object pose, mesh, request label or simulator feedback enters the fit.
"""
from dataclasses import asdict
import math
import numpy as np

from .cv_grasp import point_to_grasp, surface_at_pixel, unproject_pixels
from .open_lift import GraspPose

POLICY = {'rule': 'nearby_depth_circle', 'seed': 17, 'candidate_triples': 256,
          'sample_stride': 2, 'maximum_points': 640, 'radius_bounds_m': [.025, .075],
          'neighbour_radius_m': .14, 'surface_tolerance_m': .005,
          'minimum_support_points': 30, 'minimum_support_fraction': .15,
          'surface_offset_m': .004, 'fallback': 'fixed_vertical_yaw'}


def fit_circle_near_point(points, point, policy=POLICY):
    """Deterministic robust 2D circle fit; diagnostics do not certify a grasp."""
    points, point = np.asarray(points, float), np.asarray(point, float)
    if points.ndim != 2 or points.shape[1] != 2 or point.shape != (2,):
        raise ValueError('Expected Nx2 depth samples and a 2D surface point')
    if not np.isfinite(points).all() or not np.isfinite(point).all():
        raise ValueError('Finite public geometry required')
    if len(points) < policy['minimum_support_points']:
        return None
    # Repeated vertical observations should not dominate one camera column.
    _, unique = np.unique(np.round(points/.002).astype(np.int32), axis=0, return_index=True)
    points = points[np.sort(unique)]
    if len(points) < 12:
        return None
    if len(points) > policy['maximum_points']:
        points = points[np.linspace(0, len(points)-1, policy['maximum_points'], dtype=int)]
    rng = np.random.default_rng(policy['seed'])
    triples = np.array([rng.choice(len(points), 3, replace=False)
                        for _ in range(policy['candidate_triples'])])
    a, b, c = (points[triples[:, i]] for i in range(3))
    ab, ac = b-a, c-a
    determinant = 2*(ab[:, 0]*ac[:, 1]-ab[:, 1]*ac[:, 0])
    valid = np.abs(determinant) > 1e-6
    ab, ac, a, determinant = ab[valid], ac[valid], a[valid], determinant[valid]
    if not len(a):
        return None
    qb, qc = (ab*ab).sum(1), (ac*ac).sum(1)
    centres = a+np.c_[(qb*ac[:, 1]-qc*ab[:, 1])/determinant,
                       (qc*ab[:, 0]-qb*ac[:, 0])/determinant]
    radii = np.linalg.norm(centres-a, axis=1)
    distance = np.linalg.norm(centres-point, axis=1)
    valid = ((radii >= policy['radius_bounds_m'][0]) & (radii <= policy['radius_bounds_m'][1])
             & (distance >= radii-.012) & (distance <= radii+.085))
    centres, radii, distance = centres[valid], radii[valid], distance[valid]
    if not len(centres):
        return None
    residual = np.abs(np.linalg.norm(points[None]-centres[:, None], axis=-1)-radii[:, None])
    inliers = residual <= policy['surface_tolerance_m']
    counts = inliers.sum(1)
    # Support dominates; the tiny tie term prefers the nearest plausible body.
    best = int(np.argmax(counts-.01*distance))
    minimum = max(12, int(np.ceil(len(points)*policy['minimum_support_fraction'])))
    if counts[best] < minimum:
        return None
    return {'centre_xy_m': centres[best].tolist(), 'radius_m': float(radii[best]),
            'support_points': int(counts[best]), 'sample_points': len(points),
            'mean_inlier_residual_m': float(residual[best, inliers[best]].mean())}


def depth_radial_grasp(geometry, pixel_uv, rule='circle'):
    if rule not in ('circle', 'fixed'):
        raise ValueError('Unknown public depth adapter')
    observation = {'depth_m': geometry['depth_m'], 'depth_valid': geometry['depth_valid'],
                   'camera': {k: geometry[k] for k in ('K', 'camera_to_world')}}
    action = point_to_grasp(observation, pixel_uv)
    point, _ = surface_at_pixel(observation, pixel_uv)
    metadata = {'rule': 'fixed_vertical_yaw' if rule == 'fixed' else POLICY['rule'],
                'fallback_fixed_yaw': True, 'private_information_used': False}
    if rule == 'circle':
        depth, valid = geometry['depth_m'], geometry['depth_valid']
        yy, xx = np.mgrid[0:depth.shape[0]:POLICY['sample_stride'], 0:depth.shape[1]:POLICY['sample_stride']]
        mask = valid[yy, xx] & np.isfinite(depth[yy, xx]) & (depth[yy, xx] > 0)
        xy = np.c_[xx[mask], yy[mask]]
        cloud = unproject_pixels(xy, depth[yy[mask], xx[mask]], geometry)
        selected = ((cloud[:, 2] > .02) & (cloud[:, 2] < .18)
                    & (np.abs(cloud[:, 2]-point[2]) < .065)
                    & (np.linalg.norm(cloud[:, :2]-point[:2], axis=1) < POLICY['neighbour_radius_m']))
        fit = fit_circle_near_point(cloud[selected, :2], point[:2])
        metadata['circle_fit'] = fit
        if fit is not None:
            radial = np.asarray(action['position_world'])[:2]-fit['centre_xy_m']
            if np.linalg.norm(radial) >= .02:
                angle = math.atan2(radial[1], radial[0])
                action['quaternion_xyzw'] = [math.sin(angle/2), -math.cos(angle/2), 0., 0.]
                metadata.update(fallback_fixed_yaw=False, yaw_rad=angle)
    pose = GraspPose(**{k: action[k] for k in ('position_world', 'quaternion_xyzw', 'opening_m', 'approach_distance_m')})
    return asdict(pose), metadata
