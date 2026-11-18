"""Shared grasp execution with fixed public safety waypoints.

This changes neither perception nor grasp targets. The archived executor sends
the final 24 cm lift target in one command. Here intermediate Cartesian targets
limit the large initial motion. No cup pose, part label, contact identity or
grasp-success feedback steers the motion. Physics, force, metric and 1190-tick
budget remain identical. Archived experiments continue using their old code.
"""
import numpy as np

from .open_lift import GraspPose, OpenLiftEnv


PROFILE_CONFIGS = {
    'original': {'lift_interpolation': 'single_target', 'lift_ticks': 260,
                 'waypoint_ticks': 260, 'total_ticks': 1190},
    'linear_lift': {'lift_interpolation': 'linear', 'lift_ticks': 260,
                    'waypoint_ticks': 10, 'total_ticks': 1190},
    'smooth_lift': {'lift_interpolation': 'minimum_jerk', 'lift_ticks': 260,
                    'waypoint_ticks': 10, 'total_ticks': 1190},
    'safe_transit_smooth_lift': {
        'pregrasp_interpolation': 'fixed_high_transit', 'transit_height_m': .45,
        'transit_ticks': 110, 'pregrasp_ticks': 110,
        'lift_interpolation': 'minimum_jerk', 'lift_ticks': 260,
        'waypoint_ticks': 10, 'total_ticks': 1190},
}


def validate_profile(profile):
    if type(profile) is not str or profile not in PROFILE_CONFIGS:
        raise ValueError('unknown stable-executor profile')
    return profile


def lift_waypoints(pose, profile):
    """Pure target generation; no simulator access or object-dependent rule."""
    validate_profile(profile)
    if not isinstance(pose, GraspPose):
        raise ValueError('pose must be a GraspPose')
    count = 260 // PROFILE_CONFIGS[profile]['waypoint_ticks']
    result = []
    xyz = np.asarray(pose.position_world)
    for index in range(1, count + 1):
        t = index / count
        smooth = PROFILE_CONFIGS[profile]['lift_interpolation'] == 'minimum_jerk'
        fraction = 10*t**3 - 15*t**4 + 6*t**5 if smooth else t
        result.append(GraspPose(tuple(xyz + [0., 0., .24*fraction]),
                                pose.quaternion_xyzw, .004, 0.))
    return result


def execute_stable_grasp(env, pose, profile):
    """Same grasp/lift endpoints, force and total time; no object-state feedback."""
    validate_profile(profile)
    if profile == 'original':
        return OpenLiftEnv.execute_grasp(env, pose)
    env._ensure_ticks(1190)
    if env.mode not in (None, 'one_shot', 'active_view'):
        raise ValueError('one-shot grasp is not an action in this mode')
    if not isinstance(pose, GraspPose):
        raise ValueError('pose must be a GraspPose')
    rotation = np.asarray(env.p.getMatrixFromQuaternion(pose.quaternion_xyzw)).reshape(3, 3)
    xyz = np.asarray(pose.position_world)
    pre = GraspPose(tuple(xyz - rotation[:, 2]*pose.approach_distance_m),
                    pose.quaternion_xyzw, pose.opening_m, 0.)
    # Validate the complete trajectory before any motor command, as before.
    lifts = lift_waypoints(pose, profile)
    config = PROFILE_CONFIGS[profile]
    if profile == 'safe_transit_smooth_lift':
        transit = GraspPose((xyz[0], xyz[1], config['transit_height_m']),
                            pose.quaternion_xyzw, pose.opening_m, 0.)
        env._move(transit, config['transit_ticks'])
        if hasattr(env, 'motion_phases'):
            env.motion_phases[-1]['phase'] = 'transit'
        env._move(pre, config['pregrasp_ticks'])
    else:
        env._move(pre, 220)
    env._move(pose, 170)
    env._gripper(.004)
    env._advance(240)
    for index, waypoint in enumerate(lifts):
        env._move(waypoint, config['waypoint_ticks'])
        if index + 1 < len(lifts) and hasattr(env, 'motion_phases'):
            env.motion_phases[-1]['phase'] = 'lift_waypoint'
    return env.finish()

