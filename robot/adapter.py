"""Student world point -> fixed grasp pose. No re-lifting of the target pixel.

Public depth may estimate a normal/body radius for the shared gripper policy.
The submitted world point is the actual position anchor. It is never replaced
by a coordinate recomputed from pixel_uv or by simulator ground truth.
"""
from dataclasses import asdict
import math
import numpy as np
from .backend.open_lift import GraspPose
from .backend.cv_grasp import local_surface_normal, unproject_pixels
from .backend.cafe_pattern_grasp import fit_circle_near_point, POLICY as CIRCLE_POLICY


def validate_prediction(prediction, observations):
    if not isinstance(prediction, dict):
        raise ValueError('Prediction must be a dictionary')
    if prediction.get('status') == 'NO_PICK':
        if not isinstance(prediction.get('reason'), str) or not prediction['reason']:
            raise ValueError('NO_PICK requires a reason')
        return
    if prediction.get('status') != 'PICK':
        raise ValueError('status must be PICK or NO_PICK')
    if prediction.get('selected_view') not in observations:
        raise ValueError('selected_view must identify a received camera')
    xyz = np.asarray(prediction.get('world_xyz_m'), dtype=float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError('world_xyz_m must be three finite metre coordinates')
    uv = prediction.get('pixel_uv')
    if uv is not None:
        uv = np.asarray(uv, dtype=float)
        if uv.shape != (2,) or not np.isfinite(uv).all():
            raise ValueError('pixel_uv must contain two finite values')
        h,w = observations[prediction['selected_view']]['rgb'].shape[:2]
        if not (0 <= uv[0] <= w-1 and 0 <= uv[1] <= h-1):
            raise ValueError('Diagnostic pixel lies outside the selected image')


def command_from_prediction(observations, prediction):
    validate_prediction(prediction, observations)
    if prediction['status'] == 'NO_PICK':
        return {'status':'no_pick', 'command':None, 'reason':prediction['reason']}
    point = np.asarray(prediction['world_xyz_m'], dtype=float).copy()
    geometry = observations[prediction['selected_view']]['geometry']
    observation = {'depth_m':geometry['depth_m'], 'depth_valid':geometry['depth_valid'],
                   'camera':{k:geometry[k] for k in ('K','camera_to_world')}}
    # Keep the original 4 mm inward sensor correction. This estimates only the
    # normal; the world position above is not recomputed from the diagnostic UV.
    normal = np.asarray(geometry['camera_to_world'])[:3,3] - point
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    if prediction.get('pixel_uv') is not None:
        try:
            normal = local_surface_normal(observation, prediction['pixel_uv'])
        except ValueError:
            pass  # 3D methods may predict a point with no valid depth in this view.
    position = point - .004*normal
    quaternion = [0.,-1.,0.,0.]
    depth,valid = geometry['depth_m'], geometry['depth_valid']
    yy,xx = np.mgrid[0:depth.shape[0]:2,0:depth.shape[1]:2]
    mask = valid[yy,xx] & np.isfinite(depth[yy,xx]) & (depth[yy,xx] > 0)
    cloud = unproject_pixels(np.c_[xx[mask],yy[mask]], depth[yy[mask],xx[mask]], geometry)
    selected = ((cloud[:,2] > .02) & (cloud[:,2] < .18)
                & (np.abs(cloud[:,2]-point[2]) < .065)
                & (np.linalg.norm(cloud[:,:2]-point[:2],axis=1) < CIRCLE_POLICY['neighbour_radius_m']))
    fit = fit_circle_near_point(cloud[selected,:2], point[:2])
    details = {'student_world_point_used':point.tolist(), 'surface_offset_m':.004,
               'circle_fit':fit, 'private_information_used':False, 'rim_raise_applied':False}
    if fit is not None:
        radial = position[:2] - fit['centre_xy_m']
        if np.linalg.norm(radial) >= .02:
            angle = math.atan2(radial[1],radial[0])
            quaternion = [math.sin(angle/2),-math.cos(angle/2),0.,0.]
        # Same original public rim-clearance policy and 10 mm deadband.
        gap = float(np.linalg.norm(position[:2]-fit['centre_xy_m'])-fit['radius_m'])
        if gap <= .0316158+.003:
            near = (np.linalg.norm(cloud[:,:2]-fit['centre_xy_m'],axis=1) <= fit['radius_m']+.008)
            near &= (cloud[:,2] > .02) & (cloud[:,2] < .20)
            if int(near.sum()) >= 30:
                top = float(np.quantile(cloud[near,2],.99))
                original_z = float(position[2])
                target_z = max(original_z,min(original_z+.04,top-.0390378+.003))
                if target_z-original_z >= .010:
                    position[2] = target_z
                    details['rim_raise_applied'] = True
    # Workspace checks live in the fixed GraspPose, before the physics command.
    command = asdict(GraspPose(tuple(position),tuple(quaternion),.08,.16))
    return {'status':'attempted', 'command':command, 'adapter_details':details,
            'reason':'student_world_point_with_shared_pose_policy'}
