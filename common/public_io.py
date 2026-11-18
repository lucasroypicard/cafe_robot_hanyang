"""Fixed observation loader for evaluation, independent of the notebook data loader.

This loader keeps input conditions consistent across local evaluation and grading.
No labels or robot_world information are included in model observations.
"""
import numpy as np
from PIL import Image
from .catalog import VIEWS, scene_folder, read_json


def load_observations(row, data_root):
    folder = scene_folder(row, data_root)
    cameras = read_json(folder/'camera.json')
    result = {}
    for view in VIEWS:
        rgb = np.array(Image.open(folder/f'rgb/{view}.png').convert('RGB'))
        depth = np.array(Image.open(folder/f'depth/{view}.tiff'), dtype=np.float32)
        valid = np.array(Image.open(folder/f'depth/{view}_valid.png')) > 0
        camera = cameras[view]
        if rgb.shape[:2] != depth.shape or valid.shape != depth.shape:
            raise ValueError('RGB/depth/valid dimensions differ')
        result[view] = {'rgb': rgb, 'geometry': {
            'depth_m': depth, 'depth_valid': valid,
            'K': np.array(camera['K'], dtype=np.float64),
            'camera_to_world': np.array(camera['camera_to_world'], dtype=np.float64)}}
    return result
