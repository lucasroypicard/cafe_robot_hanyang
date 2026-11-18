"""Fixed robot execution and success checks, using student coordinates through robot.adapter."""
from pathlib import Path
import numpy as np
from common.catalog import ROOT, read_json
from .backend.cafe_attribute_robot import AttributeRobotScene
from .backend.open_lift import GraspPose


class PlainRobotScene(AttributeRobotScene):
    def _camera(self, view_id):
        """Recording only: outline the target in cyan, without changing any physics."""
        from PIL import Image, ImageFilter
        p, cid = self.p, self.client
        # Identical presentation camera to CafeRobotScene; not a model input.
        view = p.computeViewMatrix([1.15, -1.15, .85], [.46, .02, .14], [0, 0, 1])
        projection = p.computeProjectionMatrixFOV(49, 1, .02, 3.)
        pixels = p.getCameraImage(self.size, self.size, view, projection,
            renderer=p.ER_TINY_RENDERER, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            physicsClientId=cid)
        rgb = np.asarray(pixels[2], np.uint8).reshape(self.size, self.size, 4)[..., :3].copy()
        ids = np.asarray(pixels[4], np.int64).reshape(self.size, self.size)
        target = (ids >= 0) & ((ids & ((1 << 24) - 1)) == self.cup)
        mask = Image.fromarray(target.astype(np.uint8) * 255)
        outer = np.asarray(mask.filter(ImageFilter.MaxFilter(3))) > 0
        inner = np.asarray(mask.filter(ImageFilter.MinFilter(3))) > 0
        rgb[outer & ~inner] = [0, 255, 255]
        return {'rgb': rgb}

    def _load_mug(self, cup_id, xy, yaw):
        for asset in self.plan['assets'].values():
            for item in asset['files'].values():
                path = (ROOT/item['path']).resolve()
                if not path.is_relative_to(ROOT/'robot_world'):
                    raise ValueError('Robot mesh path escapes robot_world')
                item['path'] = str(path)
        return super()._load_mug(cup_id, xy, yaw)


def execute(row, conversion, record=False, output=None):
    scene = read_json(ROOT/f'robot_world/instructor/{row["scene_id"]}/scene.json')
    request = row['request']
    matches = [c for c in scene['cups'] if (c['body_color'],c['pattern']) == (request['color'],request['pattern'])]
    if len(matches) != 1:
        raise ValueError('Request must identify exactly one cup')
    with PlainRobotScene(ROOT/'robot_world',row['scene_id'],matches[0]['color'],record=record) as env:
        command = conversion['command']
        metric = env.execute_grasp(GraspPose(**command)) if command is not None else env.finish()
        if record and env.tick % 24:
            env.frames.append(env._camera('front')['rgb'])
        result = {'metric':metric, 'other_cup_final_rise_m':env.other_cup_rises(),
                  'gripper_attachment_constraints':0, 'render_style':'texture-free geometry; target outline in recorded frames'}
        if output is not None:
            output = Path(output)
            output.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(output/'trace.npz',**{k:np.asarray([t[k] for t in env.trace]) for k in env.trace[0]})
            if record:
                import imageio.v2 as imageio
                imageio.mimsave(output/'robot.gif',env.frames,duration=100,loop=0)
                imageio.imwrite(output/'robot_start.png',env.frames[0])
                imageio.imwrite(output/'robot_final.png',env.frames[-1])
        return result
