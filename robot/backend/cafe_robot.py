"""Reconstruct the frozen three-cup scene and score a real Panda lift.

Privileged scene state stays in this evaluator. Perception supplies a pixel;
the shared point-to-grasp converter receives public depth/calibration only.
Original meshes, gravity, finger forces and the existing held-lift metric are
used. There are no attachment constraints, teleportation during the grasp,
or forces applied to a cup to help it follow the gripper.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .open_lift import OpenLiftEnv, GraspPose
from .cv_grasp import point_to_grasp
from .cv_stable_executor import execute_stable_grasp
from common.hashing import sha256
COLORS = {"red": (.48,.035,.022), "blue": (.025,.12,.48), "green": (.035,.30,.075), "yellow": (.65,.39,.025)}


class CafeRobotScene(OpenLiftEnv):
    def __init__(self, root, scene_id, requested_color, record=False):
        self.root = Path(root)
        self.snapshot = json.loads((self.root/f"instructor/{scene_id}/scene.json").read_text())
        self.plan = json.loads((self.root/"instructor/plan.json").read_text())
        matches = [c for c in self.snapshot["cups"] if c["color"] == requested_color]
        if len(matches) != 1:
            raise ValueError("Requested colour must identify one cup")
        self.target = matches[0]
        self.bodies, self.vertices, self.start_poses = {}, {}, {}
        self.motion_phases = []
        self._last_pose = None
        super().__init__(cup_id=self.target["asset_id"], xy=self.target["xy"],
                         yaw=self.target["yaw"], image_size=320, record=record)
        self.alignment = []
        for cup in self.snapshot["cups"]:
            position, quaternion = self.p.getBasePositionAndOrientation(self.bodies[cup["instance_id"]], physicsClientId=self.client)
            expected, original_q = self.start_poses[cup["instance_id"]]
            dot = abs(float(np.dot(quaternion, original_q)))
            self.alignment.append({"instance": cup["instance_id"],
                "position_drift_m": float(np.linalg.norm(np.asarray(position)-expected)),
                "rotation_drift_rad": float(2*np.arccos(np.clip(dot, 0, 1)))})
        if any(r["position_drift_m"] > .002 or r["rotation_drift_rad"] > .02 for r in self.alignment):
            self.close()
            raise ValueError("Settled scene differs from frozen RGB-D geometry")

    def _load_mug(self, cup_id, xy, yaw):
        p, cid = self.p, self.client
        # Move the fixed base BEFORE cups are created, preserving Panda COM offset.
        base = self.snapshot["robot"]["base_world_m"]
        local, quat = p.getDynamicsInfo(self.robot, -1, physicsClientId=cid)[3:5]
        position, quaternion = p.multiplyTransforms(base, [0, 0, 0, 1], local, quat)
        p.resetBasePositionAndOrientation(self.robot, position, quaternion, physicsClientId=cid)
        p.removeBody(self.plane, physicsClientId=cid)
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[.54, .47, .025], physicsClientId=cid)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[.54, .47, .025], rgbaColor=[.75, .68, .55, 1], physicsClientId=cid)
        self.plane = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual, basePosition=[.48, 0, -.025], physicsClientId=cid)
        for cup in self.snapshot["cups"]:
            asset = self.plan["assets"][str(cup["asset_id"])]
            for record in asset["files"].values():
                if sha256(record["path"]) != record["sha256"]:
                    raise ValueError("Changed collision/visual mesh")
            height = asset["height_m"]
            shift = [0, 0, -height/2]
            linear = np.asarray(COLORS[cup["color"]])
            rgb = np.where(linear <= .0031308, linear*12.92, 1.055*linear**(1/2.4)-.055)
            visuals = [p.createVisualShape(p.GEOM_MESH, fileName=asset["files"][part]["path"],
                visualFramePosition=shift, rgbaColor=[*(rgb if part == "body" else [.87, .87, .85]), 1],
                physicsClientId=cid) for part in ("body", "handle")]
            collision = p.createCollisionShape(p.GEOM_MESH, fileName=asset["files"]["collision"]["path"],
                collisionFramePosition=shift, physicsClientId=cid)
            position, quaternion = [*cup["xy"], height/2], p.getQuaternionFromEuler([0, 0, cup["yaw"]])
            body = p.createMultiBody(baseMass=.25, baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visuals[0], basePosition=position, baseOrientation=quaternion,
                linkMasses=[0], linkCollisionShapeIndices=[-1], linkVisualShapeIndices=[visuals[1]],
                linkPositions=[[0, 0, 0]], linkOrientations=[[0, 0, 0, 1]],
                linkInertialFramePositions=[[0, 0, 0]], linkInertialFrameOrientations=[[0, 0, 0, 1]],
                linkParentIndices=[0], linkJointTypes=[p.JOINT_FIXED], linkJointAxis=[[0, 0, 0]], physicsClientId=cid)
            p.changeDynamics(body, -1, lateralFriction=1., spinningFriction=.02,
                rollingFriction=.001, collisionMargin=.0002, physicsClientId=cid)
            vertices = np.asarray([list(map(float, line.split()[1:4])) for line in
                Path(asset["files"]["collision"]["path"]).read_text().splitlines() if line.startswith("v ")])
            vertices = np.unique(vertices, axis=0)
            vertices[:, 2] -= height/2
            self.bodies[cup["instance_id"]] = body
            self.vertices[body] = vertices
            self.start_poses[cup["instance_id"]] = (position, quaternion)
        selected = self.bodies[self.target["instance_id"]]
        self.collision_vertices = self.vertices[selected]
        return selected

    def _camera(self, view_id):
        # This is a presentation camera only, never a policy observation.
        p, cid = self.p, self.client
        view = p.computeViewMatrix([1.15, -1.15, .85], [.46, .02, .14], [0, 0, 1])
        projection = p.computeProjectionMatrixFOV(49, 1, .02, 3.)
        pixels = p.getCameraImage(self.size, self.size, view, projection,
            renderer=p.ER_TINY_RENDERER, physicsClientId=cid)
        return {"rgb": np.asarray(pixels[2], np.uint8).reshape(self.size, self.size, 4)[..., :3]}

    def _move(self, pose, ticks):
        self._last_pose = pose
        super()._move(pose, ticks)
        state = self.p.getLinkState(self.robot, 11, computeForwardKinematics=True, physicsClientId=self.client)
        self.motion_phases.append({"tick": self.tick, "target": list(pose.position_world),
            "actual": list(state[4]), "tracking_error_m": float(np.linalg.norm(np.asarray(state[4])-pose.position_world))})

    def execute_grasp(self, pose):
        return execute_stable_grasp(self, pose, "safe_transit_smooth_lift")

    def other_cup_rises(self):
        return {str(i): float(self.p.getBasePositionAndOrientation(body, physicsClientId=self.client)[0][2]
                             - self.start_poses[i][0][2]) for i, body in self.bodies.items()
                if i != self.target["instance_id"]}


def public_grasp(root, scene_id, pixel_uv, view="corner_sw"):
    with np.load(Path(root)/f"observations/{scene_id}/{view}/geometry.npz") as geom:
        observation = {"depth_m": geom["depth_m"], "depth_valid": geom["depth_valid"],
                       "camera": {"K": geom["K"], "camera_to_world": geom["camera_to_world"]}}
        action = point_to_grasp(observation, pixel_uv)
    return GraspPose(**{k: action[k] for k in ("position_world", "quaternion_xyzw", "opening_m", "approach_distance_m")})
