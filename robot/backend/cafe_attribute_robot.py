"""Variable-count evaluator; identical physics and controller to CafeRobotScene.

Only the visual ceramic colour differs. Printed artwork is omitted from the
PyBullet presentation view; predictions use the original Blender RGB.
The inherited private selector is unique per cup, not the public colour name.
"""
from pathlib import Path
import numpy as np
from .cafe_robot import CafeRobotScene
from common.hashing import sha256


class AttributeRobotScene(CafeRobotScene):
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
            linear = np.asarray(cup["pattern_material"]["body_linear_rgb"])
            rgb = np.where(linear <= .0031308, linear*12.92, 1.055*linear**(1/2.4)-.055)
            visuals = [p.createVisualShape(p.GEOM_MESH, fileName=asset["files"][part]["path"],
                visualFramePosition=shift, rgbaColor=[*rgb, 1],
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

