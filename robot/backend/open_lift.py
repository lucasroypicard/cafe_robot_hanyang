"""Method-independent mug lifting prototype, separate from the legacy baseline.

The host owns PyBullet. Student code receives copied sensor arrays and returns
poses/actions, never this environment object. This in-process development runner
is NOT a security boundary for hidden classroom evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np

from .open_lift_metric import LiftRules, evaluate_lift


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "data_examples/mug_blender_course_physics_v1"
DT = 1 / 240
WORKSPACE_LOW = np.array([.05, -.65, .015])
WORKSPACE_HIGH = np.array([.85, .50, .85])
VIEWS = {"front": (-52., 38.), "left": (-105., 30.), "right": (10., 35.)}


def _vector(value, size, name):
    raw = np.asarray(value)
    if raw.shape != (size,) or raw.dtype.kind not in "iuf" or any(
            isinstance(x, (bool, np.bool_)) for x in np.asarray(value, dtype=object)):
        raise ValueError(f"{name} must be a numeric vector of length {size}")
    result = raw.astype(float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be representable as a finite number") from error
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class GraspPose:
    """TCP is Panda link 11, the center of the open fingers; units metres.

    Quaternion is xyzw in the world frame. Local +z is the approach direction.
    No semantic part, image point, surface normal, or learned output is required.
    """
    position_world: tuple
    quaternion_xyzw: tuple
    opening_m: float = .08
    approach_distance_m: float = .12

    def __post_init__(self):
        xyz = _vector(self.position_world, 3, "position_world")
        quat = _vector(self.quaternion_xyzw, 4, "quaternion_xyzw")
        if (xyz < WORKSPACE_LOW).any() or (xyz > WORKSPACE_HIGH).any():
            raise ValueError("TCP target is outside the public workspace")
        norm = np.linalg.norm(quat)
        if abs(norm - 1.) > 1e-3:
            raise ValueError("quaternion must have unit length")
        width = _scalar(self.opening_m, "opening_m")
        approach = _scalar(self.approach_distance_m, "approach_distance_m")
        if not 0 <= width <= .08:
            raise ValueError("opening_m must be in [0, 0.08]")
        if not 0 <= approach <= .20:
            raise ValueError("approach_distance_m must be in [0, 0.20]")
        object.__setattr__(self, "position_world", tuple(xyz))
        object.__setattr__(self, "quaternion_xyzw", tuple(quat / norm))
        object.__setattr__(self, "opening_m", width)
        object.__setattr__(self, "approach_distance_m", approach)


class OpenLiftEnv:
    """DIRECT-mode trusted simulator for reviewing three task variants.

    The public hardware is a Panda with an 8 cm parallel gripper. All task modes
    use identical mass/friction/forces/metric. Successful grasp diversity is a
    property to measure, not guaranteed by allowing arbitrary quaternions.
    """
    def __init__(self, cup_id=25, xy=(.48, .04), yaw=0., image_size=384,
                 max_seconds=30., record=False):
        import pybullet as p
        import pybullet_data
        if type(cup_id) is not int or not 0 <= cup_id < 30:
            raise ValueError("prototype cup_id must be an integer in [0, 29]")
        xy = _vector(xy, 2, "xy")
        yaw = _scalar(yaw, "yaw")
        if type(image_size) is not int or not 32 <= image_size <= 1024:
            raise ValueError("image_size must be an integer in [32, 1024]")
        max_seconds = _scalar(max_seconds, "max_seconds")
        if not 2 <= max_seconds <= 120:
            raise ValueError("max_seconds must be in [2, 120]")
        self.p, self.size = p, image_size
        self.client = p.connect(p.DIRECT)
        self.tick, self.max_ticks = 0, int(round(max_seconds / DT))
        self.record, self.frames, self.trace = bool(record), [], []
        self.view_requests, self.finished = 0, False
        self.mode = None
        try:
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
            p.setTimeStep(DT, physicsClientId=self.client)
            p.setGravity(0, 0, -9.81, physicsClientId=self.client)
            p.setPhysicsEngineParameter(numSolverIterations=150,
                deterministicOverlappingPairs=1, physicsClientId=self.client)
            self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
            self.robot = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, -.45, 0],
                useFixedBase=True, physicsClientId=self.client)
            for joint, angle in enumerate([0., -.45, 0., -2.10, 0., 1.75, .75]):
                p.resetJointState(self.robot, joint, angle, physicsClientId=self.client)
            for joint in (9, 10):
                p.resetJointState(self.robot, joint, .04, physicsClientId=self.client)
                p.changeDynamics(self.robot, joint, lateralFriction=2., spinningFriction=.02,
                                 physicsClientId=self.client)
            self.cup = self._load_mug(cup_id, xy, yaw)
            joints = [p.getJointState(self.robot, j, physicsClientId=self.client)[0] for j in range(7)]
            self._arm_targets(joints)
            self._gripper(.08)
            for _ in range(120):
                p.stepSimulation(physicsClientId=self.client)
            self.initial_center_z = p.getBasePositionAndOrientation(self.cup, physicsClientId=self.client)[0][2]
            self.rules = LiftRules(expected_dt=DT)
            self.trace.append(self._telemetry())
            if self.record:
                self.frames.append(self._camera("front")["rgb"])
        except BaseException:
            self.close()
            raise

    def _load_mug(self, cup_id, xy, yaw):
        p, cid = self.p, self.client
        asset = ASSETS / f"mug_{cup_id:03d}"
        meta = json.loads((asset / "surface_asset.json").read_text())
        checks = json.loads((asset / "collision_checks.json").read_text())
        paths = {"collision": asset / "collision.obj", **{
            part: asset / f"{part}_visual.obj" for part in ("body", "handle")}}
        hashes = {"collision": checks["collision_sha256"], **meta["visual_sha256"]}
        if checks["accepted"] is not True:
            raise ValueError("collision asset is not accepted")
        for name, path in paths.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name]:
                raise ValueError(f"asset hash mismatch: {name}")
        height = meta["source_shape"]["height"]
        # A rotated world AABB overestimates clearance requirements differently
        # for different grasps. Use mesh support vertices in the rigid base frame.
        vertices = [list(map(float, line.split()[1:4]))
                    for line in paths["collision"].read_text().splitlines() if line.startswith("v ")]
        self.collision_vertices = np.unique(np.asarray(vertices), axis=0)
        self.collision_vertices[:, 2] -= height/2
        source = ROOT / "data_examples/mug_blender_course_v1/collection" / f"mug_{cup_id:03d}/asset.json"
        linear = np.array(json.loads(source.read_text())["material_linear_rgb"])
        rgb = np.where(linear <= .0031308, linear * 12.92, 1.055 * linear ** (1/2.4) - .055)
        shift = [0, 0, -height/2]
        visual = [p.createVisualShape(p.GEOM_MESH, fileName=str(paths[part]),
                    visualFramePosition=shift, rgbaColor=[*rgb, 1.],
                    specularColor=[.25]*3, physicsClientId=cid) for part in ("body", "handle")]
        collision = p.createCollisionShape(p.GEOM_MESH, fileName=str(paths["collision"]),
                    collisionFramePosition=shift, physicsClientId=cid)
        cup = p.createMultiBody(baseMass=.25, baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual[0], basePosition=[*xy, height/2],
            baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]),
            linkMasses=[0], linkCollisionShapeIndices=[-1], linkVisualShapeIndices=[visual[1]],
            linkPositions=[[0, 0, 0]], linkOrientations=[[0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0]], linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0], linkJointTypes=[p.JOINT_FIXED], linkJointAxis=[[0, 0, 0]],
            physicsClientId=cid)
        p.changeDynamics(cup, -1, lateralFriction=1., spinningFriction=.02,
            rollingFriction=.001, collisionMargin=.0002, physicsClientId=cid)
        return cup

    def _arm_targets(self, joints):
        self.p.setJointMotorControlArray(self.robot, list(range(7)), self.p.POSITION_CONTROL,
            targetPositions=list(joints[:7]), forces=[220.]*7, positionGains=[.08]*7,
            physicsClientId=self.client)

    def _gripper(self, opening_m):
        self.p.setJointMotorControlArray(self.robot, [9, 10], self.p.POSITION_CONTROL,
            targetPositions=[opening_m/2]*2, forces=[25.]*2, physicsClientId=self.client)

    def _telemetry(self):
        p, cid = self.p, self.client
        pos, quat = p.getBasePositionAndOrientation(self.cup, physicsClientId=cid)
        linear, angular = p.getBaseVelocity(self.cup, physicsClientId=cid)
        # All collision geometry is in base link -1; link 0 is visual-only.
        rotation = np.asarray(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        bottom = np.min(self.collision_vertices @ rotation[2]) + pos[2] - .0002
        contacts = p.getContactPoints(bodyA=self.cup, physicsClientId=cid)
        hand = forbidden = False
        for c in contacts:
            if c[9] <= .01:
                continue
            allowed = c[2] == self.robot and c[4] in (8, 9, 10, 11)
            hand = hand or allowed
            forbidden = forbidden or not allowed
        return dict(time_s=self.tick * DT, center_z_m=float(pos[2]), bottom_z_m=float(bottom),
            linear_speed_m_s=float(np.linalg.norm(linear)), angular_speed_rad_s=float(np.linalg.norm(angular)),
            gripper_contact=bool(hand), forbidden_contact=bool(forbidden))

    def _advance(self, ticks):
        self._ensure_ticks(ticks)
        for _ in range(ticks):
            self.p.stepSimulation(physicsClientId=self.client)
            self.tick += 1
            self.trace.append(self._telemetry())
            if self.record and self.tick % 24 == 0:
                self.frames.append(self._camera("front")["rgb"])

    def _ensure_ticks(self, ticks):
        if self.client is None or self.finished:
            raise RuntimeError("episode has finished or closed")
        if type(ticks) is not int or ticks <= 0:
            raise ValueError("ticks must be a positive integer")
        if self.tick + ticks > self.max_ticks:
            raise ValueError("simulated-time budget exceeded")

    def _camera_parameters(self, view_id):
        if view_id not in VIEWS:
            raise ValueError("unknown camera view")
        p, size = self.p, self.size
        azimuth, elevation = VIEWS[view_id]
        azimuth, elevation = math.radians(azimuth), math.radians(elevation)
        target, distance = np.array([.38, -.02, .12]), .90
        eye = target + distance*np.array([math.cos(elevation)*math.cos(azimuth),
            math.cos(elevation)*math.sin(azimuth), math.sin(elevation)])
        view = p.computeViewMatrix(eye.tolist(), target.tolist(), [0, 0, 1])
        near, far, fov = .02, 2.5, 52.
        proj = p.computeProjectionMatrixFOV(fov, 1., near, far)
        gl_world_to_camera = np.asarray(view).reshape(4, 4, order="F")
        cv_world_to_camera = np.diag([1., -1., -1., 1.]) @ gl_world_to_camera
        focal = size / (2 * math.tan(math.radians(fov)/2))
        camera = {"view_id": view_id, "K": np.array([[focal, 0, size/2],
                [0, focal, size/2-1], [0, 0, 1.]]),
                "camera_to_world": np.linalg.inv(cv_world_to_camera),
                "width": size, "height": size,
                "convention": "OpenCV x right y down z forward; integer TinyRenderer raster"}
        return view, proj, near, far, camera

    def _camera(self, view_id):
        p, size = self.p, self.size
        view, proj, near, far, camera = self._camera_parameters(view_id)
        result = p.getCameraImage(size, size, view, proj, renderer=p.ER_TINY_RENDERER,
            flags=p.ER_NO_SEGMENTATION_MASK, physicsClientId=self.client)
        rgb = np.array(result[2], dtype=np.uint8).reshape(size, size, 4)[..., :3].copy()
        zbuffer = np.array(result[3], dtype=np.float64).reshape(size, size)
        depth = (far * near / (far - (far - near) * zbuffer)).astype(np.float32)
        valid = zbuffer < .9999
        depth[~valid] = 0.
        return {"rgb": rgb, "depth_m": depth, "depth_valid": valid,
            "camera": camera}

    def observe(self, view_id="front"):
        """Host sensor snapshot. Rendering does not advance or reset physics."""
        obs = self._camera(view_id)
        p, cid = self.p, self.client
        tcp = p.getLinkState(self.robot, 11, computeForwardKinematics=True, physicsClientId=cid)
        obs["robot"] = {"joint_positions_rad": np.array([
            p.getJointState(self.robot, j, physicsClientId=cid)[0] for j in range(7)]),
            "tcp_position_world": np.array(tcp[4]), "tcp_quaternion_xyzw": np.array(tcp[5]),
            "opening_m": sum(p.getJointState(self.robot, j, physicsClientId=cid)[0] for j in (9, 10))}
        obs["budget"] = {"sim_seconds_remaining": (self.max_ticks-self.tick)*DT,
            "additional_views_remaining": 2-self.view_requests}
        obs["available_views"] = list(VIEWS)
        obs["view_catalog"] = {view: self._camera_parameters(view)[4] for view in VIEWS}
        return obs

    def request_view(self, view_id):
        if type(view_id) is not str or view_id not in VIEWS:
            raise ValueError("view_id must name a public camera")
        if self.client is None:
            raise RuntimeError("episode has closed")
        if self.finished or self.mode not in (None, "active_view"):
            raise ValueError("view requests require an active-view episode")
        if self.view_requests >= 2:
            raise ValueError("additional-view budget exceeded")
        observation = self.observe(view_id)
        self.view_requests += 1
        observation["budget"]["additional_views_remaining"] = 2-self.view_requests
        return observation

    def _move(self, pose, ticks):
        self._ensure_ticks(ticks)
        joints = self.p.calculateInverseKinematics(self.robot, 11,
            targetPosition=pose.position_world, targetOrientation=pose.quaternion_xyzw,
            maxNumIterations=250, residualThreshold=1e-5, physicsClientId=self.client)
        self._arm_targets(joints)
        self._gripper(pose.opening_m)
        self._advance(ticks)

    def execute_grasp(self, pose):
        """Fixed public one-shot executor; orientation is never overwritten."""
        self._ensure_ticks(1190)
        if self.mode not in (None, "one_shot", "active_view"):
            raise ValueError("one-shot grasp is not an action in this mode")
        if not isinstance(pose, GraspPose):
            raise ValueError("pose must be a GraspPose")
        rotation = np.array(self.p.getMatrixFromQuaternion(pose.quaternion_xyzw)).reshape(3, 3)
        xyz = np.array(pose.position_world)
        pre = GraspPose(tuple(xyz - rotation[:, 2]*pose.approach_distance_m),
                        pose.quaternion_xyzw, pose.opening_m, 0.)
        lift = GraspPose(tuple(xyz + [0, 0, .24]), pose.quaternion_xyzw, .004, 0.)
        # Validate all waypoints before making any movement.
        if self.tick + 1190 > self.max_ticks:
            raise ValueError("insufficient time budget for grasp and terminal hold")
        self._move(pre, 220)
        self._move(pose, 170)
        self._gripper(.004)
        self._advance(240)
        self._move(lift, 260)
        return self.finish()

    def step(self, pose):
        """B: one 100 ms Cartesian command, bounded from the current TCP."""
        self._ensure_ticks(324)
        if self.mode not in (None, "closed_loop"):
            raise ValueError("Cartesian step is not an action in this mode")
        if not isinstance(pose, GraspPose):
            raise ValueError("pose must be a GraspPose")
        tcp = self.p.getLinkState(self.robot, 11, computeForwardKinematics=True, physicsClientId=self.client)
        if np.linalg.norm(np.array(pose.position_world)-tcp[4]) > .03 + 1e-9:
            raise ValueError("Cartesian step exceeds 3 cm")
        dot = abs(np.dot(pose.quaternion_xyzw, tcp[5]))
        if 2*math.acos(np.clip(dot, -1, 1)) > math.radians(15) + 1e-6:
            raise ValueError("orientation step exceeds 15 degrees")
        if self.tick + 24 + 300 > self.max_ticks:
            raise ValueError("time budget must reserve terminal hold")
        self._move(pose, 24)
        return self.observe()

    def finish(self):
        """Hold the last motor targets for 1.25 seconds, then evaluate all ticks."""
        self._advance(300)
        self.finished = True
        return evaluate_lift(self.trace, self.initial_center_z, rules=self.rules)

    def close(self):
        if getattr(self, "client", None) is not None:
            if self.p.isConnected(self.client):
                self.p.disconnect(physicsClientId=self.client)
            self.client = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def run_policy(env, policy, mode="one_shot"):
    """Trusted local development dispatch, not isolated student-code hosting.

    ``policy.act(observation)`` returns finish, view, grasp, or move dictionaries.
    Exceptions deliberately propagate for the host to classify. Never interpret
    a worker's own reported success as evaluator truth.
    """
    if mode not in ("one_shot", "closed_loop", "active_view"):
        raise ValueError("unknown task mode")
    if env.mode is not None or env.finished or env.client is None:
        raise ValueError("run_policy requires a fresh unassigned environment")
    env.mode = mode
    obs = env.observe()
    while True:
        action = policy.act(obs)
        if not isinstance(action, dict) or "type" not in action:
            raise ValueError("action must be a dictionary with type")
        kind = action["type"]
        if type(kind) is not str:
            raise ValueError("action type must be a string")
        allowed = {"finish"} | ({"move"} if mode == "closed_loop" else {"grasp"})
        if mode == "active_view":
            allowed.add("view")
        if kind not in allowed:
            raise ValueError(f"{kind!r} is not an allowed action in {mode}")
        if kind == "finish":
            if set(action) != {"type"}:
                raise ValueError("unexpected finish fields")
            return env.finish()
        if kind == "view":
            if set(action) != {"type", "view_id"}:
                raise ValueError("view action requires only type and view_id")
            obs = env.request_view(action["view_id"])
            continue
        fields = {"position_world", "quaternion_xyzw", "opening_m"}
        if kind == "grasp":
            fields.add("approach_distance_m")
        if set(action) != fields | {"type"}:
            raise ValueError(f"{kind} requires fields: {sorted(fields)}")
        pose = GraspPose(**{key: action[key] for key in fields})
        if kind == "grasp":
            return env.execute_grasp(pose)
        obs = env.step(pose)
