import numpy as np
from functools import cache
from yourdfpy import Joint, URDF
from cli_tools import select_link


def rotation_x(angle: float) -> np.ndarray:
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle), np.cos(angle), 0],
            [0, 0, 0, 1],
        ]
    )


def rotation_y(angle: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(angle), 0, np.sin(angle), 0],
            [0, 1, 0, 0],
            [-np.sin(angle), 0, np.cos(angle), 0],
            [0, 0, 0, 1],
        ]
    )


def rotation_z(angle: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def rotation_xyz(axis: np.ndarray, angle: float) -> np.ndarray:
    r = np.eye(4)
    if axis[0] != 0:
        r = r @ rotation_x(axis[0] * angle)
    if axis[1] != 0:
        r = r @ rotation_y(axis[1] * angle)
    if axis[2] != 0:
        r = r @ rotation_z(axis[2] * angle)
    return r


def create_tf_rad(joint: Joint, joint_value: float) -> np.ndarray:
    t_fixed = joint.origin
    if joint.type == "fixed":
        return t_fixed
    elif joint.type == "revolute":
        return t_fixed @ rotation_xyz(joint.axis, joint_value)
    elif joint.type == "prismatic":
        raise NotImplementedError("Prismatic joints are not supported")
    else:
        raise ValueError(f"Unknown joint type: {joint.type}")


def create_tf_step(
    joint: Joint, discrete_step: float, total_steps: int = 360
) -> np.ndarray:
    if joint.limit is None:
        # Assume 360 degrees with step of 1 degree
        step_value = np.pi / 180
        lower_limit = 0
    else:
        step_value = (joint.limit.upper - joint.limit.lower) / total_steps
        lower_limit = joint.limit.lower

    joint_value = discrete_step * step_value + lower_limit

    t_fixed = joint.origin
    if joint.type == "fixed":
        return t_fixed
    elif joint.type == "revolute":
        return t_fixed @ rotation_xyz(joint.axis, joint_value)
    elif joint.type == "prismatic":
        raise NotImplementedError("Prismatic joints are not supported")
    else:
        raise ValueError(f"Unknown joint type: {joint.type}")


class RobotResolver:
    def __init__(self, urdf: URDF, joint_steps: list[int] | int | None = None):
        self.urdf = urdf
        if joint_steps is None:
            self.joint_steps = [100] * len(self.urdf.robot.joints)
        elif isinstance(joint_steps, int):
            self.joint_steps = [joint_steps] * len(self.urdf.robot.joints)
        elif isinstance(joint_steps, list):
            self.joint_steps = joint_steps
        else:
            raise ValueError("Invalid joint steps")
        base_link_name = self.urdf.base_link

        links = {link.name: link for link in self.urdf.robot.links}
        ef_link = select_link(list(links.values()))

        child_2_joints: dict[str, Joint] = {
            joint.child: joint for joint in self.urdf.robot.joints
        }

        # Build chain from EF link to base link
        self.joint_chain: list[Joint] = [child_2_joints[ef_link.name]]

        while self.joint_chain[0].parent != base_link_name:
            self.joint_chain.insert(0, child_2_joints[self.joint_chain[0].parent])

        self.resolve_chains: set[tuple[int]] = set()

    def sample_end_effector_pos(self) -> np.ndarray | None:
        steps = tuple(np.random.randint(0, self.joint_steps).tolist())
        if steps in self.resolve_chains:
            return None
        self.resolve_chains.add(steps)
        tf = self.resolve_chain_steps(steps)
        xyz = tf[:3, 3]
        return xyz

    @property
    def joints(self) -> list[Joint]:
        return self.urdf.robot.joints

    @joints.setter
    def joints(self, joints: list[Joint]):
        self.urdf.robot.joints = joints
        self.resolve_chain_steps.cache_clear()

    @cache
    def resolve_chain_steps(self, discrete_steps: tuple[int] | int) -> np.ndarray:
        if len(discrete_steps) > len(self.joint_chain):
            raise ValueError("Too many discrete steps")
        if len(discrete_steps) == 0:
            return np.eye(4)
        elif len(discrete_steps) == 1:
            return create_tf_step(
                self.joint_chain[len(discrete_steps) - 1],
                discrete_steps[0],
                self.joint_steps[len(discrete_steps) - 1],
            )
        else:
            chain = self.resolve_chain_steps(discrete_steps[:-1])
            tf = create_tf_step(
                self.joint_chain[len(discrete_steps) - 1],
                discrete_steps[-1],
                self.joint_steps[len(discrete_steps) - 1],
            )
            return chain @ tf

    # No need to cache this, its not used often
    def resolve_chain_rad(self, radians: list[float]) -> np.ndarray:
        if len(radians) > len(self.joint_chain):
            raise ValueError("Too many radians")
        if len(radians) == 0:
            return np.eye(4)
        elif len(radians) == 1:
            return create_tf_rad(
                self.joint_chain[len(radians) - 1],
                radians[0],
            )
        else:
            chain = self.resolve_chain_rad(radians[:-1])
            tf = create_tf_rad(
                self.joint_chain[len(radians) - 1],
                radians[-1],
            )
            return chain @ tf
