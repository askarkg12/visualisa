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
        r = r @ rotation_x(axis[0])
    if axis[1] != 0:
        r = r @ rotation_y(axis[1])
    if axis[2] != 0:
        r = r @ rotation_z(axis[2])
    return r


def create_tf(joint: Joint, discrete_step: float, total_steps: int = 360) -> np.ndarray:
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
        return (rotation_xyz(joint.axis, joint_value) @ t_fixed).astype(np.float32)
    elif joint.type == "prismatic":
        raise NotImplementedError("Prismatic joints are not supported")
    else:
        raise ValueError(f"Unknown joint type: {joint.type}")


class RobotResolver:
    def __init__(self, urdf: URDF):
        self.urdf = urdf
        base_link_name = self.urdf.base_link

        links = {link.name: link for link in self.urdf.robot.links}
        ef_link = select_link(list(links.values()))
        base_link = links[base_link_name]

        child_2_joints: dict[str, Joint] = {
            joint.child: joint for joint in self.urdf.robot.joints
        }

        # Build chain from EF link to base link
        joint_chain: list[Joint] = [child_2_joints[ef_link.name]]

        while joint_chain[-1].parent != base_link_name:
            joint_chain.append(child_2_joints[joint_chain[-1].parent])

    @property
    def joints(self) -> list[Joint]:
        return self.urdf.robot.joints

    @joints.setter
    def joints(self, joints: list[Joint]):
        self.urdf.robot.joints = joints
        self.resolve_chain.cache_clear()

    @cache
    def resolve_chain(self, discrete_steps: tuple[int] | int) -> list[np.ndarray]:
        if len(discrete_steps) == 0:
            return [np.eye(4)]
        else:
            return self.resolve_chain(discrete_steps[1:]) @ create_tf(
                self.joints[discrete_steps[0]], discrete_steps[0]
            )
