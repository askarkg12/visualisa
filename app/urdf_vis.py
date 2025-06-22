from transform import RobotResolver
from yourdfpy import Link
import pyvista as pv
import random
import numpy as np

colours = [
    "red",
    "green",
    "blue",
    "yellow",
    "purple",
    "orange",
    "pink",
    "brown",
    "gray",
    "cyan",
    "magenta",
    "lime",
]


def populate_plotter_with_robot(plotter: pv.Plotter, robot_resolver: RobotResolver):
    for link in robot_resolver.urdf.robot.links:
        link: Link
        if link.visuals is not None:
            for visual in link.visuals:
                if visual.geometry is not None:
                    if visual.geometry.box is not None:
                        bounds = np.empty(6)
                        bounds[0::2] = -visual.geometry.box.size / 2
                        bounds[1::2] = visual.geometry.box.size / 2
                        mesh = pv.Box(bounds)

                        # Resolve transform from visual to link
                        if visual.origin is None:
                            box_to_link = np.eye(4)
                        else:
                            box_to_link = visual.origin

                        # Resolve link to base_link
                        if link.name == "base_link":
                            link_to_base = np.eye(4)
                        else:
                            # create chain for this link
                            chain = []
                            for joint in robot_resolver.joints:
                                chain.append(joint)
                                if joint.child == link.name:
                                    break
                            link_to_base = robot_resolver.resolve_chain_rad([0] * len(chain))

                        # Combine transforms
                        tf = link_to_base @ box_to_link
                        mesh.transform(tf)

                        plotter.add_mesh(mesh, color=random.choice(colours))
