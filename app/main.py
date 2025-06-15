from yourdfpy import Link
import random
import pyvista as pv
import yourdfpy
import numpy as np
from transform import RobotResolver
from tqdm import tqdm
import threading
from scipy.spatial import cKDTree

CMAP = "plasma"
OPACITY = 0.7


with open("data/gemini.urdf", "r") as f:
    urdf = yourdfpy.urdf.URDF.load(f)

robot_resolver = RobotResolver(urdf)

points_list = []

# Create initial points
for _ in tqdm(range(100_00)):
    sample = robot_resolver.sample_end_effector_pos()
    # Its ok if we end up sampling less
    if sample is not None:
        points_list.append(sample)

points = np.array(points_list)


tree = cKDTree(points)
k = 20
distances, _ = tree.query(points, k=k)
density = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-8)
cloud = pv.PolyData(points)
cloud["density [1/m]"] = density
clipped = cloud

# Plot the robot in 3D

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

plotter = pv.Plotter()
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
                        link_to_base = robot_resolver.resolve_chain_rad(
                            [0] * len(chain)
                        )

                    # Combine transforms
                    tf = link_to_base @ box_to_link
                    mesh.transform(tf)

                    plotter.add_mesh(mesh, color=random.choice(colours))


actor_container = {}  # dictionary to hold the active actor reference


# Add the full point cloud initially
def refresh_opacity(opacity):
    global clipped
    plotter.remove_actor(actor_container["actor"], reset_camera=False)

    # Add new clipped mesh and store new actor reference
    new_actor = add_cloud(clipped, opacity)
    actor_container["actor"] = new_actor


def add_cloud(cloud, opacity=OPACITY):
    return plotter.add_mesh(
        cloud,
        render_points_as_spheres=True,
        point_size=5,
        scalars="density [1/m]",
        opacity=opacity,
        cmap=CMAP,
    )


actor = add_cloud(cloud)
actor_container["actor"] = actor  # store reference for later replacement

plotter.add_axes()
# plotter.add_title("Point Cloud Density with Interactive Section Box")


# Step 5: Define clipping callback
def clip_with_box(box):
    global clipped
    clipped = cloud.clip_box(box, invert=False)

    # Remove previous actor
    plotter.remove_actor(actor_container["actor"], reset_camera=False)

    # Add new clipped mesh and store new actor reference
    new_actor = add_cloud(clipped)
    actor_container["actor"] = new_actor


def background_worker():
    global points, cloud

    points_buffer = []

    for _ in range(10_000_000):
        sample = robot_resolver.sample_end_effector_pos()
        # Its ok if we end up sampling less
        if sample is not None:
            points_buffer.append(sample)

        if len(points_buffer) % 30_000 == 0:
            points = np.concatenate([points, np.array(points_buffer)])
            points_buffer = []
            cloud = pv.PolyData(points)


# threading.Thread(target=background_worker).start()

# Step 6: Enable interactive box widget
plotter.add_box_widget(
    callback=clip_with_box, bounds=cloud.bounds, rotation_enabled=False
)
plotter.add_slider_widget(refresh_opacity, [0, 1], value=OPACITY, title="Opacity")
plotter.show()
