import pyvista as pv
import os
import sys
import time
import yourdfpy
import numpy as np
from transform import RobotResolver
from tqdm import tqdm
from scipy.spatial import KDTree
from urdf_vis import populate_plotter_with_robot
import threading
import queue
from pyvistaqt import BackgroundPlotter
import psutil

CMAP = "plasma"
OPACITY = 0.7
THROTTLE_MULT = 1

points_queue = queue.Queue()

with open("data/gemini.urdf", "r") as f:
    urdf = yourdfpy.urdf.URDF.load(f)

robot_resolver = RobotResolver(urdf)


def set_priority(priority: str = "low"):
    try:
        if sys.platform.startswith("win"):
            p = psutil.Process()
            if priority == "low":
                p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            elif priority == "normal":
                p.nice(psutil.NORMAL_PRIORITY_CLASS)
            elif priority == "high":
                p.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            if priority == "low":
                os.nice(10)
            elif priority == "normal":
                os.nice(5)
            elif priority == "high":
                os.nice(0)
    except ImportError:
        pass


def background_worker(q: queue.Queue):
    set_priority("low")
    for _ in tqdm(range(500_000)):
        start = time.perf_counter()
        sample = robot_resolver.sample_end_effector_pos()
        duration = time.perf_counter() - start
        if sample is not None:
            q.put(sample)

        time.sleep(duration * THROTTLE_MULT)


worker_thread = threading.Thread(target=background_worker, args=(points_queue,), daemon=True)
worker_thread.start()

points_list = []

# Create initial points
for _ in tqdm(range(1_000)):
    sample = robot_resolver.sample_end_effector_pos()
    # Its ok if we end up sampling less
    if sample is not None:
        points_list.append(sample)

if points_list:
    points = np.array(points_list)
else:
    points = np.empty((0, 3))


tree = KDTree(points)
k = 20
distances, _ = tree.query(points, k=k)
density = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-8)
cloud = pv.PolyData(points)
cloud["density [1/m]"] = density
clipped = cloud

# Plot the robot in 3D

plotter = BackgroundPlotter()
populate_plotter_with_robot(plotter, robot_resolver)


actor_container = {}  # dictionary to hold the active actor reference
active_clip_box = None


# Add the full point cloud initially
def refresh_opacity(opacity):
    global clipped, OPACITY
    OPACITY = opacity

    plotter.remove_actor(actor_container["actor"], reset_camera=False)

    # Add new clipped mesh and store new actor reference
    new_actor = add_cloud(clipped)
    actor_container["actor"] = new_actor


def add_cloud(cloud: pv.PolyData | pv.UnstructuredGrid):
    if isinstance(cloud, pv.UnstructuredGrid):
        cloud_pv = pv.PolyData(cloud.points)
        cloud_pv["density [1/m]"] = cloud["density [1/m]"]
        cloud = cloud_pv

    if cloud.n_points == 0:
        return None
    return plotter.add_mesh(
        cloud,
        # render_points_as_spheres=True,
        style="points_gaussian",
        point_size=5,
        scalars="density [1/m]",
        opacity=OPACITY,
        cmap=CMAP,
    )


actor = add_cloud(cloud)
actor_container["actor"] = actor  # store reference for later replacement

plotter.add_axes()
# plotter.add_title("Point Cloud Density with Interactive Section Box")


# Step 5: Define clipping callback
def clip_with_box(box):
    global clipped, active_clip_box
    active_clip_box = box
    clipped = cloud.clip_box(box, invert=False)

    # Remove previous actor
    if "actor" in actor_container and actor_container["actor"] is not None:
        plotter.remove_actor(actor_container["actor"], reset_camera=False)

    # Add new clipped mesh and store new actor reference
    new_actor = add_cloud(clipped)
    actor_container["actor"] = new_actor


def update_cloud():
    global points, cloud, clipped, tree, actor_container, k, active_clip_box
    new_points = []
    while not points_queue.empty():
        try:
            new_points.append(points_queue.get_nowait())
        except queue.Empty:
            break

    if not new_points:
        return

    new_points = np.array(new_points)
    points = np.concatenate([points, new_points])

    tree = KDTree(points)
    distances, _ = tree.query(points, k=k)
    density = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-8)

    cloud = pv.PolyData(points)
    cloud["density [1/m]"] = density

    if active_clip_box is not None:
        clipped = cloud.clip_box(active_clip_box, invert=False)
    else:
        clipped = cloud

    plotter.remove_actor(actor_container["actor"], reset_camera=False)
    actor_container["actor"] = add_cloud(clipped)


def foo(flag):
    update_cloud()


def update_throttle(value):
    global THROTTLE_MULT
    THROTTLE_MULT = value


# Step 6: Enable interactive box widget
plotter.add_box_widget(callback=clip_with_box, bounds=cloud.bounds, rotation_enabled=False)
plotter.add_slider_widget(update_throttle, [0.001, 10], value=THROTTLE_MULT, title="Throttle")
plotter.add_checkbox_button_widget(callback=foo, value=True)
# plotter.add_timer_event(callback=update_cloud, duration=3000, max_steps=10)

plotter.app.exec()
