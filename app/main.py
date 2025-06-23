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
from PySide6.QtWidgets import QMainWindow


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
for _ in tqdm(range(100)):
    sample = robot_resolver.sample_end_effector_pos()
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


plotter = BackgroundPlotter(show=True)
populate_plotter_with_robot(plotter, robot_resolver)


actor_containers = {}
active_clip_box = None


def refresh_opacity(opacity):
    global clipped, OPACITY
    OPACITY = opacity

    refresh_cloud(clipped)


def refresh_cloud(cloud: pv.PolyData | pv.UnstructuredGrid):
    if "cloud" in actor_containers and actor_containers["cloud"] is not None:
        plotter.remove_actor(actor_containers["cloud"], reset_camera=False)
    if isinstance(cloud, pv.UnstructuredGrid):
        cloud_pv = pv.PolyData(cloud.points)
        cloud_pv["density [1/m]"] = cloud["density [1/m]"]
        cloud = cloud_pv

    if cloud.n_points == 0:
        return None
    new_actor = plotter.add_mesh(
        cloud,
        # render_points_as_spheres=True,
        style="points_gaussian",
        point_size=5,
        scalars="density [1/m]",
        opacity=OPACITY,
        cmap=CMAP,
    )
    actor_containers["cloud"] = new_actor


refresh_cloud(cloud)

plotter.add_axes()


def clip_with_box(box):
    update_cloud_from_queue()
    global clipped, active_clip_box
    active_clip_box = box
    clipped = cloud.clip_box(box, invert=False)

    refresh_cloud(clipped)


def update_cloud_from_queue(_=None):
    global points, cloud, clipped, tree, actor_containers, k, active_clip_box
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

    refresh_cloud(clipped)


def update_throttle(value):
    global THROTTLE_MULT
    THROTTLE_MULT = value


plotter.add_box_widget(
    callback=clip_with_box,
    bounds=cloud.bounds if cloud.n_points > 0 else None,
    rotation_enabled=False,
)
plotter.add_slider_widget(update_throttle, [0.001, 10], value=THROTTLE_MULT, title="Throttle")

# Let system autopopulate the cloud for first 5 seconds
plotter.add_callback(update_cloud_from_queue, interval=500, count=30)

# main_window = QMainWindow()
# main_window.setCentralWidget(plotter)
# main_window.setWindowTitle("Point Cloud Density with Interactive Section Box")


menu_bar = plotter.main_menu


def foo():
    print("foo")


action_menu = menu_bar.addMenu("Actions")
foo_action = action_menu.addAction("Foo")
foo_action.triggered.connect(foo)

# main_window.show()


plotter.app.exec()
