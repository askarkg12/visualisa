import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

# Step 1: Generate normally distributed 3D point cloud
n_points = 10_000
points = np.random.normal(loc=0.0, scale=1.0, size=(n_points, 3))

# Step 2: Estimate density using k-nearest neighbors
tree = cKDTree(points)
k = 20
distances, _ = tree.query(points, k=k)
density = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-8)

# Step 3: Create PyVista PolyData
cloud = pv.PolyData(points)
cloud["density"] = density

# Step 4: Setup plotter and actor container
plotter = pv.Plotter()
actor_container = {}  # dictionary to hold the active actor reference

# Add the full point cloud initially
actor = plotter.add_mesh(
    cloud,
    render_points_as_spheres=True,
    point_size=5,
    scalars="density",
    cmap="viridis",
)
actor_container["actor"] = actor  # store reference for later replacement

plotter.add_axes()
plotter.add_title("Point Cloud Density with Interactive Section Box")


# Step 5: Define clipping callback
def clip_with_box(box):
    clipped = cloud.clip_box(box, invert=False)

    # Remove previous actor
    plotter.remove_actor(actor_container["actor"], reset_camera=False)

    # Add new clipped mesh and store new actor reference
    new_actor = plotter.add_mesh(
        clipped,
        render_points_as_spheres=True,
        point_size=5,
        scalars="density",
        cmap="viridis",
    )
    actor_container["actor"] = new_actor


# Step 6: Enable interactive box widget
plotter.add_box_widget(
    callback=clip_with_box, bounds=cloud.bounds, rotation_enabled=False
)

# Step 7: Show the plot
plotter.show()
