import yourdfpy
from transform import RobotResolver
from tqdm import tqdm

with open("data/gemini.urdf", "r") as f:
    urdf = yourdfpy.urdf.URDF.load(f)

robot_resolver = RobotResolver(urdf)

points_list = []
for _ in tqdm(range(10_000)):
    if robot_resolver.sample_end_effector() is not None:
        points_list.append(robot_resolver.sample_end_effector())


pass
