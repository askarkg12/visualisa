import yourdfpy
import numpy as np
from yourdfpy import Joint
from cli_tools import select_link
from transform import create_tf

with open("data/gemini.urdf", "r") as f:
    urdf = yourdfpy.urdf.URDF.load(f)


# Assuming every joint has 100 steps
steps = 100
random_steps = tuple(np.random.randint(0, steps, size=5))


pass
