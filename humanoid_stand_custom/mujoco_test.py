"""
Example of how bodies interact with each other. For a body to be able to
move it needs to have joints. In this example, the "robot" is a red ball
with X and Y slide joints (and a Z slide joint that isn't controlled).
On the floor, there's a cylinder with X and Y slide joints, so it can
be pushed around with the robot. There's also a box without joints. Since
the box doesn't have joints, it's fixed and can't be pushed around.
"""
from mujoco_py import load_model_from_xml, MjSim, MjViewer
import math
import os
import numpy as np

MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>
    <option timestep="0.005" />
    <worldbody>
        <body name="box1" pos="1 3 0" euler="0 0 30">
            <geom mass="0.1" size="0.05 0.1 0.05" type="box" rgba="1 0 0 1"/>
        </body>        
        <body name="box2" pos="3 0 0" euler="0 0 45">
            <geom mass="0.1" size="0.05 0.05 0.05" type="box" rgba="0 0 1 1"/>
        </body>
    </worldbody>
</mujoco>
"""

model = load_model_from_xml(MODEL_XML)
sim = MjSim(model)
viewer = MjViewer(sim)
t = 0
while True:
    #sim.data.ctrl[0] = math.cos(t / 10.) * 0.01
    #sim.data.ctrl[1] = math.sin(t / 10.) * 0.01
    t += 1
    box = sim.data.get_body_xpos("box1")
    box_mat = sim.data.get_body_xmat("box1")
    box2 = sim.data.get_body_xpos("box2")
    aux_pos = np.subtract(box2, box)
    res_pos = aux_pos.dot(box_mat)
    print(box_mat)
    print(res_pos)
    sim.step()
    viewer.render()
    if t > 100 and os.getenv('TESTING') is not None:
        break
