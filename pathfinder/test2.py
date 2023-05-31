from mayavi import mlab
from world import World

world = World()
mlab.figure(size=(800, 800))
mlab.mesh(world.x, world.y, world.z)


def drawPoint(x, y, scale_factor=0.02, color=(1, 0, 0)):
    mlab.points3d(world.lin_x[x], world.lin_y[y], world.z[y][x], 0, scale_mode='none', scale_factor=scale_factor,
                  color=color)


@mlab.animate(delay=50)
def anim():
    for i in range(6000):
        drawPoint(50 + i, 500, color=(0, 1, 0))
        yield


ua = anim()
mlab.show()
