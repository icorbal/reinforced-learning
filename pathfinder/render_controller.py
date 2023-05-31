from mayavi import mlab
from scipy.optimize._cobyla_py import synchronized

from world import calculate_next_point


class RenderController:
    def __init__(self):
        self.world = None
        self.points = []
        self.rendered_points = []
        self.start_point = None
        self.dest_point = None
        self.last_point_rendered = None
        self.debug_points = []
        mlab.figure(size=(800, 800))

    def set_world(self, world):
        self.world = world
        mlab.mesh(self.world.x, self.world.y, self.world.z)

        @mlab.animate(delay=50)
        def anim():
            while True:
                self.anim()
                yield

        self.ua = anim()

    @synchronized
    def reset(self):
        self.points = []
        for point in self.rendered_points:
            point.remove()
        self.rendered_points = []
        self.last_point_rendered = None
        if self.start_point is not None:
            self.start_point.remove()
            self.start_point = None
        if self.dest_point is not None:
            self.dest_point.remove()
            self.dest_point = None

    @synchronized
    def anim(self):
        num_rendered = len(self.rendered_points)
        missing_points = len(self.points) - num_rendered
        for i in range(missing_points):
            point = self.points[num_rendered + i]
            level_diff = abs(self.world.calc_level(point) - self.world.calc_level(self.last_point_rendered))
            color = (1, 0, 0) if level_diff > self.world.max_level_diff else (0, 0, 1)
            draw_point = self.drawPoint(point[0], point[1], color=color)
            self.rendered_points.append(draw_point)
            self.last_point_rendered = point

    def add_point(self, point):
        self.points.append(point)

    def set_start_point(self, point):
        self.start_point = self.drawPoint(point[0], point[1], color=(1, 1, 1))
        self.last_point_rendered = point

    def set_dest_point(self, point):
        self.dest_point = self.drawPoint(point[0], point[1], color=(0, 1, 0))

    def drawPoint(self, x, y, scale_factor=0.02, color=(1, 0, 0)):
        return mlab.points3d(
            self.world.lin_x[x],
            self.world.lin_y[y],
            self.world.z[y][x],
            0,
            scale_mode='none',
            scale_factor=scale_factor,
            color=color
        )

    def start(self):
        mlab.show()
