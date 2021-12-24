from sys import getdlopenflags
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from functools import reduce

from utils import make_rays, make_screen_pts

WHITE_OPAQUE = (1, 1, 1, 1)
RED_OPAQUE = (1, 0, 0, 1)

def init():
    app = pg.mkQApp()
    w = gl.GLViewWidget()
    w.opts['distance'] = 20
    w.show()
    w.setWindowTitle('stl-surface')
    return w


def show():
    pg.mkQApp().exec_()


def plot_scene(scene, rays, choice=None):
    w = init()
    plot_objects(w, scene.objects)
    plot_camera(w, scene.eye)
    plot_screen(w, *scene.ortho, scene.width, scene.height)
    plot_rays(w, rays, 40)
    # plot_intersections(w, intersections)
    show()


def plot_objects(widget, objects, colors_pg=[(1, 0, 0, 1),
                                             (0, 1, 0, 1),
                                             (0, 0, 1, 1)]):
    for i, object in enumerate(objects):
        triangles = np.array(object['geometry'].triangles)
        plot_triangles_3d(widget, triangles, colors_pg[i % len(colors_pg)])


def plot_triangles_3d(widget, triangles, color=WHITE_OPAQUE):
    points = []
    points_colors = []
    for i, (v1, v2, v3) in enumerate(triangles):
        points.extend([v1, v1, v2, v3, v1, v1])
        points_colors.extend(
            [(0, 0, 0, 0)] + [color]*4 + [(0, 0, 0, 0)])
    polygon = gl.GLLinePlotItem(pos=np.array(
        points), color=np.array(points_colors), glOptions='translucent')
    widget.addItem(polygon)


def plot_camera(widget, eye, color=WHITE_OPAQUE):
    x, y, z = eye
    pt = gl.GLScatterPlotItem(
        pos=np.array([(x, y, z)]), color=color, glOptions='translucent')
    widget.addItem(pt)


def plot_screen(widget, x0, y0, x1, y1, n_pxls_x, n_pxls_y, color=WHITE_OPAQUE):
    pts = make_screen_pts(x0, y0, x1, y1, n_pxls_x, n_pxls_y)
    widget.addItem(gl.GLScatterPlotItem(pos=np.array(
        pts), size=1, color=color, glOptions='translucent'))


def plot_rays(widget, rays, t, color=WHITE_OPAQUE):
    for pt, vector in rays:
        widget.addItem(gl.GLLinePlotItem(
            pos=np.array([pt, pt + t*vector]), color=color, glOptions='translucent'))


def plot_intersections(widget, intersections, color=RED_OPAQUE):
    widget.addItem(gl.GLLinePlotItem(pos=np.array(
        intersections), color=color, glOptions='translucent'))
