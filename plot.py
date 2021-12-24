import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from functools import reduce


def init():
    app = pg.mkQApp()
    w = gl.GLViewWidget()
    w.opts['distance'] = 20
    w.show()
    w.setWindowTitle('stl-surface')
    return w


def show():
    pg.mkQApp().exec_()


def plot_objects(objects, colors_pg=[(1, 0, 0, 1),
                                     (0, 1, 0, 1),
                                     (0, 0, 1, 1)]):
    w = init()
    for i, object in enumerate(objects):
        triangles = np.array(object['geometry'].triangles)
        plot_triangles_3d(w, triangles, colors_pg[i % len(colors_pg)])
    show()


def plot_triangles_3d(widget, triangles, color=(1, 0, 0, 1)):
    points = []
    points_colors = []
    for i, (v1, v2, v3) in enumerate(triangles):
        points.extend([v1, v1, v2, v3, v1, v1])
        points_colors.extend(
            [(0, 0, 0, 0)] + [color]*4 + [(0, 0, 0, 0)])
    polygon = gl.GLLinePlotItem(pos=np.array(
        points), color=np.array(points_colors), glOptions='translucent')
    widget.addItem(polygon)
