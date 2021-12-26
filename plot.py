import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from functools import reduce
import ipdb

from utils import make_rays, make_screen_pts

WHITE_OPAQUE = (1, 1, 1, 1)
WHITE_TRANSLUCENT = (1, 1, 1, 0.2)
RED_OPAQUE = (1, 0, 0, 1)
GREEN_OPAQUE = (0, 1, 0, 1)


def init():
    app = pg.mkQApp()
    w = gl.GLViewWidget()
    w.opts['distance'] = 100
    w.show()
    w.setWindowTitle('stl-surface')
    return w


def show():
    pg.mkQApp().exec_()


def plot_scene(scene, rays, data,
               show_normals=False, show_screen=False, show_inter=False):
    w = init()
    if not show_inter:
        plot_objects(w, scene.objects, normals=show_normals)
    plot_camera(w, scene.eye)
    # plot_rays(w, rays, 40)
    if show_inter:
        intersection_data = [y for x, y in data]
        plot_intersections(w,intersection_data)
    if show_screen:
        screen_data = [x for x, y in data]
        list_of_2d_colors = screen_data
        plot_screen(w,*scene.ortho, scene.width, scene.height, list_of_2d_colors)
    show()


def plot_objects(widget, objects, normals=False):
    for i, object in enumerate(objects):
        color = [object[c] for c in ('red', 'green', 'blue')] + [1]
        triangles = np.array(object['geometry'].triangles)
        plot_triangles_3d(widget, triangles, color)
        if normals:
            plot_normals(widget, triangles,
                         object['geometry'].normals, color=color)


def plot_normals(widget, triangles, normals, color=GREEN_OPAQUE, size_endpoint=3):
    endpoints = []
    for normal, vertices in zip(normals, triangles):
        pt_avg = sum([np.array(v) for v in vertices])/len(vertices)
        widget.addItem(gl.GLLinePlotItem(
            pos=np.array([pt_avg, pt_avg + normal]),
            color=color))
        endpoints.append(pt_avg + normal)
    widget.addItem(gl.GLScatterPlotItem(pos=np.array(endpoints),
                                        color=color, size=size_endpoint))


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


def plot_screen(widget, x0, y0, x1, y1, n_pxls_x, n_pxls_y, list_of_2d_colors):
    pts = make_screen_pts(x0, y0, x1, y1, n_pxls_x, n_pxls_y)
    print (np.array(list_of_2d_colors))
    widget.addItem(gl.GLScatterPlotItem(pos=np.array(
        pts), size=3, color=np.array(list_of_2d_colors), glOptions='translucent'))


def plot_rays(widget, rays, t, color=WHITE_TRANSLUCENT):
    for pt, vector in rays:
        widget.addItem(gl.GLLinePlotItem(
            pos=np.array([pt, pt + t*vector]), color=color, glOptions='translucent'))


def plot_intersections(widget, intersection_lists):
    
    colors = [[color for point, color in intersection_list] for intersection_list in intersection_lists]
    pts_3d = [[point for point, color in intersection_list] for intersection_list in intersection_lists]
    #print (np.array(colors))
    widget.addItem(gl.GLScatterPlotItem(pos=np.array(pts_3d),
                                        size=3,
                                        color=np.array(colors),
                                        glOptions='translucent'))
    '''
    if screen_pts:
        pts_2d = [p for _, p in intersections]
        widget.addItem(gl.GLScatterPlotItem(pos=np.array(pts_2d),
                                            size=3,
                                            color=np.array(colors),
                                            glOptions='translucent'))
    '''