#!/usr/bin/env python3

from bezier import BezierCurve, BezierSurface
import numpy as np
from plot import (init, plot_pts_normals, show, plot_curve, plot_intersections,
                  plot_surface,
                  RED_OPAQUE, BLUE_OPAQUE, WHITE_OPAQUE, GREEN_OPAQUE)
from utils import NoIntersection


def curve(widget):
    ctrlpts = np.random.rand(4, 3)
    curve = BezierCurve(ctrlpts)
    _, left, right = curve.eval(0.5, split=True)
    plot_curve(widget, left.translate(np.array([2, 0, 0])), color=RED_OPAQUE)
    plot_curve(widget, right.translate(np.array([2, 0, 0])), color=BLUE_OPAQUE)
    plot_curve(widget, curve)


def normals(widget, surface, color):
    pts = []
    normals = []
    samples = np.linspace(0, 1, 10)
    for u in samples:
        for v in samples:
            pts.append(surface.eval(u, v))
            normals.append(surface.eval_normal(u, v))
    plot_surface(widget, surface, color=color)
    plot_pts_normals(widget, pts, normals,
                   color_normal=color)


def surface_subdivision(widget):
    np.random.seed(1)
    ctrlpts = np.random.rand(2, 2, 3)
    surface = BezierSurface(ctrlpts).scaled(30)
    surfaces = surface.split_uv()
    colors = [WHITE_OPAQUE, GREEN_OPAQUE, RED_OPAQUE, BLUE_OPAQUE]

    normals(widget, surface.translated(np.array((10, 0, 0))), WHITE_OPAQUE)

    for i, s in enumerate(surfaces):
        normals(widget, s, colors[i % len(colors)])


def surface_intersection(widget):
    np.random.seed(1)
    ctrlpts = np.random.rand(2, 2, 3)
    surface = BezierSurface(ctrlpts).scaled(30)
    base_ray = [np.array((0, 0, 0)), np.array((1, 1, 1))]
    plot_surface(widget, surface, 30)
    n_random_rays = 200
    normals = []
    pts = []
    for _ in range(n_random_rays):
        try:
            ray = [base_ray[0], base_ray[1] + np.random.rand(3)*5]
            ray[1] = ray[1]/np.linalg.norm(ray[1])
            pt, normal = surface.intersect(ray)
            pts.append(pt)
            normals.append(normal)
            plot_intersections(widget, ray, [pt])
        except NoIntersection:
            pass
    plot_pts_normals(widget, pts, normals)


def main():
    w = init()
    surface_intersection(w)
    show()


if __name__ == '__main__':
    main()
