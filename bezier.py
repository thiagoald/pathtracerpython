#!/usr/bin/env python3

from multiprocessing import Value
import numpy as np
from utils import NoIntersection, cpu_intersect, bbox_to_triangles


def lerp(pt1, pt2, t):
    return (1-t)*pt1 + t*pt2


class BezierCurve:
    def __init__(self, ctrlpts):
        self.ctrlpts = np.array(ctrlpts)

    def eval(self, t, split=False):
        if split:
            leftpts, rightpts = [], []
        interpolations = self.ctrlpts
        while len(interpolations) != 1:
            if split:
                leftpts.append(interpolations[0])
                rightpts.append(interpolations[-1])
            interpolations = [lerp(pt1, pt2, t) for pt1, pt2 in zip(
                interpolations, interpolations[1:])]
        if split:
            leftpts.append(interpolations[0])
            rightpts.append(interpolations[0])
            return (interpolations[0],
                    BezierCurve(leftpts),
                    BezierCurve(rightpts))
        else:
            return interpolations[0]

    def split(self, t):
        return self.eval(t, split=True)[1:]

    def translate(self, vector):
        self.ctrlpts = [pt + vector for pt in self.ctrlpts]
        return self


class BezierSurface:
    def __init__(self, ctrlpts):
        self.ctrlpts = np.array(ctrlpts)
        self.children = None
        self.delta_u = None
        self.delta_v = None

    def eval_normal(self, u, v):
        if self.delta_u is None:
            self.delta_u = self.delta_surface('u')
            self.delta_v = self.delta_surface('v')
        tan_u = self.delta_u.eval(u, v)
        tan_v = self.delta_v.eval(u, v)
        normal = np.cross(tan_u, tan_v)
        normal /= np.linalg.norm(normal)
        return normal

    def delta_surface(self, u_or_v='u'):
        if u_or_v not in ['u', 'v']:
            raise ValueError(f'u_or_v should be "u" or "v" (not {u_or_v}).')
        n_i, n_j, _ = self.ctrlpts.shape
        if u_or_v == 'u':
            n_i -= 1
        else:
            n_j -= 1
        new_ctrlpts = np.zeros((n_i, n_j, 3))
        for i in range(n_i):
            for j in range(n_j):
                if u_or_v == 'u':
                    new_ctrlpts[i, j] = self.ctrlpts[i+1, j] - \
                        self.ctrlpts[i, j]
                else:
                    new_ctrlpts[i, j] = self.ctrlpts[i, j+1] - \
                        self.ctrlpts[i, j]
        return BezierSurface(new_ctrlpts)

    def bbox(self):
        x_min = np.min(self.ctrlpts[:, :, 0])
        x_max = np.max(self.ctrlpts[:, :, 0])
        y_min = np.min(self.ctrlpts[:, :, 1])
        y_max = np.max(self.ctrlpts[:, :, 1])
        z_min = np.min(self.ctrlpts[:, :, 2])
        z_max = np.max(self.ctrlpts[:, :, 2])
        return (x_min, y_min, z_min), (x_max, y_max, z_max)

    def bbox_deltas(self):
        pt0, pt1 = self.bbox()
        deltas = []
        for c0, c1 in zip(pt0, pt1):
            deltas.append(c1-c0)
        return deltas

    def transposed(self):
        n_i, n_j, n_pt = self.ctrlpts.shape
        new_ctrlpts = np.zeros((n_j, n_i, n_pt))
        for i in range(n_i):
            for j in range(n_j):
                new_ctrlpts[j, i] = self.ctrlpts[i, j]
        return BezierSurface(new_ctrlpts)

    def eval(self, u, v):
        ctrlpts = []
        for i in range(self.ctrlpts.shape[0]):
            curve = BezierCurve(self.ctrlpts[i, :])
            ctrlpts.append(curve.eval(u))
        return BezierCurve(ctrlpts).eval(v)

    def split_u(self, u=0.5):
        ctrlpts_mat_1 = []
        ctrlpts_mat_2 = []
        for i in range(len(self.ctrlpts)):
            curve = BezierCurve(self.ctrlpts[i, :])
            c_left, c_right = curve.split(u)
            ctrlpts_mat_1.append(c_left.ctrlpts)
            ctrlpts_mat_2.append(c_right.ctrlpts)
        return BezierSurface(ctrlpts_mat_1), BezierSurface(ctrlpts_mat_2)

    def split_uv(self, u=0.5, v=0.5):
        if self.children is not None:
            return self.children
        s1, s2 = self.split_u(u)
        s1_1, s1_2 = s1.transposed().split_u(v)
        s2_1, s2_2 = s2.transposed().split_u(v)
        return s1_1.transposed(), s1_2, s2_1, s2_2.transposed()

    def translated(self, vector):
        n_i, n_j, _ = self.ctrlpts.shape
        new_ctrlpts = np.zeros(self.ctrlpts.shape)
        for i in range(n_i):
            for j in range(n_j):
                new_ctrlpts[i, j] = self.ctrlpts[i, j] + vector
        return BezierSurface(new_ctrlpts)

    def scaled(self, scale):
        n_i, n_j, _ = self.ctrlpts.shape
        new_ctrlpts = np.zeros(self.ctrlpts.shape)
        for i in range(n_i):
            for j in range(n_j):
                new_ctrlpts[i, j] = self.ctrlpts[i, j]*scale
        return BezierSurface(new_ctrlpts)

    def intersect_bbox(self, ray):
        triangles = bbox_to_triangles(self.bbox())
        intersections = []
        for tri in triangles:
            try:
                intersections.append(cpu_intersect(ray, tri))
            except NoIntersection:
                pass
        if len(intersections) == 0:
            raise NoIntersection
        else:
            return intersections

    def intersect(self, ray, max_delta=0.1, widget=None):
        from plot import plot_bbox, plot_surface
        try:
            self.intersect_bbox(ray)
        except NoIntersection:
            raise NoIntersection

        stack = [self]
        found = False
        while stack != []:
            parent = stack.pop()
            try:
                parent.intersect_bbox(ray)
                if parent.within_delta(max_delta):
                    found = True
                    break
            except NoIntersection:
                pass
            else:
                children = parent.split_uv()
                for c in children:
                    try:
                        if widget is not None:
                            plot_surface(widget, c)
                            plot_bbox(widget, c)
                        stack.append(c)
                    except NoIntersection:
                        pass

        if found:
            return (parent.eval(0.5, 0.5), parent.eval_normal(0.5, 0.5))
        else:
            raise NoIntersection

    def within_delta(self, max_delta):
        return all([d <= max_delta for d in self.bbox_deltas()])
