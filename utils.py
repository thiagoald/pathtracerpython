#!/usr/bin/python3 env

from ipdb.__main__ import set_trace
import numpy as np
import ipdb
import colorama
from colorama import Fore, Back, Style
from PIL import Image

colorama.init(autoreset=True)


class NoIntersection(Exception):
    pass

ZERO = 1E-1


def squared_dist(pt1, pt2):
    return sum([(c2 - c1)*(c2 - c1) for c2, c1 in zip(pt1, pt2)])

def lerp(pt1, pt2, t):
    return (1-t)*np.array(pt1) + t*np.array(pt2)


def make_rays(start_pt, pts):
    rays = []
    for pt in pts:
        v = np.array(pt)-np.array(start_pt)
        # v = v/np.linalg.norm(v)
        rays.append((start_pt, v))
    return rays


def make_screen_pts(x0, y0, x1, y1, n_pxls_x, n_pxls_y):
    pts = []
    for x in np.linspace(x0, x1, n_pxls_x):
        for y in np.linspace(y0, y1, n_pxls_y):
            pts.append((x, y, 0))
    return pts


def in_triangle(pt, triangle):
    # check if point is in triangle through cross product between edges AB, BC, AC and segments AP, BP, CP
    if pt is None:
        return False
    v1, v2, v3 = triangle
    v1 = np.array(v1)
    v2 = np.array(v2)
    v3 = np.array(v3)
    p = np.array(pt)
    cross1 = np.cross(v1-v2, p-v2)
    cross2 = np.cross(v2-v3, p-v3)
    cross3 = np.cross(v3-v1, p-v1)
    cross1 = cross1/np.linalg.norm(cross1)
    cross2 = cross2/np.linalg.norm(cross2)
    cross3 = cross3/np.linalg.norm(cross3)
    dot12 = np.dot(cross1, cross2)
    dot13 = np.dot(cross1, cross3)
    sign12 = np.sign(dot12)
    sign13 = np.sign(dot13)
    return sign12 > 0 and sign13 > 0


def f(exp):
    return Fore.GREEN + f'{exp:.2f}' + Style.RESET_ALL if abs(exp) > ZERO else Fore.RED + f'{exp:.2f}' + Style.RESET_ALL


def intersect(ray, triangle):
    # Vertices
    v1, v2, v3 = triangle
    x1, y1, z1 = v1  # Vertex 1
    x2, y2, z2 = v2  # Vertex 2
    x3, y3, z3 = v3  # Vertex 3

    # Ray point and vector
    pr, vr = ray
    x_pr, y_pr, z_pr = pr  # Point
    x_vr, y_vr, z_vr = vr  # Vector

    vr = vr/np.linalg.norm(vr)  # normalize vector of line
    # get director vector of plane
    v_plane = np.cross((np.array(v1)-np.array(v2)),
                       (np.array(v3)-np.array(v2)))
    v_plane = v_plane/np.linalg.norm(v_plane)  # normalize vector of plane
    # check if they're not collinear (line parallel to plane)
    dot = np.dot(vr, v_plane)
    if (abs(dot) > 1e-5):
        t = (np.dot(v_plane, np.array(v1)) - np.dot(v_plane, np.array(pr))
             )/np.dot(v_plane, vr)  # calculate intersection parameter
        P = np.array(pr) + vr*t  # calculate intersection point
    else:
        P = None  # in collinear case, there's no intersection point

    if in_triangle(P, triangle):

        '''debug
        print(Fore.WHITE + Back.GREEN + 'IN TRIANGLE')
        print ('VERTEXES')
        print (v1)
        print(v2)
        print(v3)
        print ('INTERSECTION P')
        print (P)
        '''
        return P

    else:
        '''debug
        print(Fore.WHITE + Back.RED + 'NOT IN TRIANGLE')
        print ('VERTEXES')
        print (v1)
        print(v2)
        print(v3)
        print ('INTERSECTION P')
        print (P)
        '''
        raise NoIntersection


def make_image(x1, y1, x2, y2, width, height, intersections):
    mat = np.zeros((height, width, 3), dtype='uint8')
    for pt_3d, pt_2d, color in intersections:
        if pt_3d is not None:
            x, y, _ = pt_2d
            i = int(((y-y1)/(y2-y1))*height)
            j = int(((x-x1)/(x2-x1))*width)
            mat[i, j] = np.array(color).astype('uint8')*255
    return Image.fromarray(mat)
