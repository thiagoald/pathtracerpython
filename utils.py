#!/usr/bin/python3 env

from ipdb.__main__ import set_trace
import numpy as np
import ipdb
import colorama
from colorama import Fore, Back, Style
from PIL import Image
from random import uniform

colorama.init(autoreset=True)


class NoIntersection(Exception):
    pass


ZERO = 1E-5


def sample_bary_coords():
    '''Return barycentric coordinates'''
    coords = [uniform(0, 1) for _ in range(3)]
    sum_ = sum(coords)
    return [c/sum_ for c in coords]


def pick_random_triangle(areas):
    '''Return index of triangle picked'''
    n = uniform(0, sum(areas))
    numbers = [0]
    acc = 0
    for a in areas:
        acc += a
        numbers.append(acc)
    for i, interval in enumerate(zip(numbers, numbers[1:])):
        n1, n2 = interval
        if (n1 <= n < n2):
            return i


def sample_random_pt(triangle):
    '''Return a point sampled in the triangle'''
    v1, v2, v3 = triangle
    a, b, c = sample_bary_coords()
    return a*np.array(v1) + b*np.array(v2) + c*np.array(v3)

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
    if (abs(dot) > ZERO):
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
    mat = np.zeros((height, width, 3), dtype='float64')
    counter =0 
    for color, _ in intersections:
        i=counter//width
        j = counter%width
        mat[height-1-j, i] = np.array(color)
        counter+=1
    mat = mat - np.min(mat)
    mat = mat / np.max(mat)
    mat *= 255
    return Image.fromarray(mat.astype('uint8'))


def bbox_to_triangles(bbox):
    triangles = []
    (x0, y0, z0), (x1, y1, z1) = bbox
    pt0 = np.array((x0, y0, z0))
    pt1 = np.array((x1, y1, z1))
    dx, dy, dz = (np.array([(x1 - x0), 0, 0]),
                  np.array([0, (y1 - y0), 0]),
                  np.array([0, 0, (z1 - z0)]))
    # x_min
    triangles.extend([[pt0, (pt0 + dy), (pt0 + dy + dz)],
                      [pt0, (pt0 + dz), (pt0 + dy + dz)]])
    # x_max
    triangles.extend([[pt1, (pt1 - dy), (pt1 - dy - dz)],
                      [pt1, (pt1 - dz), (pt1 - dy - dz)]])
    # y_min
    triangles.extend([[pt0, (pt0 + dx), (pt0 + dx + dz)],
                      [pt0, (pt0 + dz), (pt0 + dx + dz)]])
    # y_max
    triangles.extend([[pt1, (pt1 - dx), (pt1 - dx - dz)],
                      [pt1, (pt1 - dz), (pt1 - dx - dz)]])
    # z_min
    triangles.extend([[pt0, (pt0 + dy), (pt0 + dy + dx)],
                      [pt0, (pt0 + dx), (pt0 + dy + dx)]])
    # z_max
    triangles.extend([[pt1, (pt1 - dy), (pt1 - dy - dx)],
                      [pt1, (pt1 - dx), (pt1 - dy - dx)]])
    return triangles
