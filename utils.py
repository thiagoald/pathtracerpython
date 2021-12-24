#!/usr/bin/python3 env

import numpy as np
import ipdb
import colorama
from colorama import Fore, Back, Style

colorama.init(autoreset=True)

ZERO = 1E-1


def lerp(pt1, pt2, t):
    return (1-t)*np.array(pt1) + t*np.array(pt2)


def make_rays(start_pt, pts):
    rays = []
    for pt in pts:
        v = np.array(pt)-np.array(start_pt)
        v = v/np.linalg.norm(v)
        rays.append((start_pt, v))
    return rays


def make_screen_pts(x0, y0, x1, y1, n_pxls_x, n_pxls_y):
    pts = []
    for x in np.linspace(x0, x1, n_pxls_x):
        for y in np.linspace(y0, y1, n_pxls_y):
            pts.append((x, y, 0))
    return pts


def barycentric_coords(pt, v1, v2, v3):
    # Cramer's rule
    x1, y1, z1 = v1
    x2, y2, z2 = v2
    x3, y3, z3 = v3
    xp, yp, zp = pt
    den = x1*y2*z3 + x2*y3*z1 + x3*y1*z2 - (x3*y2*z1 + x1*y3*z2 + x2*y1*z3)

    # Alpha
    num = xp*y2*z3 + x2*y3*zp + x3*yp*z2 - (x3*y2*zp + xp*y3*z2 + x2*yp*z3)
    alpha = num/den

    # Beta
    num = x1*yp*z3 + xp*y3*z1 + x3*y1*zp - (x3*yp*z1 + x1*y3*zp + xp*y1*z3)
    beta = num/den

    # Gamma
    num = x1*y2*zp + x2*yp*z1 + xp*y1*z2 - (xp*y2*z1 + x1*yp*z2 + x2*y1*zp)
    gamma = num/den

    return alpha, beta, gamma


def in_triangle(pt, triangle):
    return abs(sum(barycentric_coords(pt, *triangle)) - 1) < ZERO


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

    # Deltas
    d_x2, d_y2, d_z2 = (x2 - x1), (y2 - y1), (z2 - z1)
    d_x3, d_y3, d_z3 = (x3 - x1), (y3 - y1), (z3 - z1)
    d_xpr, d_ypr, d_zpr = (x_pr - x1), (y_pr - y1), (z_pr - z1)
    print((f'{d_x2=}, {d_y2=}, {d_z2=}\n'
           f'{d_x3=}, {d_y3=}, {d_z3=}\n'
           f'{d_xpr=}, {d_ypr=}, {d_zpr=}\n'))

    # t (intersection parameter)
    num = d_y2*d_z3*d_xpr + d_z2*d_x3*d_ypr + d_x2*d_y3*d_zpr
    print(
        f'num: {f(num)} = {f(d_y2)}*{f(d_z3)}*{f(d_xpr)} + {f(d_z2)}*{f(d_x3)}*{f(d_ypr)} + {f(d_x2)}*{f(d_y3)}*{f(d_zpr)}')
    denom = x_vr*d_y2*d_z3 + y_vr*d_z2*d_x3 + z_vr*d_x2*d_y3
    print(
        f'denom: {f(denom)} = {f(x_vr)}*{f(d_y2)}*{f(d_z3)} + {f(y_vr)}*{f(d_z2)}*{f(d_x3)} + {f(z_vr)}*{f(d_x2)}*{f(d_y3)}')
    t = -num/denom
    print(Fore.BLUE + f't = {f(t)}')

    # Intersection point
    P = np.array(pr) + t*np.array(vr)
    print(f'P = ({f(P[0])}, {f(P[1])}, {f(P[2])})')
    print('\n\n')

    alpha, beta, gamma = barycentric_coords(P, v1, v2, v3)
    print(f'{alpha=}, {beta=}, {gamma=}')
    print(f'sum = {sum([alpha, beta, gamma])}')

    if in_triangle(P, triangle):
        print(Fore.WHITE + Back.GREEN + 'IN TRIANGLE')
    else:
        print(Fore.WHITE + Back.RED + 'NOT IN TRIANGLE')

    # # TODO: Check if intersects triangle using barycentric coordinates
    # if in_triangle(P, triangle):
    #     print(Fore.WHITE + Back.GREEN + 'IN TRIANGLE')
    #     return P
    # else:
    #     print(Fore.WHITE + Back.RED + 'NOT IN TRIANGLE')
    #     return None
