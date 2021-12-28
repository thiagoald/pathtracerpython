#!/usr/bin/python3 env

from numba import cuda
from ipdb.__main__ import set_trace
import numpy as np
from numba.typed import List

import ipdb
import colorama
from colorama import Fore, Back, Style
from PIL import Image
from random import uniform
import math
import scipy    

colorama.init(autoreset=True)


class NoIntersection(Exception):
    pass


ZERO = 1E-10


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

@cuda.jit(device=True)
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


@cuda.jit(device=True)
def in_triangle(a12, a23, a31, ap2, ap3, ap1):
    '''checks if point P is in triangle ABC through cross product between edges AB, BC, AC and segments AP, BP, CP'''
    
    cross2 = cuda.local.array(3, 'float64')
    cross3 = cuda.local.array(3, 'float64')
    cross1 = cuda.local.array(3, 'float64')
    
    cross(a12, ap2, cross2)
    cross(a23, ap3, cross3)
    cross(a31, ap1, cross1)
    
    dot12 = dot(cross1, cross2) #order here does not matter!
    dot13 = dot(cross1, cross3)
    return dot12>0 and dot13>0


def f(exp):
    return Fore.GREEN + f'{exp:.2f}' + Style.RESET_ALL if abs(exp) > ZERO else Fore.RED + f'{exp:.2f}' + Style.RESET_ALL

@cuda.jit(device=True)
def cross(a, b,c):
    c[0] = a[1]*b[2] - a[2]*b[1]
    c[1] = a[2]*b[0] - a[0]*b[2]
    c[2] = a[0]*b[1] - a[1]*b[0]

@cuda.jit(device=True)
def dot (a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@cuda.jit(device=True)
def norm(a):
    n=math.sqrt(dot(a, a))
    a[0]=a[0]/n
    a[1]=a[1]/n
    a[2]=a[2]/n


@cuda.jit
def intersect(data):
    
    pos = cuda.grid(1)
    if pos<data.size:
        a12 = cuda.local.array(3, 'float64') #edge between points 1 and 2
        a12[0] = data[pos,0,0] - data[pos,1,0]
        a12[1] = data[pos,0,1] - data[pos,1,1]
        a12[2] = data[pos,0,2] - data[pos,1,2]
        
        a23 = cuda.local.array(3, 'float64') #edge between points 2 and 3
        a23[0] = data[pos,1,0] - data[pos,2,0]
        a23[1] = data[pos,1,1] - data[pos,2,1]
        a23[2] = data[pos,1,2] - data[pos,2,2]
        
        a31 = cuda.local.array(3, 'float64') #edge between points 3 and 1
        a31[0] = data[pos,2,0] - data[pos,0,0]
        a31[1] = data[pos,2,1] - data[pos,0,1]
        a31[2] = data[pos,2,2] - data[pos,0,2]
        
        v_plane = cuda.local.array(3, 'float64') #the director vector of the plane
        v_plane[0] = data[pos,0,3]
        v_plane[1] = data[pos,1,3]
        v_plane[2] = data[pos,2,3]
        norm(v_plane)
        
        vector_ray = cuda.local.array(3, 'float64') #the ray vector
        vector_ray[0] = data[pos,0,6]
        vector_ray[1] = data[pos,1,6]
        vector_ray[2] = data[pos,2,6]
        norm(vector_ray)
        
        point_ray = cuda.local.array(3, 'float64') #the ray initial point
        point_ray[0] = data[pos, 0, 5]
        point_ray[1] = data[pos, 1, 5]
        point_ray[2] = data[pos, 2, 5]
        
        p1 = cuda.local.array(3, 'float64') # a triangle point
        p1[0] = data[pos,0,0]
        p1[1] = data[pos,0,1]
        p1[2] = data[pos,0,2]
        
        p_final = cuda.local.array(3, 'float64') # the intersection point
        
        if abs(dot(vector_ray, v_plane)) > ZERO:
            t = (dot(v_plane, p1) - dot(v_plane, point_ray))/dot(vector_ray, v_plane)  # calculate intersection parameter
            
            p_final[0]= point_ray[0] + vector_ray[0]*t
            p_final[1]= point_ray[1] + vector_ray[1]*t
            p_final[2]= point_ray[2] + vector_ray[2]*t
            
            ap2 = cuda.local.array(3, 'float64') #edge between points p_final and 2
            ap2[0] = p_final[0] - data[pos,1,0]
            ap2[1] = p_final[1] - data[pos,1,1]
            ap2[2] = p_final[2] - data[pos,1,2]
           
            ap3 = cuda.local.array(3, 'float64') #edge between points p_final and 2
            ap3[0] = p_final[0] - data[pos,2,0]
            ap3[1] = p_final[1] - data[pos,2,1]
            ap3[2] = p_final[2] - data[pos,2,2]
            
            ap1 = cuda.local.array(3, 'float64') #edge between points p_final and 2
            ap1[0] = p_final[0] - data[pos,0,0]
            ap1[1] = p_final[1] - data[pos,0,1]
            ap1[2] = p_final[2] - data[pos,0,2]
            
            if in_triangle(a12, a23, a31, ap2, ap3, ap1): #in_triangle(a12, a23, a31, ap2, ap3, ap1):
                data[pos,0,7] = 1
                data[pos,1,7] = 1
                data[pos,2,7] = 1
                data[pos,0,8] = p_final[0]
                data[pos,1,8] = p_final[1]
                data[pos,2,8] = p_final[2]
            else:
                data[pos,0,7] = -1
                data[pos,1,7] = -1
                data[pos,2,7] = -1
        else:
            data[pos,0,7] = -1
            data[pos,1,7] = -1
            data[pos,2,7] = -1

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
