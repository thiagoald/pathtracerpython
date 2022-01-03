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
    a = uniform(0, 1)
    b = uniform(a, 1)
    c = 1-b-a
    sum_= a+b+c
    return a, b, c

def pick_random_triangle(areas):
    '''Return index of triangle picked'''
    n = uniform(0, sum(areas))
    numbers = []
    acc = 0
    for a in areas:
        acc += a
        numbers.append(acc)
    for i, number in enumerate(numbers):
        if n<number:
            return i

def sample_random_pt(triangle):
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
    for i, pt in enumerate(pts):
        v = np.array(pt)-np.array(start_pt)
        v = v/np.linalg.norm(v)
        rays.append((start_pt, v, i))
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

@cuda.jit(device=True)
def sq_distance(a, b):
    return (a[0]-b[0]) * (a[0]-b[0]) + (a[1]-b[1]) * (a[1]-b[1]) + (a[2]-b[2]) * (a[2]-b[2]) 

@cuda.jit(device=True)
def intersect_one_triangle(one_triangle, v_plane, point_ray, vector_ray, out):
    a12 = cuda.local.array(3, 'float64') #edge between points 1 and 2
    a12[0] = one_triangle[0,0] - one_triangle[1,0]
    a12[1] = one_triangle[0,1] - one_triangle[1,1]
    a12[2] = one_triangle[0,2] - one_triangle[1,2]
    
    a23 = cuda.local.array(3, 'float64') #edge between points 2 and 3
    a23[0] = one_triangle[1,0] - one_triangle[2,0]
    a23[1] = one_triangle[1,1] - one_triangle[2,1]
    a23[2] = one_triangle[1,2] - one_triangle[2,2]
    
    a31 = cuda.local.array(3, 'float64') #edge between points 3 and 1
    a31[0] = one_triangle[2,0] - one_triangle[0,0]
    a31[1] = one_triangle[2,1] - one_triangle[0,1]
    a31[2] = one_triangle[2,2] - one_triangle[0,2]
    
    p1 = cuda.local.array(3, 'float64') # a triangle point
    p1[0] = one_triangle[0,0]
    p1[1] = one_triangle[0,1]
    p1[2] = one_triangle[0,2]
    
    p_final = cuda.local.array(3, 'float64') # the intersection point
    
    if abs(dot(vector_ray, v_plane)) > ZERO:
        t = (dot(v_plane, p1) - dot(v_plane, point_ray))/dot(vector_ray, v_plane)  # calculate intersection parameter
        
        if t>=0:
            p_final[0]= point_ray[0] + vector_ray[0]*t
            p_final[1]= point_ray[1] + vector_ray[1]*t
            p_final[2]= point_ray[2] + vector_ray[2]*t
            
            ap2 = cuda.local.array(3, 'float64') #edge between points p_final and 2
            ap2[0] = p_final[0] - one_triangle[1,0]
            ap2[1] = p_final[1] - one_triangle[1,1]
            ap2[2] = p_final[2] - one_triangle[1,2]
           
            ap3 = cuda.local.array(3, 'float64') #edge between points p_final and 2
            ap3[0] = p_final[0] - one_triangle[2,0]
            ap3[1] = p_final[1] - one_triangle[2,1]
            ap3[2] = p_final[2] - one_triangle[2,2]
            
            ap1 = cuda.local.array(3, 'float64') #edge between points p_final and 2
            ap1[0] = p_final[0] - one_triangle[0,0]
            ap1[1] = p_final[1] - one_triangle[0,1]
            ap1[2] = p_final[2] - one_triangle[0,2]
            
            if in_triangle(a12, a23, a31, ap2, ap3, ap1): #in_triangle(a12, a23, a31, ap2, ap3, ap1):
                out[0] = p_final[0]
                out[1] = p_final[1]
                out[2] = p_final[2]
                out[3] = sq_distance(p_final, point_ray)
                out[4] = 1
            else:
                out[4]=-1
        else:
            out[4]=-1
    else:
        out[4]=-1

@cuda.jit(device=True)
def find_closest (intersections):
    sq = 100000000. # something like infinity
    special_index = -1
    for i, one_inter in enumerate(intersections):
        if one_inter[3]<sq and one_inter[4]>0 and one_inter[3]>ZERO: #closer than we have until now AND found
            special_index=i
            sq=one_inter[3]
    return special_index

@cuda.jit
def intersect(objs_data, ray_data, out_data):

    pos_ray = cuda.grid(1)
    point = cuda.local.array(3, 'float64')
    point[0] = ray_data[pos_ray, 0]
    point[1] = ray_data[pos_ray, 1]
    point[2] = ray_data[pos_ray, 2]
    
    vector = cuda.local.array(3, 'float64')
    vector[0] = ray_data[pos_ray, 3]
    vector[1] = ray_data[pos_ray, 4]
    vector[2] = ray_data[pos_ray, 5]
    
    norm(vector)
    
    ray_index = ray_data[pos_ray, 6]
    
    one_triangle_data = cuda.local.array((3,3), 'float64')
    normal_data = cuda.local.array(3, 'float64')
    len_objs_data = 32 #should be int(len(objs_data)), but can't allocate dynamically
    inter_data = cuda.local.array((len_objs_data,9), 'float64') #x, y, z of intersection point + sq distance + found flag + obj index 
    intersect_out = cuda.local.array(5, 'float64') #x, y, z of intersection point + sq distance + found flag
    for triangle_index, triangle in enumerate(objs_data):
        one_triangle_data[0,0] = triangle[0, 0]
        one_triangle_data[0,1] = triangle[0, 1]
        one_triangle_data[0,2] = triangle[0, 2]
        one_triangle_data[1,0] = triangle[1, 0]
        one_triangle_data[1,1] = triangle[1, 1]
        one_triangle_data[1,2] = triangle[1, 2]
        one_triangle_data[2,0] = triangle[2, 0]
        one_triangle_data[2,1] = triangle[2, 1]
        one_triangle_data[2,2] = triangle[2, 2] 
        normal_data[0] = triangle[0,3]
        normal_data[1] = triangle[1,3]
        normal_data[2] = triangle[2,3]
        obj_index = triangle[0, 4]
        intersect_one_triangle(one_triangle_data, normal_data, point, vector, intersect_out)
        inter_data[triangle_index, 0] = intersect_out[0] # x 
        inter_data[triangle_index, 1] = intersect_out[1] # y
        inter_data[triangle_index, 2] = intersect_out[2] # z
        inter_data[triangle_index, 3] = intersect_out[3] # sq distance
        inter_data[triangle_index, 4] = intersect_out[4] # found flag
        inter_data[triangle_index, 5] = normal_data[0] # x of normal
        inter_data[triangle_index, 6] = normal_data[1] # y of normal
        inter_data[triangle_index, 7] = normal_data[2] # z of normal
        inter_data[triangle_index, 8] = obj_index #obj index
    
    special_index = find_closest (inter_data)
    
    
    out_data[pos_ray,0] = inter_data[special_index, 0] #x
    out_data[pos_ray,1] = inter_data[special_index, 1] #y
    out_data[pos_ray,2] = inter_data[special_index, 2] #z
    out_data[pos_ray,3] = inter_data[special_index, 5] #x of normal
    out_data[pos_ray,4] = inter_data[special_index, 6] #y of normal
    out_data[pos_ray,5] = inter_data[special_index, 7] #z of normal
    out_data[pos_ray,6] = inter_data[special_index, 8] #obj index
    out_data[pos_ray,7] = inter_data[special_index, 4] #found flag
    out_data[pos_ray,8] = ray_index #ray index
    

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
    mat = mat/ (mat+np.array((.4, .4, .4)))
    mat *= 255
    return Image.fromarray(mat.astype('uint8'))

