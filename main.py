#!/usr/bin/env python3

# Example:
#  $ ./main.py objs/cornellroom.sdl --show-scene --show-img --out out.png -r 1 -b 2

from numba import cuda
from numba.extending import overload
from numba.typed import List

from scene_reader import Scene
from plot import plot_scene
from utils import (NoIntersection, make_rays, make_screen_pts, intersect,
                   make_image, sample_random_pt, squared_dist,
                   pick_random_triangle)
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ApplyResult
import numpy as np
import argparse
from random import uniform
import math

TAU = 6.28
ZERO = 1E-5

def compute_shadow_rays(scene, point, normal,obj, n_light_samples=1):
    '''
    Return the color of a point from direct illumination by the light source
    '''
    # Vectors starting at the object and pointing to the sampled light point
    shadow_vectors = []
    areas = scene.light_obj.areas
    for _ in range(n_light_samples):
        i = pick_random_triangle(areas)
        # Random triangle
        light_tri = scene.light_obj.triangles[i]
        # Random point in this triangle
        light_pt = sample_random_pt(light_tri)
        # intersect
        ray_vector = light_pt - point
        ray_vector = ray_vector/np.linalg.norm(ray_vector)
        ray = (point, ray_vector)
        light_squared_distance = squared_dist(point, light_pt)
        done = False
        _, _, _, isItLight = intersect_objects(ray, scene.objects, scene.light_obj)
        if not isItLight:
          shadow_vectors.append(None)
        else:
          shadow_vectors.append(ray_vector)
            

    # compute light color
    light_color = scene.light_color

    # compute color
    dot = 0
    for shadow_vector in shadow_vectors:
        if shadow_vector is not None:
            dot += np.dot(shadow_vector, normal)
    dot /= len(shadow_vectors)
    obj_color = [obj[c] for c in ['red', 'green', 'blue']]
    color = [c_light*c_obj*dot for c_light,
             c_obj in zip(light_color, obj_color)]
    return np.array(color)


def compute_ambient_color(scene, obj):
    color_list = [obj[c]*obj['ka'] *
                  scene.ambient for c in ['red', 'green', 'blue']]
    color_array = np.array(color_list)
    return color_array

'''
def intersection_helper(data):
    pos = cuda.grid(1)
    if pos<data.size:
        triangle = data[pos,:,:2]
        normal = data[pos,:,3]
        obj_index = data[pos,0, 4]
        ray_point = data[pos,:,5].T
        ray_vector = data[pos,:,6].T
        intersections = [[-1.,-1.,-1.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]
        ray = ray_point, ray_vector
        pt_3d = intersect(ray_point, ray_vector, triangle)
        if squared_dist(pt_3d, ray_point)>ZERO:
            pt_screen = ray[0] + ray[1]
            sq= squared_dist(pt_3d, ray[0])
            intersections.append([np.array((sq, sq, sq)),pt_3d,pt_screen, normal ,np.array((obj_index, obj_index, obj_index))])
        else:
            intersections.append([np.array((-1.,-1.,-1.)),np.array((0.,0.,0.)),np.array((0.,0.,0.)),np.array((0.,0.,0.)),np.array((0.,0.,0.))])
        intersections.pop(0)
        intersections = np.array(intersections)
        data[pos]=intersections.reshape(data[pos].shape)
'''

def organize_objs_data(objects, ray):
    ray_point, ray_vector = ray
    objs_data_triangles=[]
    objs_data_normals=[]
    data=[]

    for obj in objects:
        obj_triangles=[]
        obj_normals=[]
        for triangle in obj['geometry'].triangles:
            list_triangle=[]
            for vector in triangle:
                list_vector= []
                for x in np.array(vector).tolist():
                    list_vector.append(x)
                list_triangle.append(list_vector)
            
            obj_triangles.append(list_triangle)
        for normal in obj['geometry'].normals:
            obj_normals.append([normal])
        objs_data_triangles.append(obj_triangles)
        objs_data_normals.append (obj_normals)
    data=[]
    for index, (triangle_data, normal_data) in enumerate(zip(objs_data_triangles, objs_data_normals)):
        for triangle, normal in zip(triangle_data, normal_data):
            np_triangle = np.array(triangle)
            concat_shape = list(np_triangle.shape)
            concat_shape[0]=1
            np_normal = np.array(normal)
            np_point = np.reshape(ray_point, concat_shape)
            np_vector= np.reshape(ray_vector, concat_shape)
            np_vector = np_vector/np.linalg.norm(np_vector)
            np_index = np.resize(np.array((index)),concat_shape)
            np_data = np.concatenate((np_triangle,np_normal.T), axis=1)
            np_data = np.concatenate((np_data,np_index.T), axis=1)
            np_data = np.concatenate((np_data, np_point.T), axis=1)
            np_data = np.concatenate((np_data, np_vector.T), axis=1)
            data.append(np_data)
            
    np_all_data = np.array(data)
    concat_shape = list (np_all_data.shape)
    concat_shape[-1]=2
    np_all_data=np.append(np_all_data, np.zeros(concat_shape), axis=len(concat_shape)-1)
    return np_all_data

def unpack_obj_data(data):
    intersection_info = []
    for i in range (data.shape[0]):
        obj_data = data[i] #data for one object
        found_flag = obj_data[0,7]
        pt_3d = obj_data[:,8]
        obj_index = int(obj_data[0,4])
        normal = obj_data[:,3]
        sq = np.linalg.norm(pt_3d - obj_data[:,5])
        if found_flag>0:
            intersection_info.append([sq, pt_3d, normal, obj_index])
    return intersection_info
    
def intersect_objects(ray, objects, light_obj):
    '''Return intersection point, triangle normal and object'''
    # this is actually important because, when ray is none, we still have to
    # count it so results align
    # (otherwise we'd skip it and mix up different pixel's paths)
    
    if ray is None:
        return None
    point_ray, vector_ray = ray
    point_ray = np.array(point_ray)
    # adds light to object list
    myObjects = objects + [{'geometry': light_obj}]
    # returns nearest intersecting object
    data = organize_objs_data(myObjects, ray)
    threadsperblock = 32 
    blockspergrid = (data.size + (threadsperblock - 1))
    intersections = intersect[blockspergrid,threadsperblock](data)
    intersections = unpack_obj_data(data)
    
    if len(intersections)==0:
        return None
    closest_inter = min(intersections, key=lambda inter: inter[0])
    i_obj = closest_inter[-1]
    obj = myObjects[i_obj]
    if obj['geometry'] == light_obj:
        isItLight = True
    else:
        isItLight = False
    return (closest_inter[1], closest_inter[2], obj, isItLight)


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene', help='SDL scene')
    parser.add_argument('--out', help='Output image')
    parser.add_argument('-r', dest='n_rays', type=int, default=1,
                        help='Number of rays per pixel')
    parser.add_argument('-b', dest='n_bounces', type=int, default=1,
                        help='Number of bounces')
    parser.add_argument('--show-img', default=False, action='store_true')
    parser.add_argument('--show-scene', default=False, action='store_true')
    parser.add_argument('--show-normals', default=False, action='store_true')
    parser.add_argument('--show-screen', default=False, action='store_true')
    parser.add_argument('--show-inter', default=False, action='store_true')
    args = parser.parse_args()
    return args


def compute_color(scene, obj, point, normal):
    amb = compute_ambient_color(scene, obj)
    #sha = compute_shadow_rays(scene, point, normal, obj)
    return ( amb )

def rotate(axis, angle, v):
    """
    Return the counterclockwise rotation about
    the given axis by 'angle' radians.
    """
    if (np.linalg.norm(axis)<ZERO):
        return v
    axis = axis / np.linalg.norm(axis)
    a = math.cos(angle / 2.0)
    b, c, d = -axis * math.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rotation_matrix = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    return np.dot(rotation_matrix, v)

def main():
    args = setup()
    scene = Scene(args.scene)
    screen_pts = make_screen_pts(
        *scene.ortho, scene.width, scene.height)
    print(f'Number of objects: {len(scene.objects)}')
    print(
        f'Number of triangles: {sum([len(o["geometry"].triangles) for o in scene.objects])}')
    results = []
    how_many_rays = args.n_rays
    how_many_bounces = args.n_bounces    
    # initialization
    colored_intersections=[]
    for _ in range(scene.width*scene.height):
        initial_color = np.array((0, 0, 0))
        initial_list_of_3d_points_and_colors = [
            (np.array((0, 0, 0)), np.array((0, 0, 0)),)]
        colored_intersections.append(
            (initial_color, initial_list_of_3d_points_and_colors,))
    # iterative (non-recursive) path tracing
    pixel_color_list=[0]*(scene.width*scene.height)
    for rays_counter in range(how_many_rays):
        print('rays_counter is ' + str(rays_counter))
        colored_intersections = [(initial_color, initial_list_of_3d_points_and_colors) for intersec in colored_intersections]
        accumulated_k = np.ones(scene.width*scene.height)
        rays = make_rays(scene.eye, screen_pts)
        for bounces_counter in range(how_many_bounces):
            print('bounces_counter is ' + str(bounces_counter))
            # compute intersections
            results = []
            print('intersecting...')       
            for i, ray in tqdm(list(enumerate(rays))):
                results.append(intersect_objects(ray, scene.objects, scene.light_obj))                    
            # compute colors
            print('calculating colors...')
            new_colors = []
            for i_ray, result in tqdm(list(enumerate(results))):
                if result is not None:
                    point, normal, obj, isItLight = result
                    if isItLight:
                        new_color = np.array(scene.light_color)
                    else:
                        new_color = compute_color(scene, obj, point, normal)
                    old_color, _ = colored_intersections[i_ray]
                    colored_intersections[i_ray] = (old_color+new_color*accumulated_k[i_ray], [(point, new_color,)])

            # now we create new rays
            old_rays = rays
            rays = []
            for i_ray, result in enumerate(results):
                if result is not None:
                    point, normal, obj, isItLight = result
                    if not isItLight:
                        ray_type_randomness = uniform(0, obj['kd']+obj['ks'])
                        if ray_type_randomness <= obj['kd']:  # Case 1: diffuse
                            phi = np.arccos(math.sqrt(uniform(0, 1)))
                            theta = TAU*uniform(0, 1)
                            ray_vector = np.array((np.sin(phi)*np.cos(theta),
                                                   np.sin(phi)*np.sin(theta),
                                                   np.cos(phi)))
                            ray_vector = ray_vector/np.linalg.norm(ray_vector)
                            local_axis = np.cross(np.array((0, 1, 0)), normal)
                            ray_vector = rotate(local_axis, np.arccos(np.dot(np.array((0,1,0)), normal)), ray_vector)
                            rays.append((point, ray_vector))
                            accumulated_k[i_ray] *= obj['kd'] * \
                                np.dot(ray_vector, normal)
                        else:
                            _, old_ray_vector = old_rays[i_ray] #Case 2: specular
                            ray_vector = np.dot(normal, old_ray_vector)*2*normal - old_ray_vector
                            ray_vector = ray_vector/np.linalg.norm(ray_vector)
                            eye_vector = scene.eye - point
                            eye_vector = eye_vector/np.linalg.norm(eye_vector)
                            local_axis = np.cross(np.array((0, 1, 0)), normal)
                            ray_vector = rotate(local_axis, np.arccos(np.dot(np.array((0,1,0)), normal)), ray_vector)
                            rays.append((point, ray_vector))
                            accumulated_k[i_ray] *= obj['ks'] * \
                                ((np.dot(eye_vector, ray_vector))**obj['n'])
                    else:
                        rays.append(None)
                else:
                    rays.append(None)
        for i, intersec in enumerate(colored_intersections):
            pixel_color, list_of_3d_points_and_colors = intersec
            pixel_color_list[i]+=pixel_color

    # Here we average the rays
    temp_intersections = []
    for i, intersec in enumerate(colored_intersections):
        _, list_of_3d_points_and_colors = intersec
        pixel_color = pixel_color_list[i]/how_many_rays
        temp_intersections += [(pixel_color, list_of_3d_points_and_colors,)]
    
    colored_intersections = temp_intersections
    if args.show_scene:
        plot_scene(scene, rays, colored_intersections,
                   show_normals=args.show_normals,
                   show_screen=args.show_screen,
                   show_inter=args.show_inter)

    im = make_image(*scene.ortho, scene.width,
                    scene.height, colored_intersections)
    if args.out is not None:
        im.save(args.out)
    if args.show_img:
        im.show()


if __name__ == '__main__':
    main()
