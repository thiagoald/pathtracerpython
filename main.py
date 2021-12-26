#!/usr/bin/env python3

# Example:
#  $ ./main.py objs/cornellroom.sdl --show-scene --show-img --out out.png

from scene_reader import Scene
from plot import plot_scene
from utils import (NoIntersection, make_rays, make_screen_pts, intersect,
                   make_image, sample_random_pt, squared_dist,
                   pick_random_triangle)
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np
import argparse
from random import uniform
import math

TAU = 6.28

    
    #compute normal vector (normalized)
    normal_vector = np.array(obj['geometry'].normals[0])
    normal_vector = normal_vector/np.linalg.norm(normal_vector)
    
    
    #compute random shadow rays
    shadow_vector_list=[]
    shadow_vector_quantity=10
    '''TO-DO'''
    
    
    
    #compute colors
    dot=0
    for shadow_vector in shadow_vector_list:
        if shadow_vector is not None:
            dot+=np.dot(shadow_vector, normal_vector)
    dot /= len(shadow_vector)
    color_list = [obj[c]*dot  for c in ['red', 'green', 'blue']]
    color_array = np.array (color_list)
    return color_array
    
def compute_ambient_color(scene, obj):
    color_list = [obj[c]*obj['ka']*scene.ambient for c in ['red', 'green', 'blue']]
    color_array = np.array(color_list)
    return color_array

def intersect_objects(ray, objects):
    if ray is None: #this is actually important because, when ray is none, we still have to count it so results align (otherwise we'd skip it and mix up different pixel's paths)
        return None
    #returns nearest intersecting object
    intersections = []
    for i, obj in enumerate(objects):
        for triangle, normal in zip(obj['geometry'].triangles, obj['geometry'].normals):
            try:
                pt_3d = intersect(ray, triangle)
                if pt_3d is not None:
                    pt_screen = ray[0] + ray[1]
                    intersections.append(
                        [squared_dist(pt_3d, ray[0]), pt_3d, pt_screen, normal, i])
            except NoIntersection:
                pass

    if intersections == []:
        return None
    else:
        # Only return the closest
        closest_inter = min(intersections, key=lambda inter: inter[0])
        i_obj = closest_inter[-1]
        obj = objects[i_obj]
        return (closest_inter[1], closest_inter[3], obj)


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene', help='SDL scene')
    parser.add_argument('--out', help='Output image')
    parser.add_argument('-r', dest='n_rays', type=int,
                        help='Number of rays per pixel')
    parser.add_argument('-b', dest='n_bounces', type=int,
                        help='Number of bounces')
    parser.add_argument('--show-img', default=False, action='store_true')
    parser.add_argument('--show-scene', default=False, action='store_true')
    parser.add_argument('--show-normals', default=False, action='store_true')
    parser.add_argument('--show-screen', default=False, action='store_true')
    parser.add_argument('--show-inter', default=False, action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = setup()
    scene = Scene(args.scene)
    screen_pts = make_screen_pts(
        *scene.ortho, scene.width, scene.height)
    rays = make_rays(scene.eye, screen_pts)
    print(f'Number of objects: {len(scene.objects)}')
    print(
        f'Number of triangles: {sum([len(o["geometry"].triangles) for o in scene.objects])}')
    print(f'Number of rays: {len(rays)}')
    intersections = []
    results = []
    how_many_rays = args.n_rays
    how_many_bounces = args.n_bounces
    accumulated_kln = np.ones(scene.width*scene.height)
    ray_type = ['dif']*(scene.width*scene.height)
    #initialization
    for _ in range (scene.width*scene.height):
            initial_color = np.array((0,0,0))
            list_of_3d_points_and_colors = [(np.array((0,0,0)), np.array((0,0,0)),)]
            colored_intersections.append((initial_color, list_of_3d_points_and_colors,))
    
    #iterative (non-recursive) path tracing
    for rays_counter in range(how_many_rays):
        print ('rays_counter is '+ str(rays_counter))
        for bounces_counter in range (how_many_bounces):
            print ('bounces_counter is '+ str(bounces_counter))
            #compute intersections
            results=[]
            for ray in rays:
                results.append(intersect_objects(ray, scene.objects))
            counter=0
            #compute colors
            for result in results:
                if result is not None:
                    point, _, obj = result
                    old_color, _ = colored_intersections[counter]
                    new_color = compute_ambient_color(scene, obj) #add shadow rays term
                    colored_intersections[counter]=(old_color+new_color*accumulated_kln[counter],[(point,new_color,)],)
                counter+=1
                
            #now we create new rays
            rays=[]
            counter=0
            for result in results:
                if result is not None:
                    point, normal, obj = result
                    ray_type_randomness=0 #toss a random "coin" here. For now, it will always be zero
                    if ray_type_randomness <= obj['kd']: #Case 1: diffuse
                        phi = np.arccos(math.sqrt(uniform(0, 1)))
                        theta = TAU*uniform(0, 1)
                else:
                    rays.append(None)
                counter+=1
    
    #Here we average the rays
    temp_intersections=[]
    for intersec in colored_intersections:
        pixel_color, list_of_3d_points_and_colors = intersec
        pixel_color = [c/how_many_rays for c in pixel_color]
        temp_intersections += [(pixel_color, list_of_3d_points_and_colors,)]
        
    colored_intersections=temp_intersections
    
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
