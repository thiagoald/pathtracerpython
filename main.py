#!/usr/bin/env python3

# Example:
#  $ ./main.py objs/cornellroom.sdl --show-scene --show-img --out out.png -r 1 -b 2

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

def compute_shadow_rays(scene, point, normal, n_light_samples=3):
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
        for obj in scene.objects:
            for triangle in obj['geometry'].triangles:
                try:
                    pt_3d = intersect(ray, triangle)
                    inter_squared_distance = squared_dist(pt_3d, point)
                    if inter_squared_distance < ZERO: #intersection point is the point itself
                        continue
                    if pt_3d is not None and inter_squared_distance < light_squared_distance:
                        done = True
                        break
                except NoIntersection:
                    pass
            if done:
                break
        if done:
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
    color = [c_light*c_obj*dot*obj['kd'] for c_light,
             c_obj in zip(light_color, obj_color)]
    return np.array(color)


def compute_ambient_color(scene, obj):
    color_list = [obj[c]*obj['ka'] *
                  scene.ambient for c in ['red', 'green', 'blue']]
    color_array = np.array(color_list)
    return color_array


def intersect_objects(ray, objects, light_obj):
    '''Return intersection point, triangle normal and object'''
    # this is actually important because, when ray is none, we still have to
    # count it so results align
    # (otherwise we'd skip it and mix up different pixel's paths)
    if ray is None:
        return None
    # adds light to object list
    myObjects = objects + [{'geometry': light_obj}]
    # returns nearest intersecting object
    intersections = []
    for i, obj in enumerate(myObjects):
        for triangle, normal in zip(obj['geometry'].triangles,
                                    obj['geometry'].normals):
            try:
                pt_3d = intersect(ray, triangle)
                ray_point, ray_vector = ray
                if pt_3d is not None and squared_dist(pt_3d, ray_point)>ZERO:
                    pt_screen = ray[0] + ray[1]
                    intersections.append(
                        [squared_dist(pt_3d, ray[0]),
                         pt_3d,
                         pt_screen,
                         normal,
                         i])
            except NoIntersection:
                pass

    if intersections == []:
        return None
    else:
        # Only return the closest
        closest_inter = min(intersections, key=lambda inter: inter[0])
        i_obj = closest_inter[-1]
        obj = myObjects[i_obj]
        if obj['geometry'] == light_obj:
            isItLight = True
        else:
            isItLight = False
        return (closest_inter[1], closest_inter[3], obj, isItLight)


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
    sha = compute_shadow_rays(scene, point, normal)
    return ( amb +sha )


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
            with Pool(cpu_count()) as pool:
                print('creating threads...')
                for ray in tqdm(rays):
                    results.append(pool.apply_async(
                        intersect_objects, (ray, scene.objects, scene.light_obj)))
                print('getting results...')
                for i in tqdm(range(len(results))):
                    results[i] = results[i].get()

            # compute colors
            print('calculating colors...')
            with Pool(cpu_count()) as pool:
                new_colors = []
                print('creating threads...')
                for i_ray, result in tqdm(list(enumerate(results))):
                    if result is not None:
                        point, normal, obj, isItLight = result
                        if isItLight:
                            new_colors.append(np.array(scene.light_color))
                        else:
                            new_colors.append(
                                pool.apply_async(compute_color,
                                                 (scene, obj, point, normal)))
                    else:
                        new_colors.append(None)
                print('getting results...')
                for i_ray, new_color in tqdm(list(enumerate(new_colors))): #tqdm(list(enumerate(new_colors))):
                    if new_color is not None:
                        point, _,_,_ = results[i_ray]
                        if type(new_color) is ApplyResult:
                            new_color = new_color.get()
                        old_color, _ = colored_intersections[i_ray]
                        colored_intersections[i_ray] = (
                            old_color+new_color*accumulated_k[i_ray], [(point, new_color,)])

            # now we create new rays
            old_rays = rays
            rays = []
            counter = 0
            for result in results:
                if result is not None:
                    point, normal, obj, isItLight = result
                    if not isItLight:
                        ray_type_randomness = uniform(0, obj['kd']+obj['ks'])
                        if ray_type_randomness <= obj['kd']:  # Case 1: diffuse
                            phi = np.arccos(math.sqrt(uniform(0, 1)))
                            theta = TAU*uniform(0, 1)
                            ray_vector = np.array((np.sin(theta)*np.cos(phi),
                                                   np.sin(theta)*np.sin(phi),
                                                   np.cos(phi)))
                            ray_vector = ray_vector/np.linalg.norm(ray_vector)
                            rays.append((point, ray_vector))
                            accumulated_k[counter] *= obj['kd'] * \
                                np.dot(ray_vector, normal)
                        else:
                            _, old_ray_vector = old_rays[counter]
                            ray_vector = np.dot(normal, old_ray_vector)*2*normal - old_ray_vector
                            ray_vector = ray_vector/np.linalg.norm(ray_vector)
                            eye_vector = scene.eye - point
                            eye_vector = eye_vector/np.linalg.norm(eye_vector)
                            rays.append((point, ray_vector))
                            accumulated_k *= obj['ks'] * \
                                ((np.dot(eye_vector, ray_vector))**obj['n'])
                    else:
                        rays.append(None)
                else:
                    rays.append(None)
                counter += 1
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
