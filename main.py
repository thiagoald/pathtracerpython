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
from bezier import BezierSurface

TAU = 6.28
ZERO = 1E-5
MAX_RAYS_AT_ONCE = 50


def compute_shadow_rays(scene, points, normals, objs, n_light_samples=1):
    '''
    Return the color of a point from direct illumination by the light source
    '''
    # Vectors starting at the object and pointing to the sampled light point
    shadow_rays = []
    light_color = scene.light_color
    areas = scene.light_obj.areas
    for point_index, point in enumerate(points):
        for ray_index in range(n_light_samples):
            i = pick_random_triangle(areas)
            # Random triangle
            light_tri = scene.light_obj.triangles[i]
            # Random point in this triangle
            light_pt = sample_random_pt(light_tri)
            # intersect
            ray_vector = light_pt - point
            ray_vector = ray_vector/np.linalg.norm(ray_vector)
            total_ray_index = point_index*n_light_samples + ray_index
            shadow_rays.append((point, ray_vector, total_ray_index))

    print("Launching shadow rays...")
    gpu_limit = math.ceil(len(shadow_rays)/MAX_RAYS_AT_ONCE)
    results = []
    for gpu_limiter in tqdm(range(gpu_limit)):
        temp_result = intersect_objects(
            shadow_rays[gpu_limiter*MAX_RAYS_AT_ONCE:\
                        (gpu_limiter+1)*MAX_RAYS_AT_ONCE],
            scene.objects,
            scene.light_obj)
        if temp_result is not None:
            results += temp_result
    colors = np.zeros((len(points), 3))
    for result in results:
        total_ray_index, _, _, _, isItLight = result
        if isItLight:
            point_index = total_ray_index//n_light_samples
            _, light_vector, _ = shadow_rays[total_ray_index]
            obj = objs[point_index]
            dot = np.dot(light_vector, normals[point_index])
            if dot < 0:
                dot = 0
            obj_color = [obj[c] for c in ['red', 'green', 'blue']]
            one_color = [c_light*c_obj*dot*obj['kd'] for c_light,
                         c_obj in zip(light_color, obj_color)]
            colors[point_index] += np.array(one_color)
    colors /= n_light_samples

    return colors


def compute_ambient_color(scene, objs):
    color_arrays = []
    for obj in objs:
        color_list = [obj[c]*obj['ka'] *
                      scene.ambient for c in ['red', 'green', 'blue']]
        color_arrays.append(np.array(color_list))
    return np.array(color_arrays)


def organize_objs_data(objects):
    objs_data_triangles = []
    objs_data_normals = []
    data = []

    for obj in objects:
        obj_triangles = []
        obj_normals = []
        for triangle in obj['geometry'].triangles:
            list_triangle = []
            for vector in triangle:
                list_vector = []
                for x in np.array(vector).tolist():
                    list_vector.append(x)
                list_triangle.append(list_vector)

            obj_triangles.append(list_triangle)
        for normal in obj['geometry'].normals:
            obj_normals.append([normal])
        objs_data_triangles.append(obj_triangles)
        objs_data_normals.append(obj_normals)
    data = []
    for index, (triangle_data, normal_data) in enumerate(zip(objs_data_triangles, objs_data_normals)):
        for triangle, normal in zip(triangle_data, normal_data):
            np_triangle = np.array(triangle)
            concat_shape = list(np_triangle.shape)
            concat_shape[0] = 1
            np_normal = np.array(normal)
            np_index = np.resize(np.array((index)), concat_shape)
            np_data = np.concatenate((np_triangle, np_normal.T), axis=1)
            np_data = np.concatenate((np_data, np_index.T), axis=1)
            data.append(np_data)

    np_all_data = np.array(data)
    return np_all_data


def unpack_ray_data(data):
    results = {}
    for datum in data:
        if datum[7] > 0:  # found flag
            pt_3d = np.array((datum[0], datum[1], datum[2]))
            normal = np.array((datum[3], datum[4], datum[5]))
            obj_index = int(datum[6])
            ray_index = int(datum[8])
            results[ray_index] = (pt_3d, normal, obj_index)
    return results


def refract(refraction_ratio, normal, incoming_ray):
    incoming_ray = -incoming_ray
    if (np.dot(incoming_ray, normal) < 0):
        normal = -normal
    incoming_cosine = np.dot(incoming_ray, normal)
    if incoming_cosine*incoming_cosine < 1-(refraction_ratio*refraction_ratio):
        return np.array((0., 0., 0.)), True
    outgoing_cosine = math.sqrt(
        1-(1/refraction_ratio*refraction_ratio)*(1-incoming_cosine*incoming_cosine))
    ray_vector = -(1/refraction_ratio)*incoming_ray - \
        (outgoing_cosine - 1/refraction_ratio*incoming_cosine)*normal
    ray_vector = ray_vector/np.linalg.norm(ray_vector)
    return ray_vector, False


def organize_ray_data(rays):
    ray_data = []
    for ray in rays:
        point, vector, i_ray = ray
        ray_data.append(np.array(
            (point[0], point[1], point[2], vector[0], vector[1], vector[2], i_ray)))
    return np.array(ray_data)


def intersect_objects(rays, objects, light_obj):
    '''Return ray index, intersection point, triangle normal, object and whether object is the light source'''

    gpu_objects = [o for o in objects if type(
        o['geometry']) is not BezierSurface]
    bezier_objects = [o for o in objects if type(
        o['geometry']) is BezierSurface]
    gpu_objects = gpu_objects + [{'geometry': light_obj}]
    obj_data = organize_objs_data(gpu_objects)
    d_obj_data = cuda.to_device(obj_data)

    ray_data = organize_ray_data(rays)
    d_ray_data = cuda.to_device(ray_data)
    # 9 for the 3 positional point values + 3 normal vector values 1 object index + 1 found flag + 1 ray index
    d_out_data = cuda.device_array((len(rays), 9), dtype='float64')
    threadsperblock = 32
    blockspergrid = (d_ray_data.size + threadsperblock-1)
    intersect[blockspergrid, threadsperblock](
        d_obj_data, d_ray_data, d_out_data)
    out_data = d_out_data.copy_to_host()

    gpu_closest_intersections = unpack_ray_data(out_data)

    bezier_closest_intersections = {}
    for ray in rays:
        bezier_intersections = []
        ray_point, _, ray_index = ray
        for one_bezier in bezier_objects:
            surface = one_bezier['geometry']
            try:
                bezier_point, bezier_normal = surface.intersect(ray)
                bezier_intersections.append(
                    (bezier_point, bezier_normal, one_bezier, np.linalg.norm(bezier_point-ray_point)))
            except NoIntersection:
                pass
        if len(bezier_intersections) > 0:
            p, n, obj, _ = min(bezier_intersections,
                               key=lambda inter: inter[-1])
            bezier_closest_intersections[ray_index] = (p, n, obj)

    results = []

    for ray in rays:
        ray_point, _, ray_index = ray
        try:
            b_point, b_normal, b_obj = bezier_closest_intersections[ray_index]
        except KeyError:
            try:
                g_point, g_normal, g_obj_index = gpu_closest_intersections[ray_index]
                g_obj = gpu_objects[g_obj_index]
            except KeyError:
                continue
            if g_obj['geometry'] == light_obj:
                isItLight = True
            else:
                isItLight = False
            results.append([ray_index, g_point, g_normal, g_obj, isItLight])
            continue
        try:
            g_point, g_normal, g_obj_index = gpu_closest_intersections[ray_index]
            g_obj = gpu_objects[g_obj_index]
        except KeyError:
            results.append([ray_index, b_point, b_normal, b_obj, False])
            continue
        if np.linalg.norm(ray_point-g_point) > np.linalg.norm(ray_point-b_point):
            results.append([ray_index, b_point, b_normal, b_obj, False])
        else:
            if g_obj['geometry'] == light_obj:
                isItLight = True
            else:
                isItLight = False
            results.append([ray_index, g_point, g_normal, g_obj, isItLight])
    return results


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


def compute_color(scene, objs, points, normals, AreTheyLight, i_rays):
    new_colors = []
    new_objs = []
    new_points = []
    new_normals = []
    new_i_rays = []

    for i, isIt in enumerate(AreTheyLight):
        if isIt:
            new_colors.append(
                (np.array(scene.light_color), i_rays[i], points[i]))
        else:
            new_objs.append(objs[i])
            new_points.append(points[i])
            new_normals.append(normals[i])
            new_i_rays.append(i_rays[i])

    ambs = compute_ambient_color(scene, new_objs)
    shas = compute_shadow_rays(scene, new_points, new_normals, new_objs)
    for i, (amb, sha) in enumerate(zip(ambs, shas)):
        new_colors.append((np.array(amb+sha), new_i_rays[i], new_points[i]))
    return new_colors


def rotate(axis, angle, v):
    """
    Return the counterclockwise rotation about
    the given axis by 'angle' radians.
    """
    if (np.linalg.norm(axis) < ZERO):
        return v
    axis = axis / np.linalg.norm(axis)
    a = math.cos(angle / 2.0)
    b, c, d = -axis * math.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rotation_matrix = np.array(
        [[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
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
        f'Number of triangles: {sum([len(o["geometry"].triangles) for o in scene.objects if type(o["geometry"]) is not BezierSurface])}')
    results = []
    how_many_rays = args.n_rays
    how_many_bounces = args.n_bounces
    # initialization
    colored_intersections = []
    for _ in range(scene.width*scene.height):
        initial_color = np.array((0., 0., 0.))
        initial_list_of_3d_points_and_colors = [
            (np.array((0, 0, 0)), np.array((0, 0, 0)),)]
        colored_intersections.append(
            (initial_color, initial_list_of_3d_points_and_colors,))
    # iterative (non-recursive) path tracing
    pixel_color_list = [0]*(scene.width*scene.height)
    for rays_counter in tqdm(range(how_many_rays)):
        print('rays_counter is ' + str(rays_counter))
        colored_intersections = [(initial_color, initial_list_of_3d_points_and_colors)
                                 for intersec in colored_intersections]
        accumulated_k = np.ones(scene.width*scene.height)
        rays = make_rays(scene.eye, screen_pts)
        for bounces_counter in range(how_many_bounces):
            print('bounces_counter is ' + str(bounces_counter))
            # compute intersections

            print('intersecting...')
            results = []
            gpu_limit = math.ceil(len(rays)/MAX_RAYS_AT_ONCE)
            for gpu_limiter in tqdm(range(gpu_limit)):
                limited_rays = rays[gpu_limiter *
                                    MAX_RAYS_AT_ONCE: (gpu_limiter+1)*MAX_RAYS_AT_ONCE]
                temp_result = intersect_objects(
                    limited_rays, scene.objects, scene.light_obj)
                if temp_result is not None:
                    results += temp_result

            # now we create new rays
            print('computing new rays...')
            old_rays = {}
            for ray in rays:
                _, _, index = ray
                old_rays[index] = ray
            rays = []
            isItDiffuse = [True] * (scene.width*scene.height)
            isItSpecular = [False] * (scene.width*scene.height)
            for result in tqdm(results):
                if result is not None:

                    i_ray, point, normal, obj, isItLight = result

                    if not isItLight:

                        ray_type_randomness = uniform(
                            0, obj['kd']+obj['ks']+obj['kt'])  # toss the coin

                        if ray_type_randomness <= obj['kd']:  # Case 1: diffuse
                            isItDiffuse[i_ray] = True
                            isItSpecular[i_ray] = False
                            phi = np.arccos(math.sqrt(uniform(0, 1)))
                            phi = phi/2
                            theta = TAU*uniform(0, 1)
                            ray_vector = np.array((np.sin(phi)*np.sin(theta),
                                                   np.cos(phi),
                                                   np.sin(phi)*np.cos(theta)))
                            ray_vector = ray_vector/np.linalg.norm(ray_vector)

                            local_axis = np.cross(np.array((0, 1, 0)), normal)
                            ray_vector = rotate(local_axis, np.arccos(
                                np.dot(np.array((0, 1, 0)), normal)), ray_vector)
                            rays.append((point, ray_vector, i_ray))
                            accumulated_k[i_ray] *= obj['kd'] * \
                                abs(np.dot(ray_vector, normal))

                        # Case 2: specular
                        elif ray_type_randomness <= obj['kd']+obj['ks']:
                            isItDiffuse[i_ray] = False
                            isItSpecular[i_ray] = True

                            # Random point in the light source
                            areas = scene.light_obj.areas
                            i = pick_random_triangle(areas)
                            light_tri = scene.light_obj.triangles[i]
                            light_pt = sample_random_pt(light_tri)
                            light_pt = (
                                np.array(light_tri[0]) + np.array(light_tri[1]) + np.array(light_tri[2]))/3
                            light_vector = light_pt - point
                            light_vector = light_vector / \
                                np.linalg.norm(light_vector)

                            reflected_ray_vector = 2 * \
                                np.dot(light_vector, normal) * \
                                normal - light_vector
                            reflected_ray_vector = reflected_ray_vector / \
                                np.linalg.norm(reflected_ray_vector)

                            eye_vector = scene.eye - point
                            eye_vector = eye_vector/np.linalg.norm(eye_vector)

                            specular_dot = np.dot(
                                reflected_ray_vector, eye_vector)
                            if specular_dot < 0:
                                specular_dot = 0

                            accumulated_k[i_ray] *= obj['ks'] * \
                                (specular_dot)**obj['n']

                        else:  # Case 3: transmitted
                            isItDiffuse[i_ray] = False
                            isItSpecular[i_ray] = False
                            _, old_ray_vector, _ = old_rays[i_ray]
                            ray_vector, _ = refract(
                                1.31, normal, old_ray_vector)
                            ray = (point, ray_vector, i_ray,)
                            total_reflection = True
                            total_reflection_counter = 3
                            while total_reflection:
                                total_reflection_counter -= 1
                                if total_reflection_counter == 0:
                                    break
                                refraction_result = intersect_objects(
                                    [ray], [obj], scene.light_obj)
                                if refraction_result is not None:
                                    _, new_point, new_normal, _, _ = refraction_result[0]
                                    new_ray_vector, total_reflection = refract(
                                        1./1.31, new_normal, ray_vector)
                                    if not total_reflection:
                                        rays.append(
                                            (new_point, new_ray_vector, i_ray))
                                        accumulated_k[i_ray] *= obj['kt']
                                        break
                                    else:  # reflect the ray inside the object
                                        new_ray_vector = -new_ray_vector
                                        if np.dot(new_ray_vector, new_normal) < 0:
                                            adjusted_new_normal = -new_normal
                                        else:
                                            adjusted_new_normal = new_normal
                                        total_reflection_ray_vector = 2 * \
                                            np.dot(
                                                new_ray_vector, adjusted_new_normal)*adjusted_new_normal - new_ray_vector
                                        ray = (
                                            new_point, total_reflection_ray_vector, i_ray)
                            if total_reflection_counter == 0:  # reflected internally too many times
                                # use the object's normal color
                                isItDiffuse[i_ray] = True
                                accumulated_k[i_ray] *= 4

            # compute colors
            new_colors = []
            i_rays = []
            points = []
            normals = []
            objs = []
            AreTheyLight = []
            obj_dict = {}
            light_dict = {}
            for result in results:
                if result is not None:
                    x, y, z, w, l = result
                    i_rays.append(x)
                    points.append(y)
                    normals.append(z)
                    objs.append(w)
                    AreTheyLight.append(l)
                    obj_dict[x] = w
                    light_dict[x] = l
            new_color_data = compute_color(
                scene, objs, points, normals, AreTheyLight, i_rays)
            for new_color, i_ray, point in new_color_data:
                old_color, _ = colored_intersections[i_ray]
                if isItDiffuse[i_ray]:
                    colored_intersections[i_ray] = (
                        old_color+new_color*accumulated_k[i_ray], [(point, new_color,)])
                elif isItSpecular[i_ray]:
                    new_color = np.array(scene.light_color)
                    colored_intersections[i_ray] = (
                        old_color+new_color*accumulated_k[i_ray], [(point, new_color,)])
                else:  # then it's transmitted
                    colored_intersections[i_ray] = (
                        old_color, [(point, new_color,)])

        for i, intersec in enumerate(colored_intersections):
            pixel_color, list_of_3d_points_and_colors = intersec
            pixel_color_list[i] += pixel_color

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
