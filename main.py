#!/usr/bin/env python3

# Example: ./main.py objs/cornellroom.sdl

from sys import argv
from scene_reader import Scene
import ipdb
from plot import plot_scene
from utils import make_rays, make_screen_pts, intersect, make_image
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def intersect_rays(color, rays, triangle):
    intersections = []
    for ray in rays:
        pt_3d = intersect(ray, triangle)
        if pt_3d is not None:
            pt_2d = ray[0] + ray[1]
            intersections.append([pt_3d, pt_2d, color])
    return intersections


def main():
    scene = Scene(argv[1])
    screen_pts = make_screen_pts(
        *scene.ortho, scene.width, scene.height)
    rays = make_rays(scene.eye, screen_pts)
    intersections = []
    n_cpus = cpu_count()
    results = []
    with Pool(n_cpus) as pool:
        n_objs = len(scene.objects)
        for i_obj, obj in enumerate(scene.objects):
            obj_color = [obj[c] for c in ['red', 'green', 'blue']]
            n_tris = len(obj['geometry'].triangles)
            for i_tri, triangle in enumerate(obj['geometry'].triangles):
                print((f'objs: {100*(i_obj+1)/n_objs:.2f}% '
                      f'triangles: {100*(i_tri+1)/n_tris:.2f}%'))
                for i_cpu in range(n_cpus):
                    results.append(
                        pool.apply_async(
                            intersect_rays,
                            (obj_color, rays[i_cpu::n_cpus], triangle)))
        for result in tqdm(results):
            intersections.extend(result.get())
    print(f'Number of objects: {len(scene.objects)}')
    print(
        f'Number of triangles: {sum([len(o["geometry"].triangles) for o in scene.objects])}')
    plot_scene(scene, rays, intersections)
    im = make_image(*scene.ortho, scene.width,
                    scene.height, intersections)
    im.show()


if __name__ == '__main__':
    main()
