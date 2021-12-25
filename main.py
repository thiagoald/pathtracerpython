#!/usr/bin/env python3

# Example: ./main.py objs/cornellroom.sdl

from scene_reader import Scene
from plot import plot_scene
from utils import make_rays, make_screen_pts, intersect, make_image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse


def intersect_rays(color, rays, triangle):
    intersections = []
    for ray in rays:
        pt_3d = intersect(ray, triangle)
        if pt_3d is not None:
            pt_2d = ray[0] + ray[1]
            intersections.append([pt_3d, pt_2d, color])
    return intersections


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene', help='SDL scene')
    parser.add_argument('--out', help='Output image')
    parser.add_argument('--show-img', default=False, action='store_true')
    parser.add_argument('--show-scene', default=False, action='store_true')
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
    if args.show_scene:
        plot_scene(scene, rays, intersections)
    im = make_image(*scene.ortho, scene.width,
                    scene.height, intersections)
    if args.out is not None:
        im.save(args.out)
    if args.show_img:
        im.show()


if __name__ == '__main__':
    main()
