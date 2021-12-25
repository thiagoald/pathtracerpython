#!/usr/bin/env python3

# Example:
#  $ ./main.py objs/cornellroom.sdl --show-scene --show-img --out out.png

from ipdb.__main__ import set_trace
from scene_reader import Scene
from plot import plot_scene
from utils import NoIntersection, make_rays, make_screen_pts, intersect, make_image, squared_dist
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import argparse


def intersect_objects(ray, objects):
    intersections = []
    for i, obj in enumerate(objects):
        for triangle in obj['geometry'].triangles:
            try:
                pt_3d = intersect(ray, triangle)
                if pt_3d is not None:
                    pt_screen = ray[0] + ray[1]
                    intersections.append(
                        [squared_dist(pt_3d, ray[0]), pt_3d, pt_screen, i])
            except NoIntersection:
                pass

    if intersections == []:
        return None
    else:
        # Only return the closest
        closest_inter = min(intersections, key=lambda inter: inter[0])
        i_obj = closest_inter[-1]
        obj = objects[i_obj]
        color = [obj[c] for c in ['red', 'green', 'blue']]
        return closest_inter[1:-1] + [color]


def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene', help='SDL scene')
    parser.add_argument('--out', help='Output image')
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
    with Pool(cpu_count()) as pool:
        for ray in rays:
            results.append(pool.apply_async(intersect_objects,
                                            (ray, scene.objects)))
        for result in tqdm(results):
            result = result.get()
            if result is not None:
                intersections.append(result)
    if args.show_scene:
        plot_scene(scene, rays, intersections,
                   show_normals=args.show_normals,
                   show_screen=args.show_screen,
                   show_inter=args.show_inter)
    im = make_image(*scene.ortho, scene.width,
                    scene.height, intersections)
    if args.out is not None:
        im.save(args.out)
    if args.show_img:
        im.show()


if __name__ == '__main__':
    main()
