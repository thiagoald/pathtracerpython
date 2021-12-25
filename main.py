#!/usr/bin/env python3

# Example: ./main.py objs/cornellroom.sdl

from sys import argv
from scene_reader import Scene
import ipdb
from plot import plot_scene
from utils import make_rays, make_screen_pts, intersect

def main():
    scene = Scene(argv[1])
    screen_pts = make_screen_pts(*scene.ortho, scene.width, scene.height)[:100]
    rays = make_rays(scene.eye, screen_pts)
    intersections = []
    for obj in scene.objects:
        for triangle in obj['geometry'].triangles:
            for ray in rays:
                intersection = intersect(ray, triangle)
                if intersection:
                    intersections.append(intersection)
    print(f'Number of objects: {len(scene.objects)}')
    print(
        f'Number of triangles: {sum([len(o["geometry"].triangles) for o in scene.objects])}')
    plot_scene(scene, rays)


if __name__ == '__main__':
    main()
