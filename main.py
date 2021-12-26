#!/usr/bin/env python3

# Example:
#  $ ./main.py objs/cornellroom.sdl --show-scene --show-img --out out.png

from ipdb.__main__ import set_trace
from scene_reader import Scene
from plot import plot_scene
from utils import NoIntersection, make_rays, make_screen_pts, intersect, make_image, squared_dist
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np
import argparse

def compute_ambient_color(obj):
    color_list = [obj[c] for c in ['red', 'green', 'blue']]
    color_array = np.array(color_list)
    return color_array    

def intersect_objects(ray, objects):
    #returns nearest intersecting object
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
        #color = [obj[c] for c in ['red', 'green', 'blue']]
        #return closest_inter[1:-1] + [color]
        return (closest_inter[1], obj)


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
    how_many_rays=1
    colored_intersections=[]
    for _ in range (scene.width*scene.height):
            initial_color = np.array((0,0,1))
            list_of_3d_points_and_colors = [(np.array((0,0,0)), np.array((0.5,0.5,0.5)),)]
            colored_intersections.append((initial_color, list_of_3d_points_and_colors,))
    
    for _ in range(how_many_rays):
        for ray in rays:
            results+=[intersect_objects(ray, scene.objects)]
        counter=0
        for result in results:
            if result is not None:
                point, obj = result
                old_color, _ = colored_intersections[counter]
                colored_intersections[counter]=(old_color,[(point,compute_ambient_color(obj),)],)
            counter+=1
        
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
                    scene.height, intersections) #change this to something with colored_intersections later
    if args.out is not None:
        im.save(args.out)
    if args.show_img:
        im.show()


if __name__ == '__main__':
    main()
