#!/usr/bin/env python3
from plot import plot_scene_raster, init, show
from scene_reader import Scene
import sys

if __name__ == '__main__':
    scene = Scene(sys.argv[1])
    print(sys.argv[1])
    widget = init()
    plot_scene_raster(widget, scene)
    show()
