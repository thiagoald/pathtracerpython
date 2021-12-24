#!/usr/bin/env python3

from sys import argv
from scene_reader import Scene
import ipdb
from plot import plot_objects


def main():
    scene = Scene(argv[1])
    plot_objects(scene.objects)


if __name__ == '__main__':
    main()
