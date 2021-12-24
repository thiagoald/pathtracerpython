# Path Tracing



**Installation of PyAssimp**

In order to load .obj files, install pyassimp library.

This library requires a dynamic library file (.dll for windows or .so for Linux), which should be compiled from this old version of the repository because of a bug:

https://github.com/assimp/assimp/tree/4d66b332534467af9a5580b8ff67906aa9c6455d
Just follow the instructions using cmake

bug: https://github.com/assimp/assimp/issues/3706

You may find the already compiled dll in this repo, under the name "assimp-vc142-mtd.dll". Use at your peril!

To check PyAssimp is working, you may run the "samples.py" or "3d_viewer_py3.py" files according to instructions in the assimp github repo