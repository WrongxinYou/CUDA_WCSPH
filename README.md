# CUDA Accelerated WCSPH Fluid Simulation

## Description

Weakly compressible Smoothed-particle hydrodynamics (WCSPH), as one of the fluid simulation methods, is widely used in games, graphics and medical industry. Though there are several ways of accelerating SPH in CPU, GPU has an ideal structure for the computation of SPH. This project will implement a SPH solver on GPU. After that, it will use OpenGL for rendering the fluid particles, or use an existing rendering engine (for example, Houdini or Blender) to display the process.

## Environment

Windows + Visual Studio

Language and Libraries: C++, CUDA, OpenGL/Houdini/Blender

## General Procedure
<img src="fig/Flow Chart Diagram.png" width="350">  

## Solution

We plan to build a real-time SPH solver using GPU acceleration and visualize the process.
The project is divided into two parts: fluid simulation and fluid particle visualization.

### Fluid Simulation
1. Problems:
    The most time consuming part in SPH simulation is neighbor search. Previously, neighbor search was executed on CPU and GPU was only used for the remaining part. So as the number of particles increases, the running time of neighbor search will increase dramatically.

2. Solution:
    In order to make more use of GPU, here we will implement neighbor searching on GPU. The method starts with applying z-index calculation for each particle and parallel radix-sort in CUDA, which keeps the spatial information while indexing. Thus, for each block we just store its contained particle number and the first particle index. Then we compute density, force, acceleration and velocity, so that we can update positions of particles in the next frame. And so on so forth.

    Here are explainations from [Reference](http://maverick.inria.fr/~Prashant.Goswami/Research/Papers/SCA10_SPH.pdf)

    <img src="fig/CUDA Block.png" width="500">

    For each non-empty block in B, a CUDA block is generated in B' and launched with N threads (N = 4 here).

    <img src="fig/CUDA Block and threads.png" width="500">

### Particle Visualization
We will choose one of the following methods to demonstrate our result.

1. Using OpenGL for rendering particles.
2. Export particle positions in each frame. Load the sequence in other engines (Houdini/Blender) for rendering.

## Expected Result
<img src="fig/WCSPH 2D.gif" width="500">  
