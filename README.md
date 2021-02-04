# Real Time 3D WCSPH Fluid Simulation

## Description
Weakly compressible Smoothed-particle hydrodynamics (WCSPH), as one of the fluid simulation methods, is widely used in games, graphics and medical industry. Though there are several ways of accelerating SPH in CPU, GPU has an ideal structure for the computation of SPH. This project implements a SPH solver on GPU using CUDA, and use OpenGL for rendering fluid particles.  

## Environment
Windows + Visual Studio  
Language and Libraries: C/C++, CUDA, OpenGL, GLSL, GLM

## Features
- Real time 3D SPH fluid simulation. 
- CUDA OpenGL Interpolation (map CUDA address to host). 
- SPH solver on GPU, including radix sorting, force computation and position update. 
- Adaptive timestep available. 
- Allow user interaction to change viewpoint, scale and rotate objects.   

## Solution
### Fluid Simulation
The most time consuming part in SPH simulation is neighbor search. Previously, neighbor search was executed on CPU. As the number of particles increases, the running time of neighbor search will increase dramatically. In order to decrease the runtime, here we implement neighborhood searching on GPU. 

One method is applying radix sort for particles. We divide the space into different small zones. Thus each zone just need to store the index of first particle inside. And we can calculate the number of particles inside one zone easily. Then we compute density, pressure, viscosity and velocity for position update.  

Each zone is assigned to one CUDA block, and it takes care of all the particles inside. And in one CUDA block, each thread takes care of a single particle. When searching in neighbors, one zone gets its 26 neighborhood information according to zone index.
<img src="data/fig/CUDA_block.png" width="500">  
<img src="data/fig/CUDA_block_and_threads.png" width="500">  

The space allocation on CPU and GPU is as below. We do not allocate any space for particles on CPU as all the computation are executed on GPU. Only two pointers for mapping color and position information from GPU memory. 
<img src="data/fig/space_alloc.png" width="500">  

The computation in each time step for density, pressure, viscosity, velocity and position is described [here](https://cg.informatik.uni-freiburg.de/publications/2007_SCA_SPH.pdf). We choose a cubic kernel function and its derivative as our filter and add adaptive time step to avoid particle explosion.


### Particle Visualization
Cuda supports interpolate with opengl, which helps map device data to host. In order to reach higher FPS, particles are rendered using GLSL shaders. In vertex shader, it sets up particle size based on the distance between camera and the particle. In fragment shader, it renders a circle with edges and shadow. Rendering a dot to a sphere uses less time than directly rendering a sphere and needs less computation.  


## Result
<img src="data/fig/CUDA_WCSPH.gif" width="500">  

## Reference
- [Weakly compressible SPH for free surface flows](https://cg.informatik.uni-freiburg.de/publications/2007_SCA_SPH.pdf)  
- [Interactive SPH Simulation and Rendering on the GPU](http://maverick.inria.fr/~Prashant.Goswami/Research/Papers/SCA10_SPH.pdf)  
- [Broad-Phase Collision Detection with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda)  

