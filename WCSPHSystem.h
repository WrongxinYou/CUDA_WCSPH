#pragma once
#ifndef WCSPHSYSTEM_H
#define WCSPHSYSTEM_H

#include "math_util.h"

class WCSPHSystem
{
public:

	// WCSPH Parameters
	int dim;					// dimension of our WCSPH System, 1D, 2D or 3D
	int3 particle_dim;			// number of particles initialized in x, y, z axis directions
	int particle_num;			// total number of particles, should equal to particle_dim.x * y * z
	float particle_radius;		// radius of particles
	float3 velo_init_min;		// minimum initial velocity of each particle in x, y, z directions
	float3 velo_init_max;		// maximum initial velocity of each particle in x, y, z directions
	char* config_filename;		// filename of config to read

	// Draw Parameters
	int step_each_frame;		// number of computations between each frame
	float3 box_size;			// size of the box that contains all particles
	float3 box_margin;			// the margin of the box when initialize position of particles, only used when initializing

	// Device Parameters
	int3 block_dim;				// number of CUDA blocks to generate in x, y, z directions
	int block_num;				// total number of CUDA blocks, should equal to block_dim.x * y * z
	float3 block_size;			// dimensional size of each CUDA block, should equal to box_size / block_dim
	int block_thread_num;		// thread number in each CUDA block

	// Function Parameters
	float rho0;					// reference density, initial density of each particle
	float gamma;				// used in PressureUpdate and AdaptiveStep
	float h;					// cutoff length, used to determine whether two particles will affect each other
	float gravity;				// gravity of our system
	float alpha;				// used in ComputeDeltaValue, for delta_viscosity computation, usually between 0.08 and 0.5
	float C0;					// used in PressureUpdate and ComputeDeltaValue, for delta_viscosity computation as well as the time_delta computation
	float CFL_v;				// used in AdaptiveStep
	float CFL_a;				// used in AdaptiveStep
	float poly6_factor;			// used in Poly6Kernel
	float spiky_grad_factor;	// used in SpikyGradientKernel
	float cubic_factor1D;		// used in CubicSplineKernel and CubicSplineKernelDerivative, coefficient for dim = 1D
	float cubic_factor2D;		// used in CubicSplineKernel and CubicSplineKernelDerivative, coefficient for dim = 2D
	float cubic_factor3D;		// used in CubicSplineKernel and CubicSplineKernelDerivative, coefficient for dim = 3D
	float mass;					// mass of the particles
	float time_delta;			// time_delta to update the system
	float eta;					// used in ConfineToBoundary, coefficient for velocity loss when particle hit the boundary
	float f_air;				// air resistance coefficient, set to 0 if not considering air resistance

public:
	WCSPHSystem();
	WCSPHSystem(char* filename);
	~WCSPHSystem();

	// Initialize particles density
	float* InitializeDensity();
	// Initialize particles position
	float3* InitializePosition();
	// Initialize particles velocity
	float3* InitializeVelocity();

};

#endif // !WCSPHSYSTEM_H
