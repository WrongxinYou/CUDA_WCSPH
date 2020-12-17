#pragma once
#ifndef WCSPHSYSTEM_H
#define WCSPHSYSTEM_H

#include "utils/math_util.h"

class WCSPHSystem
{
public:

	// WCSPH System Parameters
	char* config_filename;				// filename of config to read
	int dim;							// dimension of our WCSPH System, 1D, 2D or 3D
	float3 box_length;					// length of the box that contains all particles
	float3 box_margin;					// the margin of the box when initialize position of particles, only used when initializing
	dim3 zone_dim;						// divice the whole box into zone_dim zones
	uint zone_size;						// total number of zones
	float3 zone_length;					// dimensional size of each zone, should equal to box_length / zone_dim
	float eta;							// used in ConfineToBoundary, coefficient for velocity loss when particle hit the boundary
	float f_air;						// air resistance coefficient, set to 0 if not considering air resistance
	float gravity;						// gravity of our system
	float time_delta;					// time_delta to update the system
	
	// Particles Parameters
	uint3 particle_dim;					// number of particles initialized in x, y, z axis directions
	int particle_num;					// total number of particles, should equal to particle_dim.x * y * z
	float particle_radius;				// radius of particles
	float h;							// cutoff length, usually is radius, used to determine whether two particles will affect each other
	float mass;							// mass of the particles
	float3 velo_init_min;				// minimum initial velocity of each particle in x, y, z directions
	float3 velo_init_max;				// maximum initial velocity of each particle in x, y, z directions

	// Draw Parameters
	int step_each_frame;				// number of computations between each frame
	float velo_draw_min;				// minimum velocity to draw as blue
	float velo_draw_max;				// maximum velocity to draw as white

	// Device Parameters
	//dim3 grid_dim;						// number of CUDA blocks to generate inside one grid in x, y, z directions
	uint grid_size;						// total number of CUDA blocks in one grid, should equal to grid_dim.x * y * z
	//dim3 block_dim;						// number of CUDA threads to generate inside one block in x, y, z directions
	uint block_size;					// total number of threads in one CUDA block

	// Function Parameters
	float alpha;						// viscosity constant in between 0.08 and 0.5, used in ComputeDeltaValue, for delta_viscosity computation
	float C_s;							// speed of sound in the fluid, sqrt((2 * gravity * Height of particle + pow(initial velocity, 2)) / 0.01) 
	float gamma;						// pow in Tait¡¯s equation, used in PressureUpdate and AdaptiveStep
	float rho_0;						// initial density of each particle, used in PressureUpdate and InitializeDensity
	float CFL_a;						// used in AdaptiveStep
	float CFL_v;						// used in AdaptiveStep
	float poly6_factor;					// used in Poly6Kernel
	float spiky_grad_factor;			// used in SpikyGradientKernel
	float vis_lapla_factor;				// kernel of viscosity calculation
	float cubic_factor1D;				// used in CubicSplineKernel and CubicSplineKernelDerivative, coefficient for dim = 1D
	float cubic_factor2D;				// used in CubicSplineKernel and CubicSplineKernelDerivative, coefficient for dim = 2D
	float cubic_factor3D;				// used in CubicSplineKernel and CubicSplineKernelDerivative, coefficient for dim = 3D

public:
	WCSPHSystem();
	WCSPHSystem(char* filename);
	~WCSPHSystem();

	// Print out the parameter settings
	void Print();
	// Initialize particles density
	float* InitializeDensity();
	// Initialize particles position
	float3* InitializePosition();
	// Initialize particles velocity
	float3* InitializeVelocity();

};

#endif // !WCSPHSYSTEM_H
