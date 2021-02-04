#pragma once
#ifndef WCSPHSOLVER_CUH
#define WCSPHSOLVER_CUH

#include "WCSPHSystem.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <time.h>

// uniform random [0,1]
inline __device__ float cudaRandomFloat(curandState* state, int pid) {

	curandState localState = state[pid];
	curand_init((unsigned int)clock(), pid, 0, &localState);
	return curand_uniform(&localState);
}

// Pressure update function (Tait¡¯s equation)
inline __HOSTDEV__ double PressureUpdate(double rho, double rho_0, double C_s, double gamma) {
	// Weakly compressible, tait function
	double B = rho_0 * pow(C_s, 2) / gamma;
	return B * (pow(rho / rho_0, gamma) - 1.0);
}

// SPH kernel function

inline __HOSTDEV__ double SpikyGradientKernel(int dim, double dist, double cutoff, double spiky_grad_factor) {
	double res = 0;
	if (dist <= cutoff) {
		double x = 1.0 - pow(dist / cutoff, 2);
		res = spiky_grad_factor * x / pow(cutoff, 4);
	}
	return res;
}

inline __HOSTDEV__ double Poly6Kernel(int dim, double dist, double cutoff, double poly6_factor) {
	double res = 0;
	if (dist <= cutoff) {
		double x = 1.0 - pow(dist / cutoff, 2);
		res = poly6_factor * pow(x, 3) / pow(cutoff, 3);
	}
	return res;
}

inline __HOSTDEV__ double ViscosityKernelLaplacian(double dist, double cutoff, double vis_lapla_factor) {
	double res = 0;
	if (dist <= cutoff) {
		double x = 1.0 - dist / cutoff;
		res = vis_lapla_factor * x / pow(cutoff, 5);
	}
	return res;
}

inline __HOSTDEV__ double CubicSplineKernel(int dim, double dist, double cutoff, double cubic_factor) {
	// B - cubic spline smoothing kernel
	double res = 0;
	double x = cubic_factor / pow(cutoff, dim);
	double q = dist / cutoff;
	if (q <= 1.0 + M_EPS) {
		res = x * (1.0 - 3.0 / 2.0 * pow(q, 2) * (1.0 - q / 2.0));
	}
	else if (q <= 2.0 + M_EPS) {
		res = x / 4.0 * pow(2.0 - q, 3);
	}
	return res;
}

inline __HOSTDEV__ double CubicSplineKernelDerivative(int dim, double dist, double cutoff, double cubic_factor) {
	// B - cubic spline smoothing kernel
	double res = 0;
	double x = cubic_factor / pow(cutoff, dim);
	double q = dist / cutoff;
	if (q <= 1.0 + M_EPS) {
		res = (x / cutoff) * (-3.0 * q + 2.25 * pow(q, 2));
	}
	else if (q <= 2.0 + M_EPS) {
		res = -0.75 * (x / cutoff) * pow(2.0 - q, 2);
	}
	return res;
}

//
// for FindVelocityLenMinMax use, min max function and warpReduce function
//
typedef float (*pfunc) (float, float);
static __device__ pfunc find_minmax[2] = { fmaxf, fminf };
inline __device__ void FindMinMaxWarpReduce(unsigned int blockSize, volatile float* sdata, unsigned int tid, pfunc func) {
	if (blockSize >= 64) sdata[tid] = func(sdata[tid], sdata[tid + 32]);
	if (blockSize >= 32) sdata[tid] = func(sdata[tid], sdata[tid + 16]);
	if (blockSize >= 16) sdata[tid] = func(sdata[tid], sdata[tid + 8]);
	if (blockSize >= 8) sdata[tid] = func(sdata[tid], sdata[tid + 4]);
	if (blockSize >= 4) sdata[tid] = func(sdata[tid], sdata[tid + 2]);
	if (blockSize >= 2) sdata[tid] = func(sdata[tid], sdata[tid + 1]);
}


//
// Init CUDA Device System
//
void InitDeviceSystem(WCSPHSystem* para, float* dens_init, float3* pos_init, float3* velo_init);

//
// Free CUDA Device System
//
void FreeDeviceSystem(WCSPHSystem* para);

//
// Get next frame information
//
void getNextFrame(WCSPHSystem* sys, cudaGraphicsResource* position_resource, cudaGraphicsResource* color_resource);

////////////////////////////////////////////////////////////////////////////////

//
// Compute which zone each particle belongs to
//
__global__ void ComputeZoneIdx(WCSPHSystem* para,
	int* particle_zidx, int* zone_pidx,
	float3* cur_pos);

//
// Use Radix sort to place particle in zone order
//
__global__ void SortParticles(WCSPHSystem* para,
	int* particle_zidx, int* zone_pidx,
	float* density, float* pressure,
	float3* cur_pos, float3* velocity);

//
// Compute delta value of density, pressure and viscosity for each particle
//
__global__ void ComputeDeltaValue(WCSPHSystem* para,
	int* zone_pidx,
	float* delta_density, float* density, float* pressure,
	float3* cur_pos, float3* delta_pressure, float3* delta_viscosity, float3* velocity);

// 
// Compute delta_velocity and velocity using delta_pressure and delta_viscosity for each particle
// 
__global__ void ComputeVelocity(WCSPHSystem* para,
	float3* cur_pos, float3* delta_pressure, float3* delta_viscosity, float3* delta_velocity, float3* velocity);

//
// Compute new position using velocity for each particle
//
__global__ void ComputePosition(WCSPHSystem* para,
	float3* cur_pos, float3* next_pos, float3* velocity);

//
// If particle exceed the boundary, confine it to the inside, change the velocity and position
//
__global__ void ConfineToBoundary(WCSPHSystem* para, curandState* devStates,
	float3* cur_pos, float3* next_pos, float3* velocity);

//
// Update the new density, pressure, velocity and position for each particle
//
__global__ void UpdateParticles(WCSPHSystem* para,
	float* delta_density, float* density, float* pressure, float* velocity_len,
	float3* cur_pos, float3* next_pos, float3* velocity);

//
// Use for debug, output the variable value on gpu
//
__global__ void DebugOutput(WCSPHSystem* para,
	int* particle_zidx, int* zone_pidx,
	float* delta_density, float* density, float* pressure,
	float3* cur_pos, float3* next_pos, float3* delta_pressure, float3* delta_viscocity, float3* delta_velocity, float3* velocity);

//
// Smartly choose the time step to calculate
//
__global__ void AdaptiveStep(WCSPHSystem* para,
	float* density,
	float3* delta_velocity, float3* velocity);

//
// Find maximum and minimum value of velocity_len for each particle
//
__global__ void FindVelocityLenMinMax(unsigned int blockSize, float* velocity_len, float* g_odata, unsigned int num, bool findmin);

//
// Export particle information to VBO for drawing, blue(0, 0, 1) is slow, white(1, 1, 1) is fast
//
__global__ void ExportParticleInfo(WCSPHSystem* para,
	float* velocity_len, float* velo_min, float* velo_max,
	float3* cur_pos, float3* pos_info, float3* color_info);

#endif // !WCSPHSOLVER_CUH