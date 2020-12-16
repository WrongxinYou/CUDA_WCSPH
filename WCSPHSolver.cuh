#pragma once
#ifndef WCSPHSOLVER_CUH
#define WCSPHSOLVER_CUH

#include "WCSPHSystem.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
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
__device__ pfunc find_minmax[2] = { fmaxf, fminf };
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

#endif // !WCSPHSOLVER_CUH