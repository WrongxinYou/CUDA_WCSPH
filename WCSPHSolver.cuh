#pragma once
#ifndef WCSPHSOLVER_CUH
#define WCSPHSOLVER_CUH

#include "WCSPHSystem.h"
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <time.h>

// uniform random [0,1]
inline __device__ float cudaRandomFloat(curandState* state, int pid) {

	curandState localState = state[pid];
	curand_init((unsigned int)clock(), pid, 0, &localState);
	return curand_uniform(&localState);
}

// Pressure update function (Tait¡¯s equation)
inline __HOSTDEV__ float PressureUpdate(float rho, float rho0, float C0, float gamma) {
	// Weakly compressible, tait function
	float b = rho0 * pow(C0, 2) / gamma;
	return b * (pow(rho / rho0, gamma) - 1.0);
}

// SPH kernel function

inline __HOSTDEV__ double SpikyGradientKernel(int dim, double dist, double cutoff, double spiky_grad_factor) {
	double res = 0;
	double x;
	if (0 < dist && dist < cutoff) {
		x = 1.0 - pow(dist / cutoff, 2);
		res = spiky_grad_factor * x / pow(cutoff, 4);
	}
	return res;
}

inline __HOSTDEV__ double Poly6Kernel(int dim, double dist, double cutoff, double poly6_factor) {
	double res = 0;
	double x;
	if (0 < dist && dist < cutoff) {
		x = 1.0 - pow(dist / cutoff, 2);
		res = poly6_factor * pow(x, 3) / pow(cutoff, 3);
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