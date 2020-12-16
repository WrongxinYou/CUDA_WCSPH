#include "WCSPHSolver.cuh"
#include "utils/handler.h"

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

//#define DEBUG
//#define CONFINE_RANDOM
#define CUDA_MEMCPY_ASYNC
#define CUDA_MEMSET_ASYNC

const int kCudaSortArrayCount = 4;

#if defined(CUDA_MEMCPY_ASYNC) || defined(CUDA_MEMCPY_ASYNC)
const int kCudaMemcpyTime = 7;
#endif // defined(CUDA_MEMCPY_ASYNC) || defined(CUDA_MEMCPY_ASYNC)


////////////////////////////////////////////////////////////////////////////////
// Device array declare
////////////////////////////////////////////////////////////////////////////////

WCSPHSystem* sph_device = NULL;

int* particle_bid = NULL; // each particle belongs to which block
int* block_pidx = NULL; // first particle index in grid
int* block_pnum = NULL; // particle number in grid

curandState* devStates = NULL;

float3* color = NULL; // color of particles
float3* cur_pos = NULL;
float3* next_pos = NULL;

float* density = NULL;
float* delta_density = NULL;

float* pressure = NULL;
float3* delta_pressure = NULL;

float3* delta_viscosity = NULL;

float* velo_min = NULL;
float* velo_max = NULL;
float* velocity_len = NULL;
float3* velocity = NULL;
float3* delta_velocity = NULL;


////////////////////////////////////////////////////////////////////////////////
//
// Init CUDA Device System
//
////////////////////////////////////////////////////////////////////////////////
void InitDeviceSystem(WCSPHSystem* para, float* dens_init, float3* pos_init, float3* velo_init) {

#ifdef DEBUG
	std::cout << "Do InitDeviceSystem" << std::endl;
#endif // DEBUG

	int num = para->particle_num;
#if defined (CUDA_MEMCPY_ASYNC) || defined (CUDA_MEMSET_ASYNC)
	cudaStream_t stream[kCudaMemcpyTime];
	int streamnum = 0;
	for (int i = 0; i < kCudaMemcpyTime; i++) {
		checkCudaErrors(cudaStreamCreate(&stream[i]));
	}
#endif // CUDA_MEMCPY_ASYNC || CUDA_MEMSET_ASYNC

	checkCudaErrors(cudaMalloc((void**)&sph_device, sizeof(WCSPHSystem)));
#ifdef CUDA_MEMCPY_ASYNC
	checkCudaErrors(cudaMemcpyAsync(sph_device, para, sizeof(WCSPHSystem), cudaMemcpyHostToDevice, stream[streamnum++]));
#else
	checkCudaErrors(cudaMemcpy(sph_device, para, sizeof(WCSPHSystem), cudaMemcpyHostToDevice));
#endif // CUDA_MEMCPY_ASYNC
	

	checkCudaErrors(cudaMalloc((void**)&particle_bid, kCudaSortArrayCount * num * sizeof(int)));

	checkCudaErrors(cudaMalloc((void**)&block_pidx, para->grid_size * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&block_pnum, para->grid_size * sizeof(int)));


	checkCudaErrors(cudaMalloc((void**)&devStates, num * sizeof(curandState)));

	checkCudaErrors(cudaMalloc((void**)&color, num * sizeof(float3)));


	checkCudaErrors(cudaMalloc((void**)&cur_pos, num * sizeof(float3)));
#ifdef CUDA_MEMCPY_ASYNC
	checkCudaErrors(cudaMemcpyAsync(cur_pos, pos_init, num * sizeof(float3), cudaMemcpyHostToDevice, stream[streamnum++]));
#else
	checkCudaErrors(cudaMemcpy(cur_pos, pos_init, num * sizeof(float3), cudaMemcpyHostToDevice));
#endif // CUDA_MEMCPY_ASYNC


	checkCudaErrors(cudaMalloc((void**)&next_pos, num * sizeof(float3)));
#ifdef CUDA_MEMSET_ASYNC
	checkCudaErrors(cudaMemsetAsync(next_pos, 0, num * sizeof(float3), stream[3]));
#else
	checkCudaErrors(cudaMemset(next_pos, 0, num * sizeof(float3)));
#endif // CUDA_MEMSET_ASYNC

	checkCudaErrors(cudaMalloc((void**)&density, num * sizeof(float)));
#ifdef CUDA_MEMCPY_ASYNC
	checkCudaErrors(cudaMemcpyAsync(density, dens_init, num * sizeof(float), cudaMemcpyHostToDevice, stream[streamnum++]));
#else
	checkCudaErrors(cudaMemcpy(density, dens_init, num * sizeof(float), cudaMemcpyHostToDevice));
#endif // CUDA_MEMCPY_ASYNC

	checkCudaErrors(cudaMalloc((void**)&delta_density, num * sizeof(float)));

	checkCudaErrors(cudaMalloc((void**)&pressure, num * sizeof(float)));
#ifdef CUDA_MEMSET_ASYNC
	checkCudaErrors(cudaMemsetAsync(pressure, 0, num * sizeof(float), stream[5]));
#else
	checkCudaErrors(cudaMemset(pressure, 0, num * sizeof(float)));
#endif // CUDA_MEMSET_ASYNC

	checkCudaErrors(cudaMalloc((void**)&delta_pressure, num * sizeof(float3)));

	checkCudaErrors(cudaMalloc((void**)&delta_viscosity, num * sizeof(float3)));
	

	checkCudaErrors(cudaMalloc((void**)&velo_min, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&velo_max, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&velocity_len, num * sizeof(float)));

	checkCudaErrors(cudaMalloc((void**)&velocity, num * sizeof(float3)));
#ifdef CUDA_MEMCPY_ASYNC
	checkCudaErrors(cudaMemcpyAsync(velocity, velo_init, num * sizeof(float3), cudaMemcpyHostToDevice, stream[streamnum++]));
#else
	checkCudaErrors(cudaMemcpy(velocity, velo_init, num * sizeof(float3), cudaMemcpyHostToDevice));
#endif // CUDA_MEMCPY_ASYNC

	checkCudaErrors(cudaMalloc((void**)&delta_velocity, num * sizeof(float3)));

	

#if defined (CUDA_MEMCPY_ASYNC) || defined (CUDA_MEMSET_ASYNC)
	for (int i = 0; i < kCudaMemcpyTime; i++) {
		checkCudaErrors(cudaStreamSynchronize(stream[i]));
		checkCudaErrors(cudaStreamDestroy(stream[i]));
	}
#endif // CUDA_MEMCPY_ASYNC || CUDA_MEMSET_ASYNC

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "Finish InitDeviceSystem" << std::endl;
#endif // DEBUG
}

////////////////////////////////////////////////////////////////////////////////
//
// Free CUDA Device System
//
////////////////////////////////////////////////////////////////////////////////
void FreeDeviceSystem(WCSPHSystem* para) {

#ifdef DEBUG
	std::cout << "Do FreeDeviceSystem" << std::endl;
#endif // DEBUG
	delete para;

	checkCudaErrors(cudaFree(sph_device));

	checkCudaErrors(cudaFree(particle_bid));
	checkCudaErrors(cudaFree(block_pidx));
	checkCudaErrors(cudaFree(block_pnum));

	checkCudaErrors(cudaFree(devStates));

	checkCudaErrors(cudaFree(color));

	checkCudaErrors(cudaFree(cur_pos));
	checkCudaErrors(cudaFree(next_pos));

	checkCudaErrors(cudaFree(density));
	checkCudaErrors(cudaFree(delta_density));

	checkCudaErrors(cudaFree(pressure));
	checkCudaErrors(cudaFree(delta_pressure));

	checkCudaErrors(cudaFree(delta_viscosity));

	checkCudaErrors(cudaFree(velo_min));
	checkCudaErrors(cudaFree(velo_max));
	checkCudaErrors(cudaFree(velocity_len));
	checkCudaErrors(cudaFree(velocity));
	checkCudaErrors(cudaFree(delta_velocity));

#ifdef DEBUG
	std::cout << "Finish InitDeviceSystem" << std::endl;
#endif // DEBUG
}


////////////////////////////////////////////////////////////////////////////////
// CUDA function are implemented here
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//
// Compute which block each particle belongs to
//
////////////////////////////////////////////////////////////////////////////////
__global__ void ComputeBid(			WCSPHSystem* para,
									int* particle_bid,
									float3* cur_pos) {

#ifdef DEBUG
	if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
		printf("Block #(%d,%d,%d) Do ComputeBid\n", blockIdx.x, blockIdx.y, blockIdx.z);
#endif // DEBUG

	// compute block_id for each particle
	int i = GetBlockIdx1D(blockIdx, gridDim) * GetDimTotalSize(blockDim) + threadIdx.x;
	while (i < para->particle_num) {
		// compute particle position inside which bidx block
		int3 bidx = make_int3(cur_pos[i] / para->block_length);
		particle_bid[i] = GetBlockIdx1D(bidx, para->grid_dim);
		i += GetDimTotalSize(gridDim) * GetDimTotalSize(blockDim); // gridSize * blockSize
	}



#ifdef DEBUG
	if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
		printf("Block #(%d,%d,%d) Finish ComputeBid\n", blockIdx.x, blockIdx.y, blockIdx.z);
#endif // DEBUG
}


////////////////////////////////////////////////////////////////////////////////
//
// Use Radix sort to place particle in block order
//
////////////////////////////////////////////////////////////////////////////////
__global__ void SortParticles(		WCSPHSystem* para,
									int* particle_bid,
									float* density, float* pressure,
									float3* cur_pos, float3* velocity) {

#ifdef DEBUG
	if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
		printf("Block #(%d,%d,%d) Do SortParticles\n", blockIdx.x, blockIdx.y, blockIdx.z);
#endif // DEBUG

	int num = para->particle_num;
	if (blockIdx.x == 0) {
		thrust::stable_sort_by_key(thrust::device, particle_bid + num * 0, particle_bid + num * 1, cur_pos);
	}
	else if (blockIdx.x == 1) {
		thrust::stable_sort_by_key(thrust::device, particle_bid + num * 1, particle_bid + num * 2, density);
	}
	else if (blockIdx.x == 2) {
		thrust::stable_sort_by_key(thrust::device, particle_bid + num * 2, particle_bid + num * 3, pressure);
	}
	else if (blockIdx.x == 3) {
		thrust::stable_sort_by_key(thrust::device, particle_bid + num * 3, particle_bid + num * 4, velocity);
	}
	//else if (blockIdx.x == 4) {
	//	thrust::stable_sort_by_key(thrust::device, particle_bid + num * 4, particle_bid + num * 5, next_pos);
	//}

#ifdef DEBUG
	if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
		printf("Block #(%d,%d,%d) Finish SortParticles\n", blockIdx.x, blockIdx.y, blockIdx.z);
#endif // DEBUG
}


////////////////////////////////////////////////////////////////////////////////
//
// Compute the index of first particle and the total particle number in each block 
//
////////////////////////////////////////////////////////////////////////////////
__global__ void ComputeBlockIdxPnum(WCSPHSystem* para,
									int* particle_bid, int* block_pidx, int* block_pnum) {

#ifdef DEBUG
	if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
		printf("Block #(%d,%d,%d) Do ComputeBlockIdxPnum\n", blockIdx.x, blockIdx.y, blockIdx.z);
#endif // DEBUG

	for (int i = 0; i < para->grid_size; i++) {
		block_pidx[i] = -1;
		block_pnum[i] = 0;
	}

	for (int i = 0; i < para->particle_num; i++) {
		if (i == 0 || particle_bid[i] != particle_bid[i - 1]) {
			block_pidx[particle_bid[i]] = i;
		}
		block_pnum[particle_bid[i]]++;
		//if (block_pnum[particle_bid[i]] > para->block_size)
		//	printf("Block %d ERROR, exceed threads number\n", particle_bid[i]);
	}

#ifdef DEBUG
	if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
		printf("Block #(%d,%d,%d) Finish ComputeBlockIdxPnum\n", blockIdx.x, blockIdx.y, blockIdx.z);
#endif // DEBUG
}


////////////////////////////////////////////////////////////////////////////////
//
// Compute delta value of density, pressure and viscosity for each particle
//
////////////////////////////////////////////////////////////////////////////////
__global__ void ComputeDeltaValue(	WCSPHSystem* para,
									int* block_pidx, int* block_pnum,
									float* delta_density, float* density,  float* pressure,
									float3* cur_pos, float3* delta_pressure, float3* delta_viscosity, float3* velocity) {

#ifdef DEBUG
	if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
		printf("Block #(%d,%d,%d) Do ComputeDeltaValue\n", blockIdx.x, blockIdx.y, blockIdx.z);
#endif // DEBUG

	int3 blockIdx_i = make_int3(blockIdx.x, blockIdx.y, blockIdx.z);
	int3 blockDim_i = make_int3(para->grid_dim);
	int threadIdx_i = threadIdx.x;
	int bid = GetBlockIdx1D(blockIdx_i, blockDim_i);

	while (threadIdx_i < block_pnum[bid]) {
		// for each particle[i]
		int i = block_pidx[bid] + threadIdx_i;
		// Initialize
		delta_density[i] = 0.0;
		delta_pressure[i] = make_float3(0, 0, 0);
		delta_viscosity[i] = make_float3(0, 0, 0);

		// for each block 
		for (int ii = 0; ii < 27; ii++) {
			int3 blockIdx_nei = blockIdx_i + make_int3(ii / 9 - 1, (ii % 9) / 3 - 1, ii % 3 - 1);
			if (BlockIdxIsValid(blockIdx_nei, blockDim_i)) {
				int bid_nei = GetBlockIdx1D(blockIdx_nei, blockDim_i);
				// find neighbour particle[j]
#pragma unroll
				for (int j = block_pidx[bid_nei]; j < block_pidx[bid_nei] + block_pnum[bid_nei]; j++) {
					if (i == j) continue;
					float3 vec_ij = cur_pos[i] - cur_pos[j];
					float len_ij = Norm2(vec_ij);
					len_ij = fmaxf(len_ij, M_EPS);

					//float pol_ker = Poly6Kernel(para->dim, len_ij, para->h, para->poly6_factor);
					//float spi_ker = SpikyGradientKernel(para->dim, len_ij, para->h, para->spiky_grad_factor);
					float cub_ker = CubicSplineKernel(para->dim, len_ij, para->h, para->cubic_factor3D);
					float cub_ker_deri = CubicSplineKernelDerivative(para->dim, len_ij, para->h, para->cubic_factor3D);

					// Density (Continuity equation, summation approach)
					delta_density[i] += para->mass * cub_ker;

					//// Density (Continuity equation, differential update)
					//delta_density[i] += para->mass * cub_ker_deri * dot((velocity[i] - velocity[j]), (vec_ij / len_ij));

					// Pressure (Momentum equation)
					delta_pressure[i] -= para->mass * cub_ker_deri * (vec_ij / len_ij) *
						(pressure[i] / fmaxf(M_EPS, pow(density[i], 2)) + pressure[j] / fmaxf(M_EPS, pow(density[j], 2)));

					// Viscosity
					float v_ij = dot(velocity[i] - velocity[j], vec_ij);
					if (v_ij < 0) {
						float viscous = -2.0 * para->alpha * para->h * para->C_s / fmaxf(M_EPS, density[i] + density[j]);
						delta_viscosity[i] -= para->mass * cub_ker_deri * (vec_ij / len_ij) * 
							viscous * v_ij / fmaxf(M_EPS, pow(len_ij, 2) + 0.01 * pow(para->h, 2));
					}
				}
			}
		}
		threadIdx_i += para->block_size;
	}

#ifdef DEBUG
	if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
		printf("Block #(%d,%d,%d) Finish ComputeDeltaValue\n", blockIdx.x, blockIdx.y, blockIdx.z);
#endif // DEBUG
}


////////////////////////////////////////////////////////////////////////////////
// 
// Compute delta_velocity and velocity using delta_pressure and delta_viscosity for each particle
// 
////////////////////////////////////////////////////////////////////////////////
__global__ void ComputeVelocity(	WCSPHSystem* para,
									int* block_pidx, int* block_pnum,
									float* density,
									float3* cur_pos, float3* delta_pressure, float3* delta_viscosity, float3* delta_velocity, float3* velocity) {

#ifdef DEBUG
	if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
		printf("Block #(%d,%d,%d) Do ComputeVelocity\n", blockIdx.x, blockIdx.y, blockIdx.z);
#endif // DEBUG

	int3 blockIdx_i = make_int3(blockIdx.x, blockIdx.y, blockIdx.z);
	int3 blockDim_i = make_int3(para->grid_dim);
	int threadIdx_i = threadIdx.x;
	int bid = GetBlockIdx1D(blockIdx_i, blockDim_i);
	while (threadIdx_i < block_pnum[bid]) {
		int i = block_pidx[bid] + threadIdx_i; // for each particle[i]
		float3 G = make_float3(0, para->gravity, 0);
		// velocity (Momentum equation)
		delta_velocity[i] = delta_pressure[i] + delta_viscosity[i] + G;
		velocity[i] += para->time_delta * delta_velocity[i];
		threadIdx_i += para->block_size;
	}

#ifdef DEBUG
	if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
		printf("Block #(%d,%d,%d) Finish ComputeVelocity\n", blockIdx.x, blockIdx.y, blockIdx.z);
#endif // DEBUG
}


////////////////////////////////////////////////////////////////////////////////
//
// Compute new position using velocity for each particle
//
////////////////////////////////////////////////////////////////////////////////
__global__ void ComputePosition(	WCSPHSystem* para,
									int* block_pidx, int* block_pnum,
									float3* cur_pos, float3* next_pos, float3* velocity) {

#ifdef DEBUG
	if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
		printf("Block #(%d,%d,%d) Do ComputePosition\n", blockIdx.x, blockIdx.y, blockIdx.z);
#endif // DEBUG

	int3 blockIdx_i = make_int3(blockIdx.x, blockIdx.y, blockIdx.z);
	int3 blockDim_i = make_int3(para->grid_dim);
	int threadIdx_i = threadIdx.x;
	int bid = GetBlockIdx1D(blockIdx_i, blockDim_i); 
	while (threadIdx_i < block_pnum[bid]) {
		int i = block_pidx[bid] + threadIdx_i; // for each particle[i]
		next_pos[i] = cur_pos[i] + para->time_delta * velocity[i];
		threadIdx_i += para->block_size;
	}

#ifdef DEBUG
	if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
		printf("Block #(%d,%d,%d) Finish ComputePosition\n", blockIdx.x, blockIdx.y, blockIdx.z);
#endif // DEBUG
}


////////////////////////////////////////////////////////////////////////////////
//
// If particle exceed the boundary, confine it to the inside, change the velocity and position
//
////////////////////////////////////////////////////////////////////////////////
__global__ void ConfineToBoundary(	WCSPHSystem* para, curandState* devStates,
									int* block_pidx, int* block_pnum, 
									float3* cur_pos, float3* next_pos, float3* velocity) {

	int3 blockIdx_i = make_int3(blockIdx.x, blockIdx.y, blockIdx.z);
	int3 blockDim_i = make_int3(para->grid_dim);
	int threadIdx_i = threadIdx.x;
	int bid = GetBlockIdx1D(blockIdx_i, blockDim_i);
	while (threadIdx_i < block_pnum[bid]) {
		int i = block_pidx[bid] + threadIdx_i; // for each particle[i]
		// change position if outside
		float3 bmin = make_float3(para->particle_radius);
		float3 bmax = para->box_length - para->particle_radius;

#ifdef CONFINE_RANDOM
		if (next_pos[i].x <= bmin.x) {
			next_pos[i].x = bmin.x + M_EPS * cudaRandomFloat(devStates, i);
		}
		else if (next_pos[i].x >= bmax.x) {
			next_pos[i].x = bmax.x - M_EPS * cudaRandomFloat(devStates, i);
		}

		if (next_pos[i].y <= bmin.y) {
			next_pos[i].y = bmin.y + M_EPS * cudaRandomFloat(devStates, i);
		}
		else if (next_pos[i].y >= bmax.y) {
			next_pos[i].y = bmax.y - M_EPS * cudaRandomFloat(devStates, i);
		}

		if (next_pos[i].z <= bmin.z) {
			next_pos[i].z = bmin.z + M_EPS * cudaRandomFloat(devStates, i);
		}
		else if (next_pos[i].z >= bmax.z) {
			next_pos[i].z = bmax.z - M_EPS * cudaRandomFloat(devStates, i);
		}
		// change velocity
		velocity[i] = (next_pos[i] - cur_pos[i]) / para->time_delta;
#else
		float ETA = para->eta;
		if (next_pos[i].x <= bmin.x) {
			next_pos[i].x = min(bmax.x, bmin.x + (bmin.x - next_pos[i].x) * ETA);
			velocity[i].x = -velocity[i].x * ETA;
		}
		else if (next_pos[i].x >= bmax.x) {
			next_pos[i].x = max(bmin.x, bmax.x - (next_pos[i].x - bmax.x) * ETA);
			velocity[i].x = -velocity[i].x * ETA;
		}

		if (next_pos[i].y <= bmin.y) {
			next_pos[i].y = min(bmax.y, bmin.y + (bmin.y - next_pos[i].y) * ETA);
			velocity[i].y = -velocity[i].y * ETA;
		}
		else if (next_pos[i].y >= bmax.y) {
			next_pos[i].y = max(bmin.y, bmax.y - (next_pos[i].y - bmax.y) * ETA);
			velocity[i].y = -velocity[i].y * ETA;
		}

		if (next_pos[i].z <= bmin.z) {
			next_pos[i].z = min(bmax.z, bmin.z + (bmin.z - next_pos[i].z) * ETA);
			velocity[i].z = -velocity[i].z * ETA;
		}
		else if (next_pos[i].z >= bmax.z) {
			next_pos[i].z = max(bmin.z, bmax.z - (next_pos[i].z - bmax.z) * ETA);
			velocity[i].z = -velocity[i].z * ETA;
		}
#endif // CONFINE_RANDOM

		threadIdx_i += para->block_size;
	}
}


////////////////////////////////////////////////////////////////////////////////
//
// Update the new density, pressure, velocity and position for each particle
//
////////////////////////////////////////////////////////////////////////////////
__global__ void UpdateParticles(	WCSPHSystem* para,
									int* block_pidx, int* block_pnum,
									float* delta_density, float* density, float* pressure, float* velocity_len,
									float3* cur_pos, float3* next_pos, float3* velocity) {

#ifdef DEBUG
	if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
		printf("Block #(%d,%d,%d) Do UpdateParticles\n", blockIdx.x, blockIdx.y, blockIdx.z);
#endif // DEBUG

	int3 blockIdx_i = make_int3(blockIdx.x, blockIdx.y, blockIdx.z);
	int3 blockDim_i = make_int3(para->grid_dim);
	int threadIdx_i = threadIdx.x;
	int bid = GetBlockIdx1D(blockIdx_i, blockDim_i);
	while (threadIdx_i < block_pnum[bid]) {
		int i = block_pidx[bid] + threadIdx_i;

		density[i] += para->time_delta * delta_density[i];

		pressure[i] = PressureUpdate(density[i], para->rho_0, para->C_s, para->gamma);

#ifdef CONFINE_RANDOM
		velocity[i] = (next_pos[i] - cur_pos[i]) / para->time_delta;
#endif // CONFINE_RANDOM

		velocity[i] *= (1.0 - para->f_air); // air resistence

		velocity_len[i] = Norm2(velocity[i]);

		cur_pos[i] = next_pos[i];

		threadIdx_i += para->block_size;
	}

#ifdef DEBUG
	if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
		printf("Block #(%d,%d,%d) Finish UpdateParticles\n", blockIdx.x, blockIdx.y, blockIdx.z);
#endif // DEBUG

}


////////////////////////////////////////////////////////////////////////////////
//
// Use for debug, output the variable value on gpu
//
////////////////////////////////////////////////////////////////////////////////
__global__ void DebugOutput(		WCSPHSystem* para,
									int* particle_bid, int* block_pidx, int* block_pnum,
									float* delta_density, float* density, float* pressure,
									float3* cur_pos, float3* next_pos, float3* delta_pressure, float3* delta_viscocity, float3* delta_velocity, float3* velocity) {
								
	//for (int i = 0; i < para->grid_size; i++) {
	//	printf("Block #%d:", i);
	//	printf("     \n\t block ipdx: %d, block pnum: %d\n", block_pidx[i], block_pnum[i]);
	//	printf("\n");
	//}

	for (int i = 0; i < para->particle_num; i++) {
		printf("Particle #%d:", i);
		printf("\n\t particle_bid: %d\n\t cur_pos (%f, %f, %f)\n\t next_pos (%f, %f, %f)\n", particle_bid[i], cur_pos[i].x, cur_pos[i].y, cur_pos[i].z, next_pos[i].x, next_pos[i].y, next_pos[i].z);
		printf("\n\t delta_density (%f)\n\t delta_pressure (%f, %f, %f)\n\t delta_viscosity (%f, %f, %f)\n\t delta_velocity (%f, %f, %f)\n", delta_density[i], delta_pressure[i].x, delta_pressure[i].y, delta_pressure[i].z, delta_viscocity[i].x, delta_viscocity[i].y, delta_viscocity[i].z, delta_velocity[i].x, delta_velocity[i].y, delta_velocity[i].z);
		printf("\n\t density (%f)\n\t pressure (%f)\n\t velocity (%f, %f, %f)\n", density[i], pressure[i], velocity[i].x, velocity[i].y, velocity[i].z);
		printf("\n");
	}
}


////////////////////////////////////////////////////////////////////////////////
//
// Smartly choose the time step to calculate
//
////////////////////////////////////////////////////////////////////////////////
__global__ void AdaptiveStep(		WCSPHSystem* para, 
									float* density, 
									float3* delta_velocity, float3* velocity) {

	float max_v = FLT_MIN;
	float max_a = FLT_MIN;
	float max_r = FLT_MIN;
	for (int i = 0; i < para->particle_num; i++) {
		if (Norm2(velocity[i]) > max_v) {
			max_v = Norm2(velocity[i]);
		}

		if (Norm2(delta_velocity[i]) > max_a) {
			max_a = Norm2(delta_velocity[i]);
		}

		if (density[i] > max_r) {
			max_r = density[i];
		}
	}

	float dt_cfl = para->CFL_v * para->h / max_v;
	float dt_f = para->CFL_a * sqrt(para->h / max_a);
	float dt_a = 0.2 * para->h / (para->C_s * pow(sqrt(max_r / para->rho_0), para->gamma));

	para->time_delta = fminf(dt_cfl, fminf(dt_f, dt_a));

}


////////////////////////////////////////////////////////////////////////////////
//
// Find maximum and minimum value of velocity_len for each particle
//
////////////////////////////////////////////////////////////////////////////////
__global__ void FindVelocityLenMinMax(unsigned int blockSize, float* velocity_len, float* g_odata, unsigned int num, bool findmin) {

#ifdef DEBUG
	if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
		printf("Block #(%d,%d,%d) Do FindVelocityLenMinMax\n", blockIdx.x, blockIdx.y, blockIdx.z);
#endif // DEBUG

	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockSize * 2) + tid;
	unsigned int gridSize = blockSize * 2 * gridDim.x;
	if (findmin)
		sdata[tid] = 1e20;
	else sdata[tid] = 0;
	pfunc func = find_minmax[findmin];

	while (i < num) {
		sdata[tid] = func(sdata[tid], velocity_len[i]);
		if (i + blockSize < num)
			sdata[tid] = func(sdata[tid], velocity_len[i + blockSize]);
		i += gridSize;
	}
	__syncthreads();
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = func(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = func(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] = func(sdata[tid], sdata[tid +  64]); } __syncthreads(); }
	if (tid < 32) { FindMinMaxWarpReduce(blockSize, sdata, tid, func); }
	if (tid == 0) { g_odata[blockIdx.x] = sdata[0]; }
	if (tid == 0) { printf("velocity_max: %f\n", g_odata[blockIdx.x]); }

#ifdef DEBUG
	if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
		printf("Block #(%d,%d,%d) Finish FindVelocityLenMinMax\n", blockIdx.x, blockIdx.y, blockIdx.z);
#endif // DEBUG
}


////////////////////////////////////////////////////////////////////////////////
//
// Export particle information to VBO for drawing, blue(0, 0, 1) is slow, white(1, 1, 1) is fast
//
////////////////////////////////////////////////////////////////////////////////
__global__ void ExportParticleInfo(	WCSPHSystem* para,
									int* block_pidx, int* block_pnum,
									float* velocity_len, float* velo_min, float* velo_max,
									float3* cur_pos, float3* pos_info, float3* color_info) {

#ifdef DEBUG
	if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
		printf("Block #(%d,%d,%d) Do ExportParticleInfo\n", blockIdx.x, blockIdx.y, blockIdx.z);
#endif // DEBUG

	int3 blockIdx_i = make_int3(blockIdx.x, blockIdx.y, blockIdx.z);
	int3 blockDim_i = make_int3(para->grid_dim);
	int threadIdx_i = threadIdx.x;
	int bid = GetBlockIdx1D(blockIdx_i, blockDim_i);
	while (threadIdx_i < block_pnum[bid]) {
		int i = block_pidx[bid] + threadIdx_i;
		pos_info[i] = cur_pos[i];
		float percent = NormalizeTo01(velocity_len[i], para->velo_draw_min, para->velo_draw_max);
		//float percent = NormalizeTo01(velocity_len[i], *velo_min, *velo_max);
		color_info[i] = make_float3(percent, percent, 1.0);
		threadIdx_i += para->block_size;
	}

#ifdef DEBUG
	if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0)
		printf("Block #(%d,%d,%d) Finish ExportParticleInfo\n", blockIdx.x, blockIdx.y, blockIdx.z);
#endif // DEBUG

}


////////////////////////////////////////////////////////////////////////////////
//
// Get next frame information
//
////////////////////////////////////////////////////////////////////////////////
void getNextFrame(WCSPHSystem* para, cudaGraphicsResource* position_resource, cudaGraphicsResource* color_resource) {
	
	dim3 blocks(para->grid_dim.x, para->grid_dim.y, para->grid_dim.z);
	dim3 threads(para->block_size);

	unsigned int num = para->particle_num;
	unsigned int thread_num = para->block_size;

	for (int i = 0; i < para->step_each_frame; i++) {

		//DebugOutput <<<1, 1 >>> (sph_device, particle_bid, block_pidx, block_pnum, delta_density, density, pressure, cur_pos, next_pos, delta_pressure, delta_viscosity, delta_velocity, velocity);
		//cudaDeviceSynchronize();

		//ComputeBid <<<1, 1 >>> (sph_device, particle_bid, cur_pos);
		ComputeBid <<<blocks, threads >>> (sph_device, particle_bid, cur_pos);
		cudaDeviceSynchronize();

#ifdef CUDA_MEMCPY_ASYNC
		cudaStream_t stream[kCudaSortArrayCount];
#endif // CUDA_MEMCPY_ASYNC
		for (int k = 1; k < kCudaSortArrayCount; k++) {
#ifdef CUDA_MEMCPY_ASYNC
			checkCudaErrors(cudaStreamCreate(&stream[k]));
			checkCudaErrors(cudaMemcpyAsync(particle_bid + num * k, particle_bid, num * sizeof(int), cudaMemcpyDeviceToDevice, stream[k]));
#else
			checkCudaErrors(cudaMemcpy(particle_bid + num * k, particle_bid, num * sizeof(int), cudaMemcpyDeviceToDevice));
#endif // CUDA_MEMCPY_ASYNC
		}

#ifdef CUDA_MEMCPY_ASYNC
		for (int k = 1; k < kCudaSortArrayCount; k++) {
			checkCudaErrors(cudaStreamSynchronize(stream[k]));
			checkCudaErrors(cudaStreamDestroy(stream[k]));
		}
#endif // CUDA_MEMCPY_ASYNC

		SortParticles <<<kCudaSortArrayCount, 1 >>> (sph_device, particle_bid, density, pressure, cur_pos, velocity);
		cudaDeviceSynchronize();

		ComputeBlockIdxPnum <<<1, 1 >>> (sph_device, particle_bid, block_pidx, block_pnum);
		cudaDeviceSynchronize();

		ComputeDeltaValue <<<blocks, threads >>> (sph_device, block_pidx, block_pnum, delta_density, density, pressure, cur_pos, delta_pressure, delta_viscosity, velocity);
		cudaDeviceSynchronize();

		ComputeVelocity <<<blocks, threads >>> (sph_device, block_pidx, block_pnum, density, cur_pos, delta_pressure, delta_viscosity, delta_velocity, velocity);
		cudaDeviceSynchronize();

		ComputePosition <<<blocks, threads >>> (sph_device, block_pidx, block_pnum, cur_pos, next_pos, velocity);
		cudaDeviceSynchronize();

		ConfineToBoundary <<<blocks, threads >>> (sph_device, devStates, block_pidx, block_pnum, cur_pos, next_pos, velocity);
		cudaDeviceSynchronize();

		UpdateParticles <<<blocks, threads >>> (sph_device, block_pidx, block_pnum, delta_density, density, pressure, velocity_len, cur_pos, next_pos, velocity);
		cudaDeviceSynchronize();
	}

	//FindVelocityLenMinMax <<<1, threads, thread_num * sizeof(float)  >>> (thread_num, velocity_len, velo_min, num, true); // find min
	//cudaDeviceSynchronize();

	//FindVelocityLenMinMax <<<1, threads, thread_num * sizeof(float)  >>> (thread_num, velocity_len, velo_max, num, false); // find max
	//cudaDeviceSynchronize();

	float3* pos_info;
	float3* color_info;
	checkCudaErrors(cudaGraphicsMapResources(1, &position_resource));
	checkCudaErrors(cudaGraphicsMapResources(1, &color_resource));
	cudaDeviceSynchronize();
	size_t pbytes, cbytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&pos_info, &pbytes, position_resource));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&color_info, &cbytes, color_resource));
	cudaDeviceSynchronize();
	
	ExportParticleInfo <<<blocks, threads >>> (sph_device, block_pidx, block_pnum, velocity_len, velo_min, velo_max, cur_pos, pos_info, color_info);
	cudaDeviceSynchronize();
	
	checkCudaErrors(cudaGraphicsUnmapResources(1, &position_resource));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &color_resource));
	cudaDeviceSynchronize();
}
