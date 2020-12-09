#include "SPH_Host.cuh"
#include "Global.h"

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <time.h>

#define COPY_TIME 5

// Device 
int3 block_offset_host[] = {
	{-1, -1, -1}, {-1, -1, 0}, {-1, -1, 1},
	{-1, 0, -1}, {-1, 0, 0}, {-1, 0, 1},
	{-1, 1, -1}, {-1, 1, 0}, {-1, 1, 1},
	{0, -1, -1}, {0, -1, 0}, {0, -1, 1},
	{0, 0, -1}, {0, 0, 0}, {0, 0, 1},
	{0, 1, -1}, {0, 1, 0}, {0, 1, 0},
	{1, -1, -1}, {1, -1, 0}, {1, -1, 1},
	{1, 0, -1}, {1, 0, 0}, {1, 0, 1},
	{1, 1, -1}, {1, 1, 0}, {1, 1, 1},
};

SPHSystem* sph_device = NULL;

int* particle_bid = NULL; // each particle belongs to which block
int* block_pidx = NULL; // first particle index in grid
int* block_pnum = NULL; // particle number in grid
int3* block_offset = NULL;

curandState* devStates = NULL;

float3* color = NULL; // color of particles
float3* cur_pos = NULL;
float3* next_pos = NULL;

float* density = NULL;
float* delta_density = NULL;

float* pressure = NULL;
float3* delta_pressure = NULL;

float3* viscosity = NULL;
float3* delta_viscosity = NULL;

float3* velocity = NULL;
float3* delta_velocity = NULL;

__device__ float cudaRandomFloat(curandState* state, int pid) {

	curandState localState = state[pid];
	curand_init((unsigned int)clock(), pid, 0, &localState);
	return curand_uniform(&localState);
}

__HOSTDEV__ float cubic_kernel(int dim, float dist, float cutoff) {
	// B - cubic spline smoothing kernel
	float res = 0;
	float k;
	if (dim == 1) {
		k  = 2.0 / (3.0 * pow(cutoff, dim) * M_PI);
	}
	else if (dim == 2) {
		k = 10.0 / (7.0 * pow(cutoff, dim) * M_PI);
	}
	else { // dim == 3
		k = 1.0 / (1.0 * pow(cutoff, dim) * M_PI);
	}
	float q = dist / cutoff;
	if (q <= 1.0 + M_EPS) {
		res = k * (1.0 - 3.0 / 2.0 * Pow2(q) * (1.0 - q / 2.0));
		//res = (k / cutoff) * (-3 * q + 2.25 * Pow2(q));
	}
	else if (q <= 2.0 + M_EPS) {
		res = k / 4.0 * Pow3(2.0f - q);
		//res = -0.75 * (k / cutoff) * Pow2(2 - q);
	}
	return res;
}

__HOSTDEV__ float p_update(float rho, float rho0, float C0, float gamma) {
	// Weakly compressible, tait function
	float b = rho0 * Pow2(C0) / gamma;
	return b * (pow(rho / rho0, gamma) - 1.0);
}

void InitDeviceSystem(SPHSystem* para, float* dens_init, float3* pos_init, float3* velo_init) {

	int num = para->particle_num;
	checkCudaErrors(cudaMalloc(&sph_device, sizeof(SPHSystem)));
	checkCudaErrors(cudaMemcpy(sph_device, para, sizeof(SPHSystem), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&particle_bid, COPY_TIME * num * sizeof(int)));

	checkCudaErrors(cudaMalloc(&block_pidx, para->block_num * sizeof(int)));
	checkCudaErrors(cudaMalloc(&block_pnum, para->block_num * sizeof(int)));

	checkCudaErrors(cudaMalloc(&block_offset, 27 * sizeof(int3)));
	checkCudaErrors(cudaMemcpy(block_offset, block_offset_host, 27 * sizeof(int3), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&devStates, num * sizeof(curandState)));

	checkCudaErrors(cudaMalloc(&color, num * sizeof(float3)));

	checkCudaErrors(cudaMalloc(&cur_pos, num * sizeof(float3)));
	checkCudaErrors(cudaMemcpy(cur_pos, pos_init, num * sizeof(float3), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&next_pos, num * sizeof(float3)));
	checkCudaErrors(cudaMemset(next_pos, 0, num * sizeof(float3)));

	checkCudaErrors(cudaMalloc(&density, num * sizeof(float)));
	checkCudaErrors(cudaMemcpy(density, dens_init, num * sizeof(float3), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&delta_density, num * sizeof(float)));

	checkCudaErrors(cudaMalloc(&pressure, num * sizeof(float)));
	checkCudaErrors(cudaMemset(pressure, 0, num * sizeof(float)));

	checkCudaErrors(cudaMalloc(&delta_pressure, num * sizeof(float3)));

	checkCudaErrors(cudaMalloc(&viscosity, num * sizeof(float3)));
	checkCudaErrors(cudaMemset(viscosity, 0, num * sizeof(float3)));

	checkCudaErrors(cudaMalloc(&delta_viscosity, num * sizeof(float3)));

	checkCudaErrors(cudaMalloc(&velocity, num * sizeof(float3)));
	checkCudaErrors(cudaMemcpy(velocity, velo_init, num * sizeof(float3), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&delta_velocity, num * sizeof(float3)));

	/*checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());*/
}

void FreeDeviceSystem(SPHSystem* para) {

	delete para;

	checkCudaErrors(cudaFree(sph_device));

	checkCudaErrors(cudaFree(particle_bid));
	checkCudaErrors(cudaFree(block_pidx));
	checkCudaErrors(cudaFree(block_pnum));

	checkCudaErrors(cudaFree(block_offset));

	checkCudaErrors(cudaFree(devStates));

	checkCudaErrors(cudaFree(color));

	checkCudaErrors(cudaFree(cur_pos));
	checkCudaErrors(cudaFree(next_pos));

	checkCudaErrors(cudaFree(density));
	checkCudaErrors(cudaFree(delta_density));

	checkCudaErrors(cudaFree(pressure));
	checkCudaErrors(cudaFree(delta_pressure));

	checkCudaErrors(cudaFree(viscosity));
	checkCudaErrors(cudaFree(delta_viscosity));

	checkCudaErrors(cudaFree(velocity));
	checkCudaErrors(cudaFree(delta_velocity));
}

__global__ void ComputeAll( SPHSystem* _para,
							int* _block_pidx, int* _block_pnum,
							float* _delta_density, float* _density,  float* _pressure,
							int3* _block_offset,
							float3* _cur_pos, float3* _delta_pressure, float3* _delta_viscosity, float3* _velocity) {

	int3 blockIdx_i = make_int3(blockIdx.x, blockIdx.y, blockIdx.z);
	int3 blockDim_i = _para->block_dim;
	int3 threadIdx_i = make_int3(threadIdx.x, threadIdx.y, threadIdx.z);
	int bid = GetIdx1D(blockIdx_i, blockDim_i);

	if (threadIdx.x >= _block_pnum[bid]) { return; }
	// for each particle[i]
	int i = _block_pidx[bid] + threadIdx.x;
	// Initialize
	_delta_density[i] = 0.0;
	_delta_pressure[i] = make_float3(0, 0, 0);
	_delta_viscosity[i] = make_float3(0, 0, 0);

	// for each block 
	for (int ii = 0; ii < 27; ii++) {
		int3 blockIdx_nei = blockIdx_i + _block_offset[ii];
		if (IdxIsValid(blockIdx_nei, blockDim_i)) {
			int bid_nei = GetIdx1D(blockIdx_nei, blockDim_i);
			// find neighbour particle[j]
			for (int j = _block_pidx[bid_nei]; j < _block_pidx[bid_nei] + _block_pnum[bid_nei]; j++) {
				if (i == j) continue;
				float3 vec_ij = _cur_pos[i] - _cur_pos[j];
				float len_ij = Norm2(vec_ij);
				len_ij = fmaxf(len_ij, M_EPS);
				float kernel = cubic_kernel(_para->dim, len_ij, _para->h);
				// cut off length
				if (kernel <= 0 + M_EPS) {
					continue;
				}
				if ((len_ij / _para->h) > 2) {
					continue;
				}

				// Density
				_delta_density[i] += _para->mass * kernel * dot((_velocity[i] - _velocity[j]), (vec_ij / len_ij));

				// Pressure
				_delta_pressure[i] -= _para->mass * kernel * (vec_ij / len_ij) *
					(_pressure[i] / Pow2(_density[i] + M_EPS) + _pressure[j] / Pow2(_density[j] + M_EPS));

				// Viscosity
				float v_ij = dot(_velocity[i] - _velocity[j], vec_ij);
				if (v_ij < 0) {
					float v = -2.0 * _para->alpha * _para->particle_radius * _para->C0 / fmaxf(M_EPS, _density[i] + _density[j]);
					_delta_viscosity[i] -= _para->mass * kernel * (vec_ij / len_ij) *
						v_ij * v / (Pow2(len_ij) + 0.01 * Pow2(_para->particle_radius));
				}
			}
		}
	}
}

/*
__global__ void ComputeDensity(	SPHSystem* _para,
								int* _block_pidx, int* _block_pnum,
								float* _delta_density,
								int3* _block_offset,
								float3* _cur_pos, float3* _velocity) {

	int3 blockIdx_i = make_int3(blockIdx.x, blockIdx.y, blockIdx.z);
	int3 blockDim_i = _para->block_dim;
	int3 threadIdx_i = make_int3(threadIdx.x, threadIdx.y, threadIdx.z);
	int bid = GetIdx1D(blockIdx_i, blockDim_i);
	//printf("tid: %d, bid: %d, pnum: %d\n", threadIdx.x, bid, _block_pnum[bid]);

	if (threadIdx.x >= _block_pnum[bid]) { return; }
	int i = _block_pidx[bid] + threadIdx.x; // for each particle[i]
	_delta_density[i] = 0.0;
	// for each block 
	for (int ii = 0; ii < 27; ii++) {
		int3 blockIdx_nei = blockIdx_i + _block_offset[ii];
		if (IdxIsValid(blockIdx_nei, blockDim_i)) {
			int bid_nei = GetIdx1D(blockIdx_nei, blockDim_i);
			for (int j = _block_pidx[bid_nei]; j < _block_pidx[bid_nei] + _block_pnum[bid_nei]; j++) { // find neighbour particle[j]
				if (i == j) continue;
				float3 vec_ij = _cur_pos[i] - _cur_pos[j];
				float len_ij = Norm2(vec_ij);
				len_ij = fmaxf(len_ij, M_EPS);
				_delta_density[i] += _para->mass * cubic_kernel(_para->dim, len_ij, _para->h) * dot((_velocity[i] - _velocity[j]), (vec_ij / len_ij));
			}
		}
	}

#ifdef DEBUG
	//printf("=====Density=====\n (%d %d %d) (%d): #i: %d, curpos: (%f %f %f)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, i, _cur_pos[i].x, _cur_pos[i].y, _cur_pos[i].z);
#endif // DEBUG
}


__global__ void ComputePressure(SPHSystem* para,
								int* block_pidx, int* block_pnum,
								float* pressure, float* density,
								int3* block_offset,
								float3* cur_pos, float3* delta_pressure) {

	int3 blockIdx_i = make_int3(blockIdx.x, blockIdx.y, blockIdx.z);
	int3 blockDim_i = para->block_dim;
	int3 threadIdx_i = make_int3(threadIdx.x, threadIdx.y, threadIdx.z);
	int bid = GetIdx1D(blockIdx_i, blockDim_i);
	if (threadIdx.x >= block_pnum[bid]) { return; }
	int i = block_pidx[bid] + threadIdx.x; // for each particle[i]
	delta_pressure[i] = make_float3(0, 0, 0);
	// for each block 
	for (int ii = 0; ii < 27; ii++) {
		int3 blockIdx_nei = blockIdx_i + block_offset[ii];
		if (IdxIsValid(blockIdx_nei, blockDim_i)) {
			int bid_nei = GetIdx1D(blockIdx_nei, blockDim_i);
			for (int j = block_pidx[bid_nei]; j < block_pidx[bid_nei] + block_pnum[bid_nei]; j++) { // find neighbour particle[j]
				if (i == j) continue;
				float3 vec_ij = cur_pos[i] - cur_pos[j];
				float len_ij = Norm2(vec_ij);
				len_ij = fmaxf(len_ij, M_EPS);
				delta_pressure[i] -= para->mass * cubic_kernel(para->dim, len_ij, para->h) * (vec_ij / len_ij) *
					(_pressure[i] / Pow2(_density[i] + M_EPS) + _pressure[j] / Pow2(_density[j] + M_EPS));
			}
		}
	}

#ifdef DEBUG
	//printf("=====Pressure=====\n (%d %d %d) (%d): #i: %d, curpos: (%f %f %f)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, i, cur_pos[i].x, cur_pos[i].y, cur_pos[i].z);
#endif // DEBUG
}


__global__ void ComputeViscosity(	SPHSystem* para,
									int* block_pidx, int* block_pnum,
									float* density,
									int3* block_offset,
									float3* cur_pos, float3* delta_viscosity, float3* velocity) {

	int3 blockIdx_i = make_int3(blockIdx.x, blockIdx.y, blockIdx.z);
	int3 blockDim_i = para->block_dim;
	int3 threadIdx_i = make_int3(threadIdx.x, threadIdx.y, threadIdx.z);
	int bid = GetIdx1D(blockIdx_i, blockDim_i);
	if (threadIdx.x >= block_pnum[bid]) { return; }
	int i = block_pidx[bid] + threadIdx.x; // for each particle[i]
	delta_viscosity[i] = make_float3(0, 0, 0);
	// for each block 
	for (int ii = 0; ii < 27; ii++) {
		int3 blockIdx_nei = blockIdx_i + block_offset[ii];
		if (IdxIsValid(blockIdx_nei, blockDim_i)) {
			int bid_nei = GetIdx1D(blockIdx_nei, blockDim_i);
			for (int j = block_pidx[bid_nei]; j < block_pidx[bid_nei] + block_pnum[bid_nei]; j++) { // find neighbour particle[j]
				if (i == j) continue;
				float3 vec_ij = cur_pos[i] - cur_pos[j];
				float len_ij = Norm2(vec_ij);
				len_ij = fmaxf(len_ij, M_EPS);
				float v_ij = dot(velocity[i] - velocity[j], vec_ij);
				// Artifical ciscosity
				if (v_ij < 0) {
					float v = -2.0 * para->alpha * para->particle_radius * para->C0 / fmaxf(M_EPS, density[i] + density[j]);
					delta_viscosity[i] -= para->mass * cubic_kernel(para->dim, len_ij, para->h) * (vec_ij / len_ij) *
						v_ij * v / (Pow2(len_ij) + 0.01 * Pow2(para->particle_radius));
				}
			}
		}
	}

#ifdef DEBUG
	//printf("=====Viscosity=====\n (%d %d %d) (%d): #i: %d, curpos: (%f %f %f)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, i, cur_pos[i].x, cur_pos[i].y, cur_pos[i].z);
#endif // DEBUG
}
*/

__global__ void ComputeVelocity(SPHSystem* para,
								int* block_pidx, int* block_pnum,
								float* density,
								float3* cur_pos, float3* delta_pressure, float3* delta_viscosity, float3* delta_velocity, float3* velocity) {

	int3 blockIdx_i = make_int3(blockIdx.x, blockIdx.y, blockIdx.z);
	int3 blockDim_i = para->block_dim;
	int3 threadIdx_i = make_int3(threadIdx.x, threadIdx.y, threadIdx.z);
	int bid = GetIdx1D(blockIdx_i, blockDim_i);
	if (threadIdx.x >= block_pnum[bid]) { return; }
	int i = block_pidx[bid] + threadIdx.x; // for each particle[i]
	float3 G = make_float3(0, para->gravity, 0);
	delta_velocity[i] = delta_pressure[i] + delta_viscosity[i] + G;
	velocity[i] += para->time_delta * delta_velocity[i];

#ifdef DEBUG
	//printf("=====Velocity=====\n (%d %d %d) (%d): #i: %d, curpos: (%f %f %f)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, i, cur_pos[i].x, cur_pos[i].y, cur_pos[i].z);
#endif // DEBUG
}


__global__ void ComputePosition(SPHSystem* para,
								int* block_pidx, int* block_pnum,
								float3* cur_pos, float3* next_pos, float3* velocity) {

	int3 blockIdx_i = make_int3(blockIdx.x, blockIdx.y, blockIdx.z);
	int3 blockDim_i = para->block_dim;
	int3 threadIdx_i = make_int3(threadIdx.x, threadIdx.y, threadIdx.z);
	int bid = GetIdx1D(blockIdx_i, blockDim_i); 
	if (threadIdx.x >= block_pnum[bid]) { return; }
	int i = block_pidx[bid] + threadIdx.x; // for each particle[i]
	next_pos[i] = cur_pos[i] + para->time_delta * velocity[i];

#ifdef DEBUG
	//printf("=====Position=====\n (%d %d %d) (%d): #i: %d, curpos: (%f %f %f)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, i, cur_pos[i].x, cur_pos[i].y, cur_pos[i].z);
	//printf("=====Position=====\n (%d %d %d) (%d): #i: %d, curpos: (%f %f %f)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, i, velocity[i].x, velocity[i].y, velocity[i].z);
#endif // DEBUG
}


__global__ void ConfineToBoundary(	SPHSystem* para, curandState* devStates,
									int* block_pidx, int* block_pnum, 
									float3* cur_pos, float3* next_pos, float3* velocity) {

	int3 blockIdx_i = make_int3(blockIdx.x, blockIdx.y, blockIdx.z);
	int3 blockDim_i = para->block_dim;
	int3 threadIdx_i = make_int3(threadIdx.x, threadIdx.y, threadIdx.z);
	int bid = GetIdx1D(blockIdx_i, blockDim_i);
	if (threadIdx.x >= block_pnum[bid]) { return; }
	int i = block_pidx[bid] + threadIdx.x; // for each particle[i]
	// change position if outside
	float3 bmin = make_float3(para->particle_radius);
	float3 bmax = para->box_size - para->particle_radius;
	if (next_pos[i].x <= bmin.x) {
		next_pos[i].x = bmin.x + M_EPS * cudaRandomFloat(devStates, i);
	}
	else if (next_pos[i].x >= bmax.x) {
		next_pos[i].x = bmax.x - M_EPS * cudaRandomFloat(devStates, i);
	}

	if (next_pos[i].y <= bmin.y){
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
}


__global__ void Update(	SPHSystem* para, 
						int* block_pidx, int* block_pnum,
						float* delta_density, float* density, float* pressure, 
						float3* cur_pos, float3* next_pos, float3* velocity) {

	int3 blockIdx_i = make_int3(blockIdx.x, blockIdx.y, blockIdx.z);
	int3 blockDim_i = para->block_dim;
	int3 threadIdx_i = make_int3(threadIdx.x, threadIdx.y, threadIdx.z);
	int bid = GetIdx1D(blockIdx_i, blockDim_i);
	if (threadIdx.x >= block_pnum[bid]) { return; }
	int i = block_pidx[bid] + threadIdx.x;

	density[i] += para->time_delta * delta_density[i];

	pressure[i] = p_update(density[i], para->rho0, para->C0, para->gamma);

	velocity[i] = (next_pos[i] - cur_pos[i]) / para->time_delta;

	cur_pos[i] = next_pos[i];

#ifdef DEBUG
	//printf("=====Update=====\n (%d %d %d) (%d): #i: %d, curpos: (%f %f %f)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, i, cur_pos[i].x, cur_pos[i].y, cur_pos[i].z);
#endif // DEBUG

}


__global__ void AdaptiveStep(	SPHSystem* para,
								float* delta_density, float* density, float* pressure,
								float3* delta_pressure, float3* delta_viscocity, float3* delta_velocity, float3* velocity) {
								
	for (int i = 0; i < para->particle_num; i++) {
		printf("#%d: \n\t delta_density (%f)\n\t delta_pressure (%f, %f, %f)\n\t delta_velocity (%f, %f, %f)\n", i, delta_density[i], delta_pressure[i].x, delta_pressure[i].y, delta_pressure[i].z, delta_velocity[i].x, delta_velocity[i].y, delta_velocity[i].z);
		printf("     \n\t density (%f)\n\t pressure (%f)\n\t velocity (%f, %f, %f)\n", density[i], pressure[i], velocity[i].x, velocity[i].y, velocity[i].z);
	}
}


__global__ void ComputeBid(	SPHSystem* para,
							int* particle_bid,
							float3* cur_pos) {

	// compute block_id for each particle
	int num = para->particle_num;
	int3 blockDim_i = para->block_dim;
	for (int i = 0; i < num; i++) {
		int3 tmp_bid = float3TOint3(cur_pos[i] / para->block_size);
		particle_bid[i] = GetIdx1D(tmp_bid, blockDim_i);
	}

	// copy for sorting
	for (int i = 0; i < num; i++) {
		for (int k = 1; k < COPY_TIME; k++) {
			particle_bid[i + num * k] = particle_bid[i];
		}
	}
}

__global__ void SortParticles(	SPHSystem* para,
								int* particle_bid,
								float* density, float* pressure,
								float3* cur_pos, float3* viscosity, float3* velocity) {

	int num = para->particle_num;
	if (blockIdx.x == 1) {
		thrust::stable_sort_by_key(thrust::device, particle_bid + num * 0, particle_bid + num * 1, cur_pos);
	}
	else if (blockIdx.x == 2) {
		thrust::stable_sort_by_key(thrust::device, particle_bid + num * 1, particle_bid + num * 2, density);
	}
	else if (blockIdx.x == 3) {
		thrust::stable_sort_by_key(thrust::device, particle_bid + num * 2, particle_bid + num * 3, pressure);
	}
	else if (blockIdx.x == 4) {
		thrust::stable_sort_by_key(thrust::device, particle_bid + num * 3, particle_bid + num * 4, viscosity);
	}
	else if (blockIdx.x == 5) {
		thrust::stable_sort_by_key(thrust::device, particle_bid + num * 4, particle_bid + num * 5, velocity);
	}
	//else if (blockIdx.x == 6) {
	//	thrust::stable_sort_by_key(thrust::device, particle_bid + num * 5, particle_bid + num * 6, );
	//}

}
	
__global__ void ComputeBlockIdxPnum(SPHSystem* para,
									int* particle_bid, int* block_pidx, int* block_pnum) {

	for (int i = 0; i < para->block_num; i++) {
		block_pidx[i] = -1;
		block_pnum[i] = 0;
	}

	for (int i = 0; i < para->particle_num; i++) {
		if (i == 0 || particle_bid[i] != particle_bid[i - 1]) {
			block_pidx[particle_bid[i]] = i;
		}
		block_pnum[particle_bid[i]]++;
	}
}

__global__ void ExportParticleInfo(	SPHSystem* para,
									int* block_pidx, int* block_pnum,
									float3* cur_pos, float3* pos_info, float3* color_info) {

	int3 blockIdx_i = make_int3(blockIdx.x, blockIdx.y, blockIdx.z);
	int3 blockDim_i = para->block_dim;
	int bid = GetIdx1D(blockIdx_i, blockDim_i);
	int i;
	for (i = block_pidx[bid]; i < block_pidx[bid] + block_pnum[bid]; i++) {
		pos_info[i] = cur_pos[i];
		color_info[i] = make_float3(1, 1, 1);
	}
}

//void getFirstFrame(SPHSystem* para, cudaGraphicsResource* position_resource, cudaGraphicsResource* color_resource) {
//
//	dim3 blocks(para->block_dim.x, para->block_dim.y, para->block_dim.z);
//	dim3 threads(256);
//	SortParticles <<<1, 1 >>> (sph_device, particle_bid, block_pidx, block_pnum, cur_pos);
//	cudaDeviceSynchronize();
//	float3* pos_info;
//	float3* color_info;
//	checkCudaErrors(cudaGraphicsMapResources(1, &position_resource, 0));
//	checkCudaErrors(cudaGraphicsMapResources(1, &color_resource, 0));
//	size_t pbytes, cbytes;
//	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&pos_info, &pbytes, position_resource));
//	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&color_info, &cbytes, color_resource));
//
//	ExportParticleInfo <<<blocks, 1 >>> (sph_device, block_pidx, block_pnum, cur_pos, pos_info, color_info);
//
//	checkCudaErrors(cudaGraphicsUnmapResources(1, &position_resource, 0));
//	checkCudaErrors(cudaGraphicsUnmapResources(1, &color_resource, 0));
//}

void getNextFrame(SPHSystem* para, cudaGraphicsResource* position_resource, cudaGraphicsResource* color_resource) {
	
	dim3 blocks(para->block_dim.x, para->block_dim.y, para->block_dim.z);
	dim3 threads(para->block_thread_num);

	for (int i = 0; i < para->step_each_frame; i++) {

		AdaptiveStep <<<1, 1 >>> (sph_device, delta_density, density, pressure, delta_pressure, delta_viscosity, delta_velocity, velocity);
		cudaDeviceSynchronize();

		ComputeBid <<<1, 1 >>> (sph_device, particle_bid, cur_pos);
		cudaDeviceSynchronize();

		SortParticles <<<COPY_TIME, 1 >>> (sph_device, particle_bid, density, pressure, cur_pos, viscosity, velocity);
		cudaDeviceSynchronize();

		ComputeBlockIdxPnum <<<1, 1 >>> (sph_device, particle_bid, block_pidx, block_pnum);
		cudaDeviceSynchronize();

		ComputeAll <<<blocks, threads >>> (sph_device, block_pidx, block_pnum, delta_density, density, pressure, block_offset, cur_pos, delta_pressure, delta_viscosity, velocity);
		cudaDeviceSynchronize();

		//ComputeDensity <<<blocks, threads >>> (sph_device, block_pidx, block_pnum, delta_density, block_offset, cur_pos, velocity);
		//cudaDeviceSynchronize();

		//ComputePressure <<<blocks, threads >>> (sph_device, block_pidx, block_pnum, pressure, density, block_offset, cur_pos, delta_pressure);
		//cudaDeviceSynchronize();

		//ComputeViscosity <<<blocks, threads >>> (sph_device, block_pidx, block_pnum, density, block_offset, cur_pos, delta_viscosity, velocity);
		//cudaDeviceSynchronize();

		ComputeVelocity <<<blocks, threads >>> (sph_device, block_pidx, block_pnum, density, cur_pos, delta_pressure, delta_viscosity, delta_velocity, velocity);
		cudaDeviceSynchronize();

		ComputePosition <<<blocks, threads >>> (sph_device, block_pidx, block_pnum, cur_pos, next_pos, velocity);
		cudaDeviceSynchronize();

		ConfineToBoundary <<<blocks, threads >>> (sph_device, devStates, block_pidx, block_pnum, cur_pos, next_pos, velocity);
		cudaDeviceSynchronize();

		Update <<<blocks, threads >>> (sph_device, block_pidx, block_pnum, delta_density, density, pressure, cur_pos, next_pos, velocity);
		cudaDeviceSynchronize();
	}

	float3* pos_info;
	float3* color_info;
	checkCudaErrors(cudaGraphicsMapResources(1, &position_resource, 0));
	checkCudaErrors(cudaGraphicsMapResources(1, &color_resource, 0));
	size_t pbytes, cbytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&pos_info, &pbytes, position_resource));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&color_info, &cbytes, color_resource));

	ExportParticleInfo <<<blocks, 1 >>> (sph_device, block_pidx, block_pnum, cur_pos, pos_info, color_info);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &position_resource, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &color_resource, 0));
}
