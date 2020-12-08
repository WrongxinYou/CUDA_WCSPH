#include "SPH_Host.cuh"
#include "Global.h"

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <time.h>

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

float3* velocity = NULL;
float3* delta_velocity = NULL;

float3* viscosity = NULL;
float3* delta_viscosity = NULL;

float* density = NULL;
float* delta_density = NULL;

float* pressure = NULL;
float3* delta_pressure = NULL;


__device__ float cudaRandomFloat(curandState* state, int pid) {

	curandState localState = state[pid];
	curand_init((unsigned int)clock(), pid, 0, &localState);
	return curand_uniform(&localState);
}

__HOSTDEV__ float cubic_kernel(int dim, float r, float hh) {
	// B - cubic spline smoothing kernel
	float res = 0;
	float k = 10.0 / (7.0 * pow(hh, dim) * M_PI);
	float q = r / hh;
	if (q < 1.0 + M_EPS) {
		res = (k / hh) * (-3 * q + 2.25 * pow(q, 2));
	}
	else if (q < 2.0 + M_EPS) {
		res = -0.75 * (k / hh) * pow(2 - q, 2);
	}
	return res;
}
__HOSTDEV__ float p_update(float rho, float rho0, float C0, float gamma) {
	// Weakly compressible, tait function
	float b = rho0 * Pow2(C0) / gamma;
	return b * (pow(rho / rho0, gamma) - 1.0);
}

void InitDeviceSystem(SPHSystem* para, float3* pos_init) {

	int num = para->particle_num;
	checkCudaErrors(cudaMalloc(&sph_device, sizeof(SPHSystem)));
	checkCudaErrors(cudaMemcpy(sph_device, para, sizeof(SPHSystem), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&particle_bid, num * sizeof(int)));

	checkCudaErrors(cudaMalloc(&block_pidx, para->block_num * sizeof(int)));
	checkCudaErrors(cudaMalloc(&block_pnum, para->block_num * sizeof(int)));

	checkCudaErrors(cudaMalloc(&block_offset, 27 * sizeof(int3)));
	checkCudaErrors(cudaMemcpy(block_offset, block_offset_host, 27 * sizeof(int3), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&devStates, num * sizeof(curandState)));

	checkCudaErrors(cudaMalloc(&color, num * sizeof(float3)));

	checkCudaErrors(cudaMalloc(&cur_pos, num * sizeof(float3)));
	checkCudaErrors(cudaMemcpy(cur_pos, pos_init, num * sizeof(float3), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&next_pos, num * sizeof(float3)));

	checkCudaErrors(cudaMalloc(&velocity, num * sizeof(float3)));
	checkCudaErrors(cudaMalloc(&delta_velocity, num * sizeof(float3)));

	checkCudaErrors(cudaMalloc(&viscosity, num * sizeof(float3)));
	checkCudaErrors(cudaMalloc(&delta_viscosity, num * sizeof(float3)));

	checkCudaErrors(cudaMalloc(&density, num * sizeof(float)));
	checkCudaErrors(cudaMalloc(&delta_density, num * sizeof(float)));

	checkCudaErrors(cudaMalloc(&pressure, num * sizeof(float)));
	checkCudaErrors(cudaMalloc(&delta_pressure, num * sizeof(float3)));


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

	checkCudaErrors(cudaFree(velocity));
	checkCudaErrors(cudaFree(delta_velocity));

	checkCudaErrors(cudaFree(viscosity));
	checkCudaErrors(cudaFree(delta_viscosity));

	checkCudaErrors(cudaFree(density));
	checkCudaErrors(cudaFree(delta_density));

	checkCudaErrors(cudaFree(pressure));
	checkCudaErrors(cudaFree(delta_pressure));
}

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
	// for each block 
	for (int ii = 0; ii < 27; ii++) {
		int3 blockIdx_nei = blockIdx_i + _block_offset[ii];
		if (IdxIsValid(blockIdx_nei, blockDim_i)) {
			int bid_nei = GetIdx1D(blockIdx_nei, blockDim_i);
			_delta_density[i] = 0.0;
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
	// for each block 
	for (int ii = 0; ii < 27; ii++) {
		int3 blockIdx_nei = blockIdx_i + block_offset[ii];
		if (IdxIsValid(blockIdx_nei, blockDim_i)) {
			int bid_nei = GetIdx1D(blockIdx_nei, blockDim_i);
			delta_pressure[i] = make_float3(0, 0, 0);
			for (int j = block_pidx[bid_nei]; j < block_pidx[bid_nei] + block_pnum[bid_nei]; j++) { // find neighbour particle[j]
				if (i == j) continue;
				float3 vec_ij = cur_pos[i] - cur_pos[j];
				float len_ij = Norm2(vec_ij);
				len_ij = fmaxf(len_ij, M_EPS);
				delta_pressure[i] -= para->mass * cubic_kernel(para->dim, len_ij, para->h) * (vec_ij / len_ij) *
					(pressure[i] / Pow2(fmaxf(M_EPS, density[j])) + pressure[j] / Pow2(fmaxf(M_EPS, density[j])));
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
	// for each block 
	for (int ii = 0; ii < 27; ii++) {
		int3 blockIdx_nei = blockIdx_i + block_offset[ii];
		if (IdxIsValid(blockIdx_nei, blockDim_i)) {
			int bid_nei = GetIdx1D(blockIdx_nei, blockDim_i);
			delta_viscosity[i] = make_float3(0, 0, 0);
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
	float3 val = make_float3(0, para->gravity, 0);
	delta_velocity[i] = delta_pressure[i] + delta_viscosity[i] + val;
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

__global__ void AdaptiveStep() {



}



__global__ void SortParticles(	SPHSystem* para,
								int* particle_bid, int* block_pidx, int* block_pnum,
								float3* cur_pos) {

	int3 blockDim_i = para->block_dim;
	for (int i = 0; i < para->particle_num; i++) {
		int3 tmp_bid = float3TOint3(cur_pos[i] / para->block_size);

		particle_bid[i] = GetIdx1D(tmp_bid, blockDim_i);
	}

#ifdef DEBUG
	//for (int i = 0; i < para->particle_num; i++)
	//	printf("particle #%d, bid: %d\n", i, particle_bid[i]);

	//printf("begin sort\n");
#endif // DEBUG

	thrust::stable_sort_by_key(thrust::device, particle_bid, particle_bid + para->particle_num, cur_pos);

#ifdef DEBUG
	//printf("after sort\n");

	//for (int i = 0; i < para->particle_num; i++)
	//	printf("particle #%d, bid: %d\n", i, particle_bid[i]);

	//printf("begin count\n");
#endif // DEBUG

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

#ifdef DEBUG
	//printf("after count\n");

	//for (int i = 0; i < para->block_num; i++)
	//	printf("block #%d, pnum: %d, pidx: %d\n", i, block_pnum[i], block_pidx[i]);
#endif // DEBUG

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


void getNextFrame(SPHSystem* para, cudaGraphicsResource* position_resource, cudaGraphicsResource* color_resource) {
	
	dim3 blocks(para->block_dim.x, para->block_dim.y, para->block_dim.z);
	dim3 threads(256);

	for (int i = 0; i < para->step_each_frame; i++) {

		SortParticles <<<1, 1 >>> (sph_device, particle_bid, block_pidx, block_pnum, cur_pos);
		cudaDeviceSynchronize();

		ComputeDensity <<<blocks, threads >>> (sph_device, block_pidx, block_pnum, delta_density, block_offset, cur_pos, velocity);
		cudaDeviceSynchronize();

		ComputePressure <<<blocks, threads >>> (sph_device, block_pidx, block_pnum, pressure, density, block_offset, cur_pos, delta_pressure);
		cudaDeviceSynchronize();

		ComputeViscosity <<<blocks, threads >>> (sph_device, block_pidx, block_pnum, density, block_offset, cur_pos, delta_viscosity, velocity);
		cudaDeviceSynchronize();

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
