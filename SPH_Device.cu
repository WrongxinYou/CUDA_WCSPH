#include "sph_host.cuh"
#include "Global.h"

#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>


void InitDeviceSystem(SPHSystem* sys_host) {
	
	/*checkCudaErrors(cudaMalloc(&sys_device, sizeof(SPHSystem)));
	checkCudaErrors(cudaMemcpy(sys_device, sys_host, sizeof(SPHSystem), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	int size = sys_host->particle_num * sizeof(float3);
	checkCudaErrors(cudaMalloc(&sys_device->cur_pos, size));
	checkCudaErrors(cudaMemcpy(sys_device->cur_pos, sys_host->pos_host, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMalloc(&sys_device->color, size));
	checkCudaErrors(cudaMemcpy(sys_device->color, sys_host->color, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());*/

	int size = sys_host->particle_num * sizeof(float3);
	checkCudaErrors(cudaMalloc(&sys_host->cur_pos, size));
	checkCudaErrors(cudaMemcpy(sys_host->cur_pos, sys_host->pos_host, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMalloc(&sys_host->color, size));
	checkCudaErrors(cudaMemcpy(sys_host->color, sys_host->color, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

}

void FreeDeviceSystem(SPHSystem* sys_host) {
	/*checkCudaErrors(cudaFree(sys_device->cur_pos));
	checkCudaErrors(cudaFree(sys_device->color));
	checkCudaErrors(cudaFree(sys_device));*/

	checkCudaErrors(cudaFree(sys_host->cur_pos));
	checkCudaErrors(cudaFree(sys_host->color));
}


__global__ void SortParticles(SPHSystem* sys) {
	int i;
	for (i = 0; i < sys->particle_num; i++) {
		int3 tmp_gid = {
			(int)(sys->cur_pos[i].x / sys->grid_size.x),
			(int)(sys->cur_pos[i].y / sys->grid_size.y),
			(int)(sys->cur_pos[i].z / sys->grid_size.z) };

		sys->particle_gid[i] = tmp_gid.x * sys->grid_num.y * sys->grid_num.z
			+ tmp_gid.y * sys->grid_num.z
			+ tmp_gid.z;
	}
	thrust::stable_sort_by_key(thrust::device, sys->particle_gid,
		sys->particle_gid + sys->particle_num, sys->cur_pos);


	/// TODO: update grid info

}

__device__ void ComputeDensity() {

	/*for p_i in range(total_num_particles[None]) :
		delta_density[p_i] = 0.0
		for j in range(particle_num_neighbors[p_i]) :
			p_j = particle_neighbors[p_i, j]
			p_ij = old_positions[p_i] - old_positions[p_j]
			r_mod = ti.max(p_ij.norm(), epsilon)
			delta_density[p_i] += mass[None] * cubic_kernel(r_mod, h) * 
			(velocity[p_i] - velocity[p_j]).dot(p_ij / r_mod)*/



}
__device__ void ComputePressure() {
	/*for p_i in range(total_num_particles[None]) :
		delta_pressure[p_i] = ti.Vector([0.0 for _ in range(dim)])
		for j in range(particle_num_neighbors[p_i]) :
			p_j = particle_neighbors[p_i, j]
			p_ij = old_positions[p_i] - old_positions[p_j]
			r_mod = ti.max(p_ij.norm(), epsilon)
			delta_pressure[p_i] -= mass[None] * (pressure[p_i] / density[p_j] * *2 + \
				pressure[p_j] / density[p_j] * *2) * \
			cubic_kernel(r_mod, h) * p_ij / r_mod*/


}
__device__ void ComputeViscosity() {
	/*for p_i in range(total_num_particles[None]) :
		delta_viscosity[p_i] = ti.Vector([0.0 for _ in range(dim)])
		for j in range(particle_num_neighbors[p_i]) :
			p_j = particle_neighbors[p_i, j]
			p_ij = old_positions[p_i] - old_positions[p_j]
			r_mod = ti.max(p_ij.norm(), epsilon)

			v_ij = (velocity[p_i] - velocity[p_j]).dot(p_ij)
			# Artifical viscosity
			if v_ij < 0:
	v = -2.0 * alpha * particle_radius * C0 / (density[p_i] + density[p_j])
		delta_viscosity[p_i] -= mass[None] * v_ij * v / (r_mod * *2 + 0.01 * particle_radius * *2) * \
		cubic_kernel(r_mod, h) * p_ij / r_mod*/
}

__device__ void ComputeForce() {
	
}

__device__ void ComputeVelocity() {
	/*for p_i in range(total_num_particles[None]) :
		val = [0.0 for _ in range(dim - 1)]
		val.extend([gravity])
		delta_velocity[p_i] = delta_pressure[p_i] + delta_viscosity[p_i] + ti.Vector(val, dt = ti.f32)
		velocity[p_i] += time_delta[None] * delta_velocity[p_i]*/
}
__device__ void UpdatePosition() {
	/*for p_i in range(total_num_particles[None]) :
		cur_positions[p_i] = old_positions[p_i] + time_delta[None] * velocity[p_i]*/
}

__global__ void ExportParticleInfo(float3* particle_pos, float3* particle_color, SPHSystem* sys) {
	int bid = blockIdx.x * blockDim.y * blockDim.z + blockIdx.y * blockDim.z + blockIdx.z;
	int i;
	for (i = sys->grid_pid[bid]; i < sys->grid_pid[bid] + sys->grid_pnum[bid]; i++) {
		particle_pos[i] = sys->cur_pos[i];
		particle_color[i] = sys->color[i];
	}
}

void getNextFrame(float3* particle_pos, float3* particle_color, SPHSystem* sys) {

	SortParticles<<<1, 1>>>(sys);

	dim3 blocks(sys->grid_num.x, sys->grid_num.y, sys->grid_num.z);
	dim3 threads(256);
	/*ComputeDensity <<<blocks, threads>>> ();
	ComputeForce <<<blocks, threads>>> ();
	UpdatePosition <<<blocks, threads>>> ();*/

	ExportParticleInfo<<<blocks, threads>>>(particle_pos, particle_color, sys);
}
