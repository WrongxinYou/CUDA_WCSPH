#include "SPHSystem.h"
#include "Global.h"

#include <iostream>
#include <time.h>

float RandomFloat(float a, float b) {
	float random = ((float)rand()) / (float)RAND_MAX;
	float diff = b - a;
	float r = random * diff;
	return a + r;
}

SPHSystem::SPHSystem() {

	// SPH Parameters
	dim = 3;
	particle_dim = make_int3(3, 3, 3);
	particle_num = particle_dim.x * particle_dim.y * particle_dim.z;
	particle_radius = 0.1;
	velo_init_min = make_float3(0, -1, 0);
	velo_init_max = make_float3(0.1, -1, 0.1);

	// Device Parameters
	block_dim = make_int3(3, 3, 3);
	block_num = block_dim.x * block_dim.y * block_dim.z;
	block_size = box_size / block_dim; //  block_dim <= 3/(4 cutoff) <= 3/(4 * 1.3 radius)
	block_thread_num = 256;

	// Draw Parameters
	step_each_frame = 5;
	box_size = make_float3(1.0, 1.0, 1.0); 
	box_margin = box_size * 0.1;

	// Function Parameters
	rho0 = 1000.0;  // reference density
	gamma = 7.0;
	h = 1.3 * particle_radius;
	gravity = -9.8 * 30;
	alpha = 0.3; // between 0.08 and 0.5
	C0 = 200;
	CFL_v = 0.20;
	CFL_a = 0.20;
	poly6_factor = 315.0 / 64.0 / M_PI;
	spiky_grad_factor = -45.0 / M_PI;
	mass = pow(particle_radius, dim) * rho0;
	time_delta = 0.1 * h / C0;
	eta = 0.8; // confine boundary loss coefficient
}

//SPHSystem::SPHSystem() {}

SPHSystem::~SPHSystem() {

}

// Initialize particles position
float3* SPHSystem::InitializePosition() {
	float3* pos_init = new float3[particle_num];
	float3 gap = box_size - box_margin - box_margin;
	gap = gap / (particle_dim + 1);

	for (int i = 0; i < particle_dim.x; i++)
	{
		for (int j = 0; j < particle_dim.y; j++)
		{
			for (int k = 0; k < particle_dim.z; k++)
			{
				int3 ii = make_int3(i, j, k);
				int index = GetIdx1D(ii, particle_dim);
				float3 p = gap * (ii + 1);
				pos_init[index] = box_margin + p;
			}
		}
	}
	return pos_init;
}

float3* SPHSystem::InitializeVelocity() {
	srand(time(0));
	float3* velo_init = new float3[particle_num];
	for (int i = 0; i < particle_num; i++) {
		velo_init[i] = M_EPS + make_float3(
			RandomFloat(velo_init_min.x, velo_init_max.x),
			RandomFloat(velo_init_min.y, velo_init_max.y),
			RandomFloat(velo_init_min.z, velo_init_max.z));
	}
	return velo_init;
}

float* SPHSystem::InitializeDensity() {
	float* dens_init = new float[particle_num];
	for (int i = 0; i < particle_num; i++) {
		dens_init[i] = rho0;
	}
	return dens_init;
}

#ifdef DEBUG
void SPHSystem::Debug()
{
	for (int i = 0; i < particle_num; i++)
		std::cout << pos_init[i] << std::endl;

	//std::cout << "float3:  " << sizeof(float3) << std::endl;
	//std::cout << "*float3:  " << sizeof(float3*) << std::endl;
	//std::cout << "int3:  " << sizeof(int3) << std::endl;
	//std::cout << "*int3:  " << sizeof(int3*) << std::endl;
	//std::cout << "float:  " << sizeof(float) << std::endl;
	//std::cout << "*float:  " << sizeof(float*) << std::endl;
	//std::cout << "int:  " << sizeof(int) << std::endl;
	//std::cout << "*int:  " << sizeof(int*) << std::endl;
}
#endif // DEBUG

