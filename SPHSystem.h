#pragma once
#include "Global.h"
//#include "json.hpp"
//using json = nlohmann::json;

class SPHSystem
{
public:

	// SPH Parameters
	int dim;
	int3 particle_dim; // number of particles in x, y, z axis to intialize
	int particle_num;
	float particle_radius;
	float3 velo_init_min;
	float3 velo_init_max;

	// Device Parameters
	float3 block_size;
	int3 block_dim;
	int block_num;
	int block_thread_num;

	// Draw Parameters
	int step_each_frame;
	float3 box_size;
	float3 box_margin;

	// Function Parameters
	float rho0;  // reference density
	float gamma;
	float h;
	float gravity;
	float alpha;
	float C0;
	float CFL_v;
	float CFL_a;
	float poly6_factor;
	float spiky_grad_factor;
	float cubic_factor1D;
	float cubic_factor2D;
	float cubic_factor3D;
	float mass;
	float time_delta;
	float eta;				// confine boundary loss coefficient
	float f_air;			// air_resistance

public:
	SPHSystem();
	~SPHSystem();

	float3* InitializePosition();
	float3* InitializeVelocity();
	float* InitializeDensity();
#ifdef DEBUG
	void Debug();
#endif // DEBUG
};
