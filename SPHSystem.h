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
	float mass;
	float time_delta;


public:
	SPHSystem();
	~SPHSystem();

	float3* InitializePosition();
#ifdef DEBUG
	void Debug();
#endif // DEBUG
};
