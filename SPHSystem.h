#pragma once
#include <vector_types.h>

class SPHSystem
{
public:

	// Common
	int particle_num;
	float particle_radius;
	float epsilon = 1e-5;

	float3 grid_size;
	int3 grid_num;

	float3 box_size;
	float3 box_margin;

	// Host
	int3 particle_dim; // number of particles in x, y, z axis to intialize
	float3* pos_host; // initial position in Host

	// Device 
	int* particle_gid;
	int* grid_pid; // first particle index in grid
	int* grid_pnum; // particle number in grid

	float3* color; // color of particles
	float3* cur_pos;
	float3* next_pos;
	float3* density;
	float3* velocity;
	float3* pressure;


public:
	SPHSystem();
	~SPHSystem();

	// Initialize particles position
	void Initialize();


};

