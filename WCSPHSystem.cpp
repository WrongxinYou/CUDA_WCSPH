#include "WCSPHSystem.h"
#include <fstream>
#include <string.h>
#include "utils/json.hpp"
using json = nlohmann::json;

////////////////////////////////////////////////////////////////////////////////
//
// Default construct function
//
////////////////////////////////////////////////////////////////////////////////
WCSPHSystem::WCSPHSystem() {

	// SPH Parameters
	dim = 3;
	particle_dim = make_int3(4, 4, 4);
	particle_num = particle_dim.x * particle_dim.y * particle_dim.z;
	particle_radius = 0.1;
	velo_init_min = make_float3(0, -1, 0);
	velo_init_max = make_float3(0.1, -1, 0.1);
	config_filename = "";

	// Draw Parameters
	step_each_frame = 5;
	box_size = make_float3(1.0, 1.0, 1.0);
	box_margin = box_size * 0.1;

	// Device Parameters
	block_dim = make_int3(3, 3, 3);
	block_num = block_dim.x * block_dim.y * block_dim.z;
	block_size = box_size / block_dim; // block_dim <= 3/(4 * cutoff) = 3/(4 * 1.3 radius)
	block_thread_num = 256;

	// Function Parameters
	rho_0 = 1000.0;  // reference density
	gamma = 7.0;
	h = 1.3 * particle_radius;
	gravity = -9.8 * 30;
	alpha = 0.3; // between 0.08 and 0.5
	C_s = 200; // sqrt((2 * gravity * Height of particle + pow(initial velocity, 2)) / 0.01) 
	CFL_v = 0.20;
	CFL_a = 0.20;
	poly6_factor = 315.0 / 64.0 / M_PI;
	spiky_grad_factor = -45.0 / M_PI;
	cubic_factor1D = 2.0 / 3.0 / M_PI;
	cubic_factor2D = 10.0 / 7.0 / M_PI;
	cubic_factor3D = 1.0 / M_PI;
	mass = 4.0 / 3.0 * M_PI * rho_0 * pow(particle_radius, dim);
	time_delta = 0.1 * h / C_s; // 0.4 * h / C_s
	eta = 0.8; // confine boundary loss coefficient
	f_air = 0.001; // air_resistance
}


WCSPHSystem::~WCSPHSystem() {}


////////////////////////////////////////////////////////////////////////////////
//
// construct function from config file
// 
////////////////////////////////////////////////////////////////////////////////

void ConstructFromJson(WCSPHSystem* sys, json config);
void ConstructFromJsonFile(WCSPHSystem* sys, const char* filename) {
	std::ifstream fin(filename);
	json config;
	fin >> config;
	fin.close();
	ConstructFromJson(sys, config);

#ifdef OUTPUT_CONFIG
	std::ofstream fout("tmp.json");
	fout << config << std::endl;
	fout.close();
	std::cout << config << std::endl;
#endif // OUTPUT_CONFIG
}

WCSPHSystem::WCSPHSystem(char* filename = "WCSPH_config.json") {
	char* dot = strrchr(filename, '.');
	if (!dot || dot == filename) {
		std::cerr << "Construct from file error: Unknown config file type" << std::endl;
	}
	else if (strcmp(dot + 1, "json") == 0) {
		config_filename = filename;
		ConstructFromJsonFile(this, filename);
	}
	else {
		std::cerr << "Construct from file error: config file type error" << std::endl;
	}
}

void ConstructFromJson(WCSPHSystem* sys, json config) {
	// SPH Parameters
	{
		auto tmp = config["dim"];
		int x = int(tmp);
		sys->dim = x;
	}
	{
		auto tmp = config["particle_dim"];
		int x = int(tmp[0]);
		int y = int(tmp[1]);
		int z = int(tmp[2]);
		sys->particle_dim = make_int3(x, y, z);
	}
	sys->particle_num = sys->particle_dim.x * sys->particle_dim.y * sys->particle_dim.z;
	{
		auto tmp = config["particle_radius"];
		float x = float(tmp);
		sys->particle_radius = x;
	}

	// Draw Parameters
	{
		auto tmp = config["step_each_frame"];
		int x = int(tmp);
		sys->step_each_frame = x;
	}
	{
		auto tmp = config["box_size"];
		float x = float(tmp[0]);
		float y = float(tmp[1]);
		float z = float(tmp[2]);
		sys->box_size = make_float3(x, y, z);
	}
	{
		auto tmp = config["box_margin_coefficient"];
		float x = float(tmp[0]);
		float y = float(tmp[1]);
		float z = float(tmp[2]);
		sys->box_margin = sys->box_size * make_float3(x, y, z);
	}

	// Device Parameters
	{
		auto tmp = config["block_dim"];
		int x = int(tmp[0]);
		int y = int(tmp[1]);
		int z = int(tmp[2]);
		sys->block_dim = make_int3(x, y, z);
	}
	sys->block_num = sys->block_dim.x * sys->block_dim.y * sys->block_dim.z;
	sys->block_size = sys->box_size / sys->block_dim;
	{
		auto tmp = config["block_thread_num"];
		int x = int(tmp);
		sys->block_thread_num = x;
	}


	// Function Parameters
	{
		auto tmp = config["rho_0"];
		float x = float(tmp);
		sys->rho_0 = x;  // reference density
	}
	{
		auto tmp = config["gamma"];
		float x = float(tmp);
		sys->gamma = x;
	}
	{
		auto tmp = config["h_coefficient"];
		float x = float(tmp);
		sys->h = sys->particle_radius * x;
	}
	{
		auto tmp = config["gravity"];
		auto tmp1 = config["gravity_coefficient"];
		float x = float(tmp);
		float y = float(tmp1);
		sys->gravity = x * y;
	}
	{
		auto tmp = config["alpha"];
		float x = float(tmp);
		sys->alpha = x;
	}
	{
		auto tmp = config["C_s"];
		float x = float(tmp);
		sys->C_s = x;
	}
	{
		auto tmp = config["CFL_v"];
		float x = float(tmp);
		sys->CFL_v = x;
	}
	{
		auto tmp = config["CFL_a"];
		float x = float(tmp);
		sys->CFL_a = x;
	}
	{
		auto tmp = config["poly6_factor_coefficient1"];
		auto tmp1 = config["poly6_factor_coefficient2"];
		float x = float(tmp);
		float y = float(tmp1);
		sys->poly6_factor = x / y / M_PI;
	}
	{
		auto tmp = config["spiky_grad_factor_coefficient"];
		float x = float(tmp);
		sys->spiky_grad_factor = x / M_PI;
	}
	{
		float x = config["cubic_factor_coefficient1D1"];
		float y = config["cubic_factor_coefficient1D2"];
		sys->cubic_factor1D = x / y / M_PI;
	}
	{
		float x = config["cubic_factor_coefficient2D1"];
		float y = config["cubic_factor_coefficient2D2"];
		sys->cubic_factor2D = x / y / M_PI;
	}
	{
		float x = config["cubic_factor_coefficient3D1"];
		float y = config["cubic_factor_coefficient3D2"];
		sys->cubic_factor3D = x / y / M_PI;
	}
	sys->mass = 4.0 / 3.0 * M_PI * sys->rho_0 * pow(sys->particle_radius, sys->dim);
	{
		auto tmp = config["time_delta_coefficient"];
		float x = float(tmp);
		sys->time_delta = x * sys->h / sys->C_s;
	}
	{
		auto tmp = config["velo_init_max"];
		float x = float(tmp[0]);
		float y = float(tmp[1]);
		float z = float(tmp[2]);
		sys->velo_init_max = make_float3(x, y, z);
	}
	{
		auto tmp = config["velo_init_min"];
		float x = float(tmp[0]);
		float y = float(tmp[1]);
		float z = float(tmp[2]);
		sys->velo_init_min = make_float3(x, y, z);
	}
	{
		auto tmp = config["eta"];
		float x = float(tmp);
		sys->eta = x;
	}
	{
		auto tmp = config["f_air"];
		float x = float(tmp);
		sys->f_air = x;
	}
}


////////////////////////////////////////////////////////////////////////////////
//
// Initialize particles density
//
////////////////////////////////////////////////////////////////////////////////
float* WCSPHSystem::InitializeDensity() {
	float* dens_init = new float[particle_num];
	for (int i = 0; i < particle_num; i++) {
		dens_init[i] = rho_0;
	}
	return dens_init;
}

////////////////////////////////////////////////////////////////////////////////
//
// Initialize particles position
//
////////////////////////////////////////////////////////////////////////////////
float3* WCSPHSystem::InitializePosition() {
	float3* pos_init = new float3[particle_num];
	float3 particle_gap = box_size - box_margin - box_margin;
	particle_gap = particle_gap / (particle_dim + 1);

	for (int i = 0; i < particle_dim.x; i++)
	{
		for (int j = 0; j < particle_dim.y; j++)
		{
			for (int k = 0; k < particle_dim.z; k++)
			{
				int3 ii = make_int3(i, j, k);
				int index = GetBlockIdx1D(ii, particle_dim);
				float3 p = particle_gap * (ii + 1);
				pos_init[index] = box_margin + p;
			}
		}
	}
	return pos_init;
}

////////////////////////////////////////////////////////////////////////////////
//
// Initialize particles density
//
////////////////////////////////////////////////////////////////////////////////
float3* WCSPHSystem::InitializeVelocity() {
	float3* velo_init = new float3[particle_num];
	for (int i = 0; i < particle_num; i++) {
		velo_init[i] = M_EPS + make_float3(
			RandomFloat(velo_init_min.x, velo_init_max.x),
			RandomFloat(velo_init_min.y, velo_init_max.y),
			RandomFloat(velo_init_min.z, velo_init_max.z));
	}
	return velo_init;
}

