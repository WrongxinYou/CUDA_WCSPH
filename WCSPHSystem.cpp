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

	// WCSPH System Parameters
	config_filename = "";
	dim = 3;
	particle_dim = make_int3(4, 4, 4);
	particle_num = particle_dim.x * particle_dim.y * particle_dim.z;
	gravity = -9.8 * 30;
	eta = 0.8; // confine boundary loss coefficient
	f_air = 0.001; // air_resistance

	// Particles Parameters
	particle_radius = 0.1;
	mass = 4.0 / 3.0 * M_PI * rho_0 * pow(particle_radius, dim);
	velo_init_min = make_float3(0, -1, 0);
	velo_init_max = make_float3(0.1, -1, 0.1);
	velo_draw_min = 0.0;
	velo_draw_max = 20.0;

	// Draw Parameters
	step_each_frame = 5;
	box_length = make_float3(1.0, 1.0, 1.0);
	box_margin = 0.1 * box_length;

	// Device Parameters
	grid_dim = dim3(3, 3, 3);
	grid_size = grid_dim.x * grid_dim.y * grid_dim.z;
	block_length = box_length / grid_dim; // grid_dim <= 3/(4 * cutoff) = 3/(4 * 1.3 radius)
	block_size = 256;

	// Function Parameters
	alpha = 0.3; // between 0.08 and 0.5
	C_s = 200; // sqrt((2 * gravity * Height of particle + pow(initial velocity, 2)) / 0.01) 
	gamma = 7.0;
	h = 1.3 * particle_radius;
	rho_0 = 1000.0;  // reference density
	CFL_a = 0.20;
	CFL_v = 0.20;
	poly6_factor = 315.0 / 64.0 / M_PI;
	spiky_grad_factor = -45.0 / M_PI;
	vis_lapla_factor = 45.0 / M_PI;
	cubic_factor1D = 2.0 / 3.0 / M_PI;
	cubic_factor2D = 10.0 / 7.0 / M_PI;
	cubic_factor3D = 1.0 / M_PI;
	time_delta = 0.1 * h / C_s; // 0.4 * h / C_s
}


WCSPHSystem::~WCSPHSystem() {}


////////////////////////////////////////////////////////////////////////////////
//
// Construct function from config file
// 
////////////////////////////////////////////////////////////////////////////////

float3 GetJsonVectorValue(const json& config, const char* name) {
	auto tmp = config[name];
	if (tmp.is_array()) {
		return make_float3(tmp[0], tmp[1], tmp[2]);
	}
	else {
		std::cerr << "GetJsonVectorValue ERROR! At " << name << std::endl;
		return make_float3(-1, -1, -1);
	}
}

void ConstructFromJson(WCSPHSystem* sys, json config) {

	// WCSPH System Parameters
	sys->dim = config["dim"].get<int>();
	sys->particle_dim = make_int3(GetJsonVectorValue(config, "particle_dim"));
	sys->particle_num = sys->particle_dim.x * sys->particle_dim.y * sys->particle_dim.z;
	sys->eta = config["eta"].get<float>();
	sys->f_air = config["f_air"].get<float>();
	sys->gravity = config["gravity"].get<float>() * config["gravity_coefficient"].get<float>();

	// Particles Parameters
	sys->particle_radius = config["particle_radius"].get<float>();
	sys->velo_init_min = GetJsonVectorValue(config, "velo_init_min");
	sys->velo_init_max = GetJsonVectorValue(config, "velo_init_max");

	// Draw Parameters
	sys->step_each_frame = config["step_each_frame"].get<int>();
	sys->box_length = GetJsonVectorValue(config, "box_length");
	sys->box_margin = GetJsonVectorValue(config, "box_margin_coefficient") * sys->box_length;
	sys->velo_draw_min = config["velo_draw_min"].get<float>();
	sys->velo_draw_max = config["velo_draw_max"].get<float>();

	// Device Parameters
	sys->grid_dim = dim3(make_uint3(make_int3(GetJsonVectorValue(config, "grid_dim"))));
	sys->grid_size = GetDimTotalSize(sys->grid_dim);
	sys->block_length = sys->box_length / sys->grid_dim;
	sys->block_size = config["block_size"].get<uint>();

	// Function Parameters
	sys->alpha = config["alpha"].get<float>();
	sys->gamma = config["gamma"].get<float>();
	sys->h = config["h_coefficient"].get<float>() * sys->particle_radius;
	sys->rho_0 = config["rho_0"].get<float>();
	sys->CFL_a = config["CFL_a"].get<float>();
	sys->CFL_v = config["CFL_v"].get<float>();
	sys->poly6_factor = config["poly6_factor_coefficient1"].get<float>() / config["poly6_factor_coefficient2"].get<float>() / M_PI;
	sys->spiky_grad_factor = config["spiky_grad_factor_coefficient"].get<float>() / M_PI;
	sys->vis_lapla_factor = config["vis_lapla_factor_coefficient"].get<float>() / M_PI;
	sys->cubic_factor1D = config["cubic_factor_coefficient1D1"].get<float>() / config["cubic_factor_coefficient1D2"].get<float>() / M_PI;
	sys->cubic_factor2D = config["cubic_factor_coefficient2D1"].get<float>() / config["cubic_factor_coefficient2D2"].get<float>() / M_PI;
	sys->cubic_factor3D = config["cubic_factor_coefficient3D1"].get<float>() / config["cubic_factor_coefficient3D2"].get<float>() / M_PI;

	// calculated para
	sys->mass = 4.0 / 3.0 * M_PI * sys->rho_0 * pow(sys->particle_radius, sys->dim);
	sys->C_s = sqrt((2 * fabs(sys->gravity) * sys->box_length.y + pow(sys->velo_init_min.y, 2)) / 0.01);
	sys->time_delta = config["time_delta_coefficient"].get<float>() * sys->h / sys->C_s;

}

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

WCSPHSystem::WCSPHSystem(char* filename = "config/WCSPH_config.json") {
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

////////////////////////////////////////////////////////////////////////////////
//
// Print settings to console
// 
////////////////////////////////////////////////////////////////////////////////
void WCSPHSystem::Print() {
	std::cout << "=============== WCSPH System Settings ================" << std::endl;
	// WCSPH System Parameters
	std::cout << "WCSPH System Parameters" << std::endl;
	std::cout << "\t config_filename: " << config_filename << std::endl;
	std::cout << "\t dim: " << dim << std::endl;
	std::cout << "\t particle_dim: " << particle_dim << std::endl;
	std::cout << "\t particle_num: " << particle_num << std::endl;
	std::cout << "\t eta: " << eta << std::endl;
	std::cout << "\t f_air: " << f_air << std::endl;
	std::cout << "\t gravity: " << gravity << std::endl;

	// Particles Parameters
	std::cout << "Particles Parameters" << std::endl;
	std::cout << "\t particle_radius: " << particle_radius << std::endl;
	std::cout << "\t mass: " << mass << std::endl;
	std::cout << "\t velo_init_min: " << velo_init_min << std::endl;
	std::cout << "\t velo_init_max: " << velo_init_max << std::endl;

	// Draw Parameters
	std::cout << "Draw Parameters" << std::endl;
	std::cout << "\t step_each_frame: " << step_each_frame << std::endl;
	std::cout << "\t box_length: " << box_length << std::endl;
	std::cout << "\t box_margin: " << box_margin << std::endl;
	std::cout << "\t velo_draw_min: " << velo_draw_min << std::endl;
	std::cout << "\t velo_draw_max: " << velo_draw_max << std::endl;


	// Device Parameters
	std::cout << "Device Parameters" << std::endl;
	std::cout << "\t grid_dim: " << grid_dim << std::endl;
	std::cout << "\t grid_size: " << grid_size << std::endl;
	std::cout << "\t block_length: " << block_length << std::endl;
	std::cout << "\t block_size: " << block_size << std::endl;

	// Function Parameters
	std::cout << "Function Parameters" << std::endl;
	std::cout << "\t alpha: " << alpha << std::endl;
	std::cout << "\t C_s: " << C_s << std::endl;
	std::cout << "\t gamma: " << gamma << std::endl;
	std::cout << "\t h: " << h << std::endl;
	std::cout << "\t rho_0: " << rho_0 << std::endl;
	std::cout << "\t CFL_a: " << CFL_a << std::endl;
	std::cout << "\t CFL_v: " << CFL_v << std::endl;
	std::cout << "\t poly6_factor: " << poly6_factor << std::endl;
	std::cout << "\t spiky_grad_factor: " << spiky_grad_factor << std::endl;
	std::cout << "\t vis_lapla_factor: " << vis_lapla_factor << std::endl;
	std::cout << "\t cubic_factor1D: " << cubic_factor1D << std::endl;
	std::cout << "\t cubic_factor2D: " << cubic_factor2D << std::endl;
	std::cout << "\t cubic_factor3D: " << cubic_factor3D << std::endl;
	std::cout << "\t time_delta: " << time_delta << std::endl;
	
	std::cout << "======================= END ==========================" << std::endl;
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
	float3 particle_gap = box_length - box_margin - box_margin;
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

