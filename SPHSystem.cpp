#include "SPHSystem.h"
#include <iostream>

int3 operator + (const int3& A, const int3& B) { return { A.x + B.x, A.y + B.y, A.z + B.z }; }
int3 operator - (const int3& A, const int3& B) { return { A.x - B.x, A.y - B.y, A.z - B.z }; }
int3 operator * (const int3& A, const int3& B) { return { A.x * B.x, A.y * B.y, A.z * B.z }; }
int3 operator / (const int3& A, const int3& B) { return { A.x / B.x, A.y / B.y, A.z / B.z }; }

int3 operator + (const int3& A, const int& B) { return { A.x + B, A.y + B, A.z + B }; }
int3 operator - (const int3& A, const int& B) { return { A.x - B, A.y - B, A.z - B }; }
int3 operator * (const int3& A, const int& B) { return { A.x * B, A.y * B, A.z * B }; }
int3 operator / (const int3& A, const int& B) { return { A.x / B, A.y / B, A.z / B }; }

float3 operator + (const float3& A, const float3& B) { return { A.x + B.x, A.y + B.y, A.z + B.z }; }
float3 operator - (const float3& A, const float3& B) { return{ A.x - B.x, A.y - B.y, A.z - B.z }; }
float3 operator * (const float3& A, const float3& B) { return { A.x * B.x, A.y * B.y, A.z * B.z }; }
float3 operator / (const float3& A, const float3& B) { return { A.x / B.x, A.y / B.y, A.z / B.z }; }

float3 operator + (const float3& A, const float& B) { return { A.x + B, A.y + B, A.z + B }; }
float3 operator - (const float3& A, const float& B) { return { A.x - B, A.y - B, A.z - B }; }
float3 operator * (const float3& A, const float& B) { return { A.x * B, A.y * B, A.z * B }; }
float3 operator / (const float3& A, const float& B) { return { A.x / B, A.y / B, A.z / B }; }

float3 operator * (const float3& A, const int3& B) { return { A.x * B.x, A.y * B.y, A.z * B.z }; }
float3 operator / (const float3& A, const int3& B) { return { A.x / B.x, A.y / B.y, A.z / B.z }; }

std::ostream& operator<<(std::ostream& out, const float3 A)
{
	out << "<float3> ( " << A.x << ", " << A.y << ", " << A.z << " )";
	return out;
}

// SPHSystem::SPHSystem() {}

SPHSystem::SPHSystem()
{
	particle_radius = 1e-3;
	box_size = { 1, 1 ,1 };
	box_margin = { 0.1, 0.1, 0.1 };

	particle_num = 27;
	particle_dim = { 3, 3, 3 };

	particle_gid = NULL;
	grid_pid = NULL;
	grid_pnum = NULL;

	color = NULL;
	pos_host = NULL;
	cur_pos = NULL;
	next_pos = NULL;
	density = NULL;
	velocity = NULL;
	pressure = NULL;
}

SPHSystem::~SPHSystem()
{
	delete particle_gid;
	delete grid_pid;
	delete grid_pnum;

	delete pos_host;
	delete color;
	delete cur_pos;
	delete next_pos;
	delete density;
	delete velocity;
	delete pressure;
}


// Initialize particles position
void SPHSystem::Initialize()
{
	pos_host = new float3[particle_num];

	std::cout << "float3:  " << sizeof(float3) << std::endl;
	std::cout << "*float3:  " << sizeof(float3*) << std::endl;
	std::cout << "int3:  " << sizeof(int3) << std::endl;
	std::cout << "*int3:  " << sizeof(int3*) << std::endl;
	std::cout << "float:  " << sizeof(float) << std::endl;
	std::cout << "*float:  " << sizeof(float*) << std::endl;
	std::cout << "int:  " << sizeof(int) << std::endl;
	std::cout << "*int:  " << sizeof(int*) << std::endl;

	float3 gap = box_size - box_margin - box_margin;
	gap = gap / (particle_dim + 1);

	for (int i = 0; i < particle_dim.x; i++)
	{
		for (int j = 0; j < particle_dim.y; j++)
		{
			for (int k = 0; k < particle_dim.z; k++)
			{
				int index = i * particle_dim.y * particle_dim.z + j * particle_dim.z + k;
				float3 p = gap * int3{ i + 1, j + 1, k + 1 };
				pos_host[index] = box_margin + p;
			}
		}
	}

	for (int i = 0; i < particle_num; i++)
		std::cout << pos_host[i] << std::endl;

}
