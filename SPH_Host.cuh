#pragma once
#include "SPHSystem.h"
// Host Parameters

// Device Parameters
//static SPHSystem* sys_device;


void getNextFrame(float3* particle_pos, float3* particle_color, SPHSystem* sys);
void InitDeviceSystem(SPHSystem* sys_host);
void FreeDeviceSystem(SPHSystem* sys_host);