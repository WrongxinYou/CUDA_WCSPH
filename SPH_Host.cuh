#pragma once
#include "SPHSystem.h"
// Host Parameters

// Device Parameters

void getFirstFrame(SPHSystem* para, cudaGraphicsResource* position_resource, cudaGraphicsResource* color_resource);
void getNextFrame(SPHSystem* sys, cudaGraphicsResource* position_resource, cudaGraphicsResource* color_resource);
void InitDeviceSystem(SPHSystem* para, float3* pos_init, float3* velo_init);
void FreeDeviceSystem(SPHSystem* para);
