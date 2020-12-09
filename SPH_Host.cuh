#pragma once
#include "SPHSystem.h"
// Host Parameters

// Device Parameters

void getNextFrame(SPHSystem* sys, cudaGraphicsResource* position_resource, cudaGraphicsResource* color_resource);
void InitDeviceSystem(SPHSystem* para, float3* pos_init);
void FreeDeviceSystem(SPHSystem* para);
