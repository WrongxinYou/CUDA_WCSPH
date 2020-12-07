#pragma once

#include <cuda.h>
#include <curand_kernel.h>
#include <iostream>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
inline void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		std::cerr << cudaGetErrorString(result) << "\n";
		cudaDeviceReset(); // Make sure we call CUDA Device Reset before exiting
		exit(99);
	}
}

//#define CUDA_CHECK(x) do { if((x)!=cudaSuccess) { printf("CUDA Error at %s:%d\t Error code = %d\n",__FILE__,__LINE__,x);}} while(0)
