#pragma once
#ifndef HANDLER_H
#define HANDLER_H

#include <iostream>
#include <cuda_runtime_api.h>
//#include <driver_types.h>

// check cuda error
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
inline void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		std::cerr << "CUDA ERROR = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		std::cerr << cudaGetErrorString(result) << "\n";
		cudaDeviceReset(); // Make sure we call CUDA Device Reset before exiting
		exit(99);
	}
}
//inline void ErrorHandler(char* err_msg) {
//	std::cerr << "ERROR: at " << __FILE__ << ":" << __LINE__ << ": " << err_msg << std::endl;
//}

#endif // !HANDLER_H

