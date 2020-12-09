#pragma once

#ifdef __CUDACC__
#define __HOSTDEV__ __host__ __device__
#else
#define __HOSTDEV__
#endif

#include <cuda.h>
#include <curand_kernel.h>
#include <vector_types.h>
#include <helper_math.h>
#include <iostream>
#include <math.h>

// Constant
#define M_PI (3.1415926535)
#define M_EPS (1e-5)

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

inline __HOSTDEV__ bool operator > (const int3& A, const int3& B) { return (A.x > B.x && A.y > B.y && A.z > B.z); }
inline __HOSTDEV__ bool operator < (const int3& A, const int3& B) { return (A.x < B.x && A.y < B.y && A.z < B.z); }
inline __HOSTDEV__ bool operator >= (const int3& A, const int3& B) { return (A.x >= B.x && A.y >= B.y && A.z >= B.z); }
inline __HOSTDEV__ bool operator <= (const int3& A, const int3& B) { return (A.x <= B.x && A.y <= B.y&& A.z <= B.z); }


inline __HOSTDEV__ int3 float3TOint3(const float3& A) { return make_int3(A.x, A.y, A.z); }
inline __HOSTDEV__ float3 int3TOfloat3(const int3& A) { return make_float3(A.x, A.y, A.z); }

inline __HOSTDEV__ int Pow2(int A) { return A * A; }
inline __HOSTDEV__ float Pow2(float A) { return A * A; }
inline __HOSTDEV__ int3 Pow2(int3 A) { return A * A; }
inline __HOSTDEV__ float3 Pow2(float3 A) { return A * A; }

inline __HOSTDEV__ float Norm2(float3 A) { return sqrt(A.x * A.x + A.y * A.y + A.z * A.z); }
inline __HOSTDEV__ int GetIdx1D(int3 bIdx, const int3 bDim) { return bIdx.x * bDim.y * bDim.z + bIdx.y * bDim.z + bIdx.z; }
inline __HOSTDEV__ bool IdxIsValid(int3 bIdx, const int3 bDim) { return (bIdx >= int3({ 0, 0, 0 }) && bIdx < bDim); }


inline float3 operator * (const float3& A, const int3& B) { return make_float3(A.x * B.x, A.y * B.y, A.z * B.z); }
inline float3 operator / (const float3& A, const int3& B) { return make_float3(A.x / B.x, A.y / B.y, A.z / B.z); }

inline std::ostream& operator<<(std::ostream& out, const int3 A)
{
	out << "<int3> ( " << A.x << ", " << A.y << ", " << A.z << " )";
	return out;
}

inline std::ostream& operator<<(std::ostream& out, const float3 A)
{
	out << "<float3> ( " << A.x << ", " << A.y << ", " << A.z << " )";
	return out;
}


//int3 operator + (const int3& A, const int3& B) { return { A.x + B.x, A.y + B.y, A.z + B.z }; }
//int3 operator - (const int3& A, const int3& B) { return { A.x - B.x, A.y - B.y, A.z - B.z }; }
//int3 operator * (const int3& A, const int3& B) { return { A.x * B.x, A.y * B.y, A.z * B.z }; }
//int3 operator / (const int3& A, const int3& B) { return { A.x / B.x, A.y / B.y, A.z / B.z }; }
//
//int3 operator + (const int3& A, const int& B) { return { A.x + B, A.y + B, A.z + B }; }
//int3 operator - (const int3& A, const int& B) { return { A.x - B, A.y - B, A.z - B }; }
//int3 operator * (const int3& A, const int& B) { return { A.x * B, A.y * B, A.z * B }; }
//int3 operator / (const int3& A, const int& B) { return { A.x / B, A.y / B, A.z / B }; }
//
//float3 operator + (const float3& A, const float3& B) { return { A.x + B.x, A.y + B.y, A.z + B.z }; }
//float3 operator - (const float3& A, const float3& B) { return{ A.x - B.x, A.y - B.y, A.z - B.z }; }
//float3 operator * (const float3& A, const float3& B) { return { A.x * B.x, A.y * B.y, A.z * B.z }; }
//float3 operator / (const float3& A, const float3& B) { return { A.x / B.x, A.y / B.y, A.z / B.z }; }
//
//float3 operator + (const float3& A, const float& B) { return { A.x + B, A.y + B, A.z + B }; }
//float3 operator - (const float3& A, const float& B) { return { A.x - B, A.y - B, A.z - B }; }
//float3 operator * (const float3& A, const float& B) { return { A.x * B, A.y * B, A.z * B }; }
//float3 operator / (const float3& A, const float& B) { return { A.x / B, A.y / B, A.z / B }; }
//
//float3 operator * (const float3& A, const int3& B) { return { A.x * B.x, A.y * B.y, A.z * B.z }; }
//float3 operator / (const float3& A, const int3& B) { return { A.x / B.x, A.y / B.y, A.z / B.z }; }
//
