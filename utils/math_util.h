#pragma once
#ifndef MATH_UTIL_H
#define MATH_UTIL_H

#ifdef __CUDACC__
#define __HOSTDEV__ __host__ __device__
#else
#define __HOSTDEV__
#endif

#include <iostream>
#include <vector_types.h>
#include <helper_math.h>

// constant
#define M_PI (3.1415926535)
#define M_EPS (1e-6)

// operator overload
// multiply
inline __HOSTDEV__ float3 operator * (const float3& A, const int3& B) { return (A * make_float3(B)); }
inline __HOSTDEV__ float3 operator * (const int3& A, const float3& B) { return (make_float3(A) * B); }

// divide
inline __HOSTDEV__ float3 operator / (const float3& A, const int3& B) { return (A / make_float3(B)); }
inline __HOSTDEV__ float3 operator / (const int3& A, const float3& B) { return (make_float3(A) / B); }

// compare
inline __HOSTDEV__ bool operator > (const int3& A, const int3& B) { return (A.x > B.x && A.y > B.y && A.z > B.z); }
inline __HOSTDEV__ bool operator < (const int3& A, const int3& B) { return (A.x < B.x&& A.y < B.y&& A.z < B.z); }
inline __HOSTDEV__ bool operator >= (const int3& A, const int3& B) { return (A.x >= B.x && A.y >= B.y && A.z >= B.z); }
inline __HOSTDEV__ bool operator <= (const int3& A, const int3& B) { return (A.x <= B.x && A.y <= B.y && A.z <= B.z); }
inline __HOSTDEV__ bool operator == (const int3& A, const int3& B) { return (A.x == B.x && A.y == B.y && A.z == B.z); }

// output stream
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


// norm2 of vector
inline __HOSTDEV__ double Norm2(float3 A) { return sqrt(dot(A, A)); }

// block index transfer from 3D to 1D
inline __HOSTDEV__ int GetBlockIdx1D(int3 block_idx, const int3 block_dim) { return block_idx.x * block_dim.y * block_dim.z + block_idx.y * block_dim.z + block_idx.z; }

// block index check 
inline __HOSTDEV__ bool BlockIdxIsValid(int3 block_idx, const int3 block_dim) { return (block_idx >= make_int3(0, 0, 0) && block_idx < block_dim); }

// random float [a, b]
inline float RandomFloat(float a, float b) {
	float random = ((float)rand()) / (float)RAND_MAX;
	float diff = b - a;
	float r = random * diff;
	return a + r;
}


#endif // !MATH_UTIL_H
