#pragma once
#include <iostream>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <vector_types.h>

enum CheckType
{
	PROGRAM,
	VERTEXSHADER,
	FRAGSHADER
};

class ShaderProgram
{

private:
	GLuint programHandle;
	GLint modelViewHandle, projectionHandle;

public:
	ShaderProgram() { programHandle = 0; }
	~ShaderProgram() { glDeleteProgram(programHandle); }

	void Init(const char* vertexShaderPath, const char* fragShaderPath);
	void SetModelViewMatrix(const float* mv);
	void SetProjectionMatrix(const float* p);
	void SetFloat(const std::string& name, float value);
	void ShaderProgram::SetFloat3(const std::string& name, float3 value);
	void Bind();
	GLuint GetProgrameHandle();

private:
	void CheckCompileErrors(GLuint shader, CheckType type);
};

