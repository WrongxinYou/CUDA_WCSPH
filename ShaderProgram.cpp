#include "ShaderProgram.h"
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

void ShaderProgram::Init(const char* vertexShaderPath, const char* fragShaderPath) {

	string vertexCode, fragCode;
	ifstream vertexFile, fragFile;

	try {
		vertexFile.open(vertexShaderPath);
		fragFile.open(fragShaderPath);

		stringstream vertexStream, fragStream;
		vertexStream << vertexFile.rdbuf();
		fragStream << fragFile.rdbuf();

		vertexFile.close();
		fragFile.close();

		vertexCode = vertexStream.str();
		fragCode = fragStream.str();
	}

	catch (ifstream::failure& e) {
		cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << endl;
	}

	const char* vCode = vertexCode.c_str();
	const char* fCode = fragCode.c_str();

	unsigned int vShader, fShader;
	vShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vShader, 1, &vCode, NULL);
	glCompileShader(vShader);
	CheckCompileErrors(vShader, VERTEXSHADER);


	// fragment shader
	fShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fShader, 1, &fCode, NULL);
	glCompileShader(fShader);
	CheckCompileErrors(fShader, FRAGSHADER);


	// shader program
	programHandle = glCreateProgram();
	glAttachShader(programHandle, vShader);
	glAttachShader(programHandle, fShader);
	glLinkProgram(programHandle);
	CheckCompileErrors(programHandle, PROGRAM);

	glDeleteShader(vShader);
	glDeleteShader(fShader);

	modelViewHandle = glGetUniformLocation(programHandle, "modelViewMatrix");
	projectionHandle = glGetUniformLocation(programHandle, "projectionMatrix");
}

GLuint ShaderProgram::GetProgrameHandle() {
	return programHandle;
}

void ShaderProgram::Bind() {
	glUseProgram(programHandle);
}

void ShaderProgram::SetModelViewMatrix(const float* mv) {
	glUniformMatrix4fv(modelViewHandle, 1, GL_FALSE, mv);
}

void ShaderProgram::SetProjectionMatrix(const float* p) {
	glUniformMatrix4fv(projectionHandle, 1, GL_FALSE, p);
}

void ShaderProgram::CheckCompileErrors(GLuint handle, CheckType type) {
	GLint success;
	char infoLog[512];

	switch (type) {
	case VERTEXSHADER: {
		glGetShaderiv(handle, GL_COMPILE_STATUS, &success);
		if (!success) {
			glGetShaderInfoLog(handle, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
		}
		break;
	}

	case FRAGSHADER: {
		glGetShaderiv(handle, GL_COMPILE_STATUS, &success);
		if (!success) {
			glGetShaderInfoLog(handle, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
		}
		break;
	}

	case PROGRAM: {
		glGetProgramiv(handle, GL_LINK_STATUS, &success);
		if (!success) {
			glGetProgramInfoLog(handle, 512, NULL, infoLog);
			std::cout << "ERROR::PROGRAM_LINKING_ERROR\n" << infoLog << std::endl;
		}
		break;
	}
	}


}