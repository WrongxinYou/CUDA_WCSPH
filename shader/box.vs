#version 150

in vec3 pos;
out vec4 col;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0f);
	col = vec4 (1.0, 1.0, 1.0, 1.0);

	if (pos == vec3(1.0, 0.0, 0.0))
	{
		col = vec4(1.0, 0.0, 0.0, 1.0);
	}
	if (pos == vec3(0.0, 1.0, 0.0))
	{
		col = vec4(0.0, 1.0, 0.0, 1.0);
	}
	if (pos == vec3(0.0, 0.0, 1.0))
	{
		col = vec4(0.0, 0.0, 1.0, 1.0);
	}
	if (pos == vec3(1.0, 1.0, 0.0))
	{
		col = vec4(1.0, 1.0, 0.0, 1.0);
	}
	if (pos == vec3(0.0, 1.0, 1.0))
	{
		col = vec4(0.0, 1.0, 1.0, 1.0);
	}
	if (pos == vec3(1.0, 0.0, 1.0))
	{
		col = vec4(1.0, 0.0, 1.0, 1.0);
	}
	if (pos == vec3(1.0, 1.0, 1.0))
	{
		col = vec4(0.0, 0.0, 0.0, 0.0);
	}
}