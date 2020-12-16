#version 150

in vec3 pos;
out vec4 col;

uniform vec3 box_length;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

float bx = box_length.x;
float by = box_length.y;
float bz = box_length.z;

void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0f);
	col = vec4 (1.0, 1.0, 1.0, 1.0);

	if (pos == vec3(bx, 0.0, 0.0) - box_length/2)
	{
		col = vec4(1.0, 0.0, 0.0, 1.0);
	}
	if (pos == vec3(0.0, by, 0.0) - box_length/2)
	{
		col = vec4(0.0, 1.0, 0.0, 1.0);
	}
	if (pos == vec3(0.0, 0.0, bz) - box_length/2)
	{
		col = vec4(0.0, 0.0, 1.0, 1.0);
	}
	if (pos == vec3(bx, by, 0.0) - box_length/2)
	{
		col = vec4(1.0, 1.0, 0.0, 1.0);
	}
	if (pos == vec3(0.0, by, bz) - box_length/2)
	{
		col = vec4(0.0, 1.0, 1.0, 1.0);
	}
	if (pos == vec3(bx, 0.0, bz) - box_length/2)
	{
		col = vec4(1.0, 0.0, 1.0, 1.0);
	}
	if (pos == vec3(bx, by, bz) - box_length/2)
	{
		col = vec4(0.0, 0.0, 0.0, 0.0);
	}
}