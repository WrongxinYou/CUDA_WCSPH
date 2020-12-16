#version 120

in vec3 pos;

varying vec3 fs_PosEye;
varying mat4 u_Persp;

uniform float pointRadius;
uniform float pointScale;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

void main() {

	vec3 posEye = (gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0f)).xyz;
	float dist = length(posEye);
	gl_PointSize = pointRadius * pointScale / dist;
	fs_PosEye = posEye;

	gl_FrontColor = gl_Color;
	u_Persp = gl_ProjectionMatrix;

	gl_Position = ftransform();
}
