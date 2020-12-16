#version 120

varying vec3 fs_PosEye;
varying mat4 u_Persp;

void main() {
	// calculate normal from texture coordinates
    vec3 N;

    N.xy = gl_PointCoord.xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);

    float mag = dot(N.xy, N.xy);
    if (mag > 1.0) discard;   // kill pixels outside circle
    N.z = sqrt(1.0-mag);

    gl_FragColor = vec4(exp(-mag*mag)*gl_Color.rgb,1.0f);
	//gl_FragColor = vec4(vec3(0.03f),1.0f);

	if (gl_Color.xyz == vec3(0,0,0)) {
    	gl_FragColor = vec4(0.0, 1.0, 1.0, 1.0);
    }
}