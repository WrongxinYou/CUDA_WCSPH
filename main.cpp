
#include "WCSPHSystem.h"
#include "WCSPHSolver.cuh"
#include "ShaderProgram.h"
#include "utils/handler.h"

#include <gl/GL.h>
#include <glm/glm.hpp>
#include <glm/common.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <time.h>

#define OUTPUT_FRAME_NUM

typedef enum { ROTATE, TRANSLATE, SCALE } CONTROL_STATE;

///////////////////////////////////////////////////////////////////////////////
// Dispaly Settings
///////////////////////////////////////////////////////////////////////////////
const int kWindowSize[2] = { 768, 768 };

int screen_size[2] = { 0, 0 };
int window_size[2] = { 0, 0 };

int fov = 45;
CONTROL_STATE control_state = TRANSLATE;
int mouse_pos[2];
bool left_mouse_button = false;
bool middle_mouse_button = false;
bool right_mouse_button = false;
bool interrupt = false;
bool is_one_frame = true;
long frame_cnt = 0;

// state of the world
float land_rotate[3]	=	{ 0.00f, 0.50f, 0.50f };
float land_translate[3] =	{ 0.00f, 0.00f, 0.00f };
float land_scale[3]		=	{ 0.75f, 0.75f, 0.75f };

// gl buffers
ShaderProgram box_program, particle_program;
unsigned int box_vbo, box_ebo, box_vao;
unsigned int position_vbo, color_vbo;

// gl functions
void initScene(WCSPHSystem* sys);
void drawParticles();
void displayFunc();
void idleFunc(); 
void closeWindow();
void mouseMotionDragFunc(int x, int y);
void mouseMotionFunc(int x, int y);
void mouseButtonFunc(int button, int state, int x, int y);
void reshapeFunc(int w, int h);
void keyboardFunc(unsigned char key, int x, int y);

// construct function
void initFluidSystem();

// box and board
float box_length = 1.0;
float board_pos = 0.5f;


////////////////////////////////////////////////////////////////////////////////
// SPH Settings
////////////////////////////////////////////////////////////////////////////////
cudaGraphicsResource* cuda_position_resource;
cudaGraphicsResource* cuda_color_resource;

WCSPHSystem* sph_host;
float3* pos_init;
float3* velo_init;
float* dens_init;

void initFluidSystem() {
	srand(time(0));

	sph_host = new WCSPHSystem("config/WCSPH_config.json");
	sph_host->Print();

	int dim_max = ceil(3 / (4 * sph_host->h));
	if (!(sph_host->grid_dim <= dim3(dim_max, dim_max, dim_max))) {
		std::cout << "WARNING: block_dimension is too large, please decrease it" << std::endl;
	}

	pos_init = sph_host->InitializePosition();
	velo_init = sph_host->InitializeVelocity();
	dens_init = sph_host->InitializeDensity();
	InitDeviceSystem(sph_host, dens_init, pos_init, velo_init);
}


int main(int argc, char* argv[]) {
	
	// Initialize CUDA
	std::cout << "Initializing CUDA..." << std::endl;

	int device_cnt, device_num = 0;
	checkCudaErrors(cudaGetDeviceCount(&device_cnt));
	std::cout << "Total CUDA Device number found : " << device_cnt << std::endl;

	checkCudaErrors(cudaSetDeviceFlags(cudaDeviceScheduleYield | cudaDeviceMapHost | cudaDeviceLmemResizeToMax));

	std::cout << "Set Device Number: " << device_num << std::endl;
	checkCudaErrors(cudaSetDevice(device_num));


	// Initialize GLUT
	std::cout << "Initializing GLUT..." << std::endl;
	glutInit(&argc, argv);

	// Initialize OpenGL
	std::cout << "Initializing OpenGL..." << std::endl;
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);
	glutInitWindowSize(kWindowSize[0], kWindowSize[1]);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("CUDA WCSPH");

	screen_size[0] = glutGet(GLUT_SCREEN_WIDTH);
	screen_size[1] = glutGet(GLUT_SCREEN_HEIGHT);
	std::cout << "Screen Size: (" << screen_size[0] << " x " << screen_size[1] << ")" << std::endl;
	window_size[0] = glutGet(GLUT_WINDOW_WIDTH);
	window_size[1] = glutGet(GLUT_WINDOW_HEIGHT);
	std::cout << "Window Size: (" << window_size[0] << " x " << window_size[1] << ")" << std::endl;
	// make window center
	glutPositionWindow((screen_size[0] - window_size[0]) / 2, (screen_size[1] - window_size[1]) / 2);
	
	std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
	std::cout << "OpenGL Renderer: " << glGetString(GL_RENDERER) << std::endl;
	std::cout << "Shading Language Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

	glutDisplayFunc(displayFunc); // tells glut to use a particular display function to redraw 
	glutIdleFunc(idleFunc); // perform animation inside idleFunc
	glutMotionFunc(mouseMotionDragFunc); // callback for mouse drags
	glutPassiveMotionFunc(mouseMotionFunc); // callback for idle mouse movement
	glutMouseFunc(mouseButtonFunc); // callback for mouse button changes
	glutReshapeFunc(reshapeFunc); // callback for resizing the window
	glutKeyboardFunc(keyboardFunc); // callback for pressing the keys on the keyboard

	GLint result = glewInit();
	if (result != GLEW_OK)
	{
		std::cout << "error: " << glewGetErrorString(result) << std::endl;
		exit(EXIT_FAILURE);
	}

	initFluidSystem();
	initScene(sph_host);
	glutMainLoop();
}

void initScene(WCSPHSystem* sys) {

	// init box
	box_program.Init("shader/box.vs", "shader/box.fs");
	box_program.Bind();

	float box_vbo_data[] = {
		0.0 - 0.5,			0.0 - 0.5,			0.0 - 0.5,
		0.0 - 0.5,			box_length - 0.5,	0.0 - 0.5,
		box_length - 0.5,	box_length - 0.5,	0.0 - 0.5,
		box_length - 0.5,	0.0 - 0.5,			0.0 - 0.5,
		0.0 - 0.5,			0.0 - 0.5,			box_length - 0.5,
		0.0 - 0.5,			box_length - 0.5,	box_length - 0.5,
		box_length - 0.5,	box_length - 0.5,	box_length - 0.5,
		box_length - 0.5,	0.0 - 0.5,			box_length - 0.5
	};

	unsigned int box_ebo_data[] = {
		0, 1,
		1, 2,
		2, 3,
		3, 0,
		4, 5,
		5, 6,
		6, 7,
		7, 4,
		1, 5,
		2, 6,
		0, 4,
		3, 7,
	};

	glGenVertexArrays(1, &box_vao);
	glBindVertexArray(box_vao);

	glGenBuffers(1, &box_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, box_vbo);
	glBufferData(GL_ARRAY_BUFFER, 3 * 8 * sizeof(float), box_vbo_data, GL_STATIC_DRAW);

	glGenBuffers(1, &box_ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, box_ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, 2 * 12 * sizeof(unsigned int), box_ebo_data, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), (void*)0);

	glEnableVertexAttribArray(0);

	glGenBuffers(1, &position_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, position_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float3) * sys->particle_num, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_position_resource, position_vbo, cudaGraphicsMapFlagsNone));


	glGenBuffers(1, &color_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float4) * sys->particle_num, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_color_resource, color_vbo, cudaGraphicsMapFlagsNone));

	particle_program.Init("shader/particle.vs", "shader/particle.fs");

}

void drawParticles() {

	glBindBuffer(GL_ARRAY_BUFFER, position_vbo);
	glVertexPointer(3, GL_FLOAT, 0, nullptr);
	glEnableClientState(GL_VERTEX_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
	glColorPointer(3, GL_FLOAT, 0, nullptr);
	glEnableClientState(GL_COLOR_ARRAY);

	glEnable(GL_POINT_SMOOTH);
	glDrawArrays(GL_POINTS, 0, sph_host->particle_num);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);

	//glutSolidSphere
	if (interrupt)
		return;

#ifdef OUTPUT_FRAME_NUM
	std::cout << "Frame # " << frame_cnt++ << std::endl;
#endif // OUTPUT_FRAME_NUM

	// update color
	getNextFrame(sph_host, cuda_position_resource, cuda_color_resource);

	if (is_one_frame) {
		interrupt = true;
	}
}

void displayFunc() {

	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	float p[16];
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fov, 1.0, 0.01, 100.0);
	glGetFloatv(GL_PROJECTION_MATRIX, p);

	float m[16];
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// intialize camera 
	glm::vec3 camera_pos = glm::vec3(0, 0, 3);
	glm::vec3 camera_look = glm::vec3(0, 0, 0);
	glm::vec3 camera_up = glm::vec3(0, 1, 0);

	camera_pos = glm::rotateX(camera_pos, land_rotate[0]);
	camera_pos = glm::rotateY(camera_pos, land_rotate[1]);
	//float dist = glm::distance(camera_pos, glm::vec3(0, 0, 0));
	camera_pos = glm::vec3(camera_pos.x * land_scale[0], camera_pos.y * land_scale[0], camera_pos.z * land_scale[0]);

	camera_up = glm::rotateX(camera_up, land_rotate[0]);
	camera_up = glm::rotateY(camera_up, land_rotate[1]);

	gluLookAt(camera_pos.x, camera_pos.y, camera_pos.z, 0, 0, 0, camera_up.x, camera_up.y, camera_up.z);
	
	glGetFloatv(GL_MODELVIEW_MATRIX, m);


	// draw box
	box_program.Bind();
	box_program.SetModelViewMatrix(m);
	box_program.SetProjectionMatrix(p);
	box_program.SetFloat3("box_length", sph_host->box_length);
	glBindVertexArray(box_vao);
	glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);

	// draw paricle
	particle_program.Bind();
	glEnable(GL_POINT_SPRITE_ARB);
	glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);

	particle_program.SetModelViewMatrix(m);
	particle_program.SetProjectionMatrix(p);
	particle_program.SetFloat("pointScale", kWindowSize[1] / tanf(fov * 0.5f * M_PI / 180.0f));
	particle_program.SetFloat("pointRadius", sph_host->particle_radius);
	drawParticles();

	glDisable(GL_POINT_SPRITE_ARB);
	glDisable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);

	glutSwapBuffers();

}

void closeWindow() {
	FreeDeviceSystem(sph_host);
	// CPU
	delete pos_init;
	delete velo_init;
	delete dens_init;

	// GPU
	glDeleteBuffers(1, &position_vbo);
	glDeleteBuffers(1, &color_vbo);

	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_position_resource));
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_color_resource));
	checkCudaErrors(cudaDeviceReset());
	exit(0);
}

void idleFunc() {
	// for example, here, you can save the screenshots to disk (to make the animation)
	// make the screen update 
	glutPostRedisplay();
}

void reshapeFunc(int w, int h) {
	glViewport(0, 0, w, h);
}

void mouseMotionDragFunc(int x, int y) {
	// mouse has moved and one of the mouse buttons is pressed (dragging)

	// the change in mouse position since the last invocation of this function
	int mousePosDelta[2] = { x - mouse_pos[0], y - mouse_pos[1] };

	switch (control_state)
	{
		// translate the landscape
	case TRANSLATE:
		//if (left_mouse_button)
		//{
		//	// control x,y translation via the left mouse button
		//	land_translate[0] += mousePosDelta[0] * 0.01f;
		//	land_translate[1] -= mousePosDelta[1] * 0.01f;
		//}
		//if (right_mouse_button)
		//{
		//	// control z translation via the right mouse button
		//	land_translate[2] += mousePosDelta[1] * 0.01f;
		//}
		break;

		// rotate the landscape
	case ROTATE:
		if (left_mouse_button)
		{
			// control x,y rotation via the left mouse button
			/*land_rotate[0] += mousePosDelta[1];
			land_rotate[1] += mousePosDelta[0];*/
			land_rotate[1] -= mousePosDelta[0] / 100.0;
			land_rotate[0] -= mousePosDelta[1] / 100.0;
		}
		if (right_mouse_button)
		{
			// control z rotation via the right mouse button
			/*land_rotate[2] += mousePosDelta[1];*/
		}
		break;

		// scale the landscape
	case SCALE:
		if (left_mouse_button)
		{
			// control x,y scaling via the left mouse button
			/*land_scale[0] *= 1.0f + mousePosDelta[0] * 0.01f;
			land_scale[1] *= 1.0f - mousePosDelta[1] * 0.01f;*/
			land_scale[0] *= 1.0 + mousePosDelta[1] * 0.01;
		}
		if (right_mouse_button)
		{
			// control z scaling via the right mouse button
			/*land_scale[2] *= 1.0f - mousePosDelta[1] * 0.01f;*/

		}
		break;
	}

	// store the new mouse position
	mouse_pos[0] = x;
	mouse_pos[1] = y;
}

void mouseMotionFunc(int x, int y) {
	// mouse has moved
	// store the new mouse position
	mouse_pos[0] = x;
	mouse_pos[1] = y;
}

void mouseButtonFunc(int button, int state, int x, int y) {
	// a mouse button has has been pressed or depressed
	switch (button)
	{
	case GLUT_LEFT_BUTTON:
		left_mouse_button = (state == GLUT_DOWN);
		break;

	case GLUT_MIDDLE_BUTTON:
		middle_mouse_button = (state == GLUT_DOWN);
		break;

	case GLUT_RIGHT_BUTTON:
		right_mouse_button = (state == GLUT_DOWN);
		break;
	}

	// keep track of whether CTRL and SHIFT keys are pressed
	switch (glutGetModifiers())
	{
	case GLUT_ACTIVE_CTRL:
		control_state = TRANSLATE;
		break;

	case GLUT_ACTIVE_SHIFT:
		control_state = SCALE;
		break;

		// if CTRL and SHIFT are not pressed, we are in rotate mode
	default:
		control_state = ROTATE;
		break;
	}

	// store the new mouse position
	mouse_pos[0] = x;
	mouse_pos[1] = y;
}

void keyboardFunc(unsigned char key, int x, int y) {
	switch (key)
	{
	case 27: // ESC key
		closeWindow();
		break;

	case ' ':
		std::cout << "You pressed the spacebar." << std::endl;
		break;

	case 'x':
		// take a screenshot
		//saveScreenshot("screenshot.jpg");
		break;

	case 'p':
		is_one_frame = false;
		interrupt = !interrupt;
		break;

	case 'f':
		is_one_frame = true;
		interrupt = false;
		break;
	}
}
