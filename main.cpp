
#include "WCSPHSystem.h"
#include "WCSPHSolver.cuh"
#include "ShaderProgram.h"
#include "utils/handler.h"

#include <gl/GL.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <time.h>

#define OUTPUT_FRAME_NUM

typedef enum { ROTATE, TRANSLATE, SCALE } CONTROL_STATE;

const int kWindowSize[2] = { 512,512 };

///////////////////////////////////////////////////////////////////////////////
// Dispaly Settings
///////////////////////////////////////////////////////////////////////////////

int fov = 45;
CONTROL_STATE controlState = TRANSLATE;
int mousePos[2];
bool leftMouseButton = false;
bool middleMouseButton = false;
bool rightMouseButton = false;
bool interrupt = false;
bool oneFrame = true;
long frameCnt = 0;

// state of the world
float landRotate[3] = { 0.0f, 0.0f, 0.0f };
float landTranslate[3] = { 0.0f, 0.0f, 0.0f };
float landScale[3] = { 1.0f, 1.0f, 1.0f };

// gl buffers
ShaderProgram boxProgram, particleProgram;
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
float box_size = 1.0;
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

	sph_host = new WCSPHSystem("WCSPH_config.json");
	int dim_max = ceil(3 / (4 * sph_host->h));
	if (!(sph_host->block_dim <= make_int3(dim_max))) {
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

	//
	glutInitWindowSize(kWindowSize[0], kWindowSize[1]);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("CUDA WCSPH");

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
	boxProgram.Init("shader/box.vs", "shader/box.fs");
	boxProgram.Bind();

	float box_vbo_data[] = {
		0.0,      0.0,      0.0,
		0.0,      box_size, 0.0,
		box_size, box_size, 0.0,
		box_size, 0.0,	    0.0,
		0.0,	  0.0,      box_size,
		0.0,      box_size, box_size,
		box_size, box_size, box_size,
		box_size, 0.0,      box_size
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

	particleProgram.Init("shader/particle.vs", "shader/particle.fs");

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
	std::cout << "frame: " << frameCnt++ << std::endl;
#endif // OUTPUT_FRAME_NUM

	// update color
	getNextFrame(sph_host, cuda_position_resource, cuda_color_resource);

	if (oneFrame) {
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
	glTranslatef(landTranslate[0], landTranslate[1], landTranslate[2]);
	glRotatef(landRotate[0], 1.0, 0.0, 0.0);
	glRotatef(landRotate[1], 0.0, 1.0, 0.0);
	glRotatef(landRotate[2], 0.0, 0.0, 1.0);
	glScalef(landScale[0], landScale[1], landScale[2]);
	gluLookAt(1.6, 0.8, 2.3, 0.5, 0.5, 0.5, 0, 1, 0);
	glGetFloatv(GL_MODELVIEW_MATRIX, m);


	// draw box
	boxProgram.Bind();
	boxProgram.SetModelViewMatrix(m);
	boxProgram.SetProjectionMatrix(p);
	glBindVertexArray(box_vao);
	glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, 0);
	glBindVertexArray(0);

	// draw paricle
	particleProgram.Bind();
	glEnable(GL_POINT_SPRITE_ARB);
	glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);

	particleProgram.SetModelViewMatrix(m);
	particleProgram.SetProjectionMatrix(p);
	particleProgram.SetFloat("pointScale", kWindowSize[1] / tanf(fov * 0.5f * M_PI / 180.0f));
	particleProgram.SetFloat("pointRadius", sph_host->particle_radius);
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
	int mousePosDelta[2] = { x - mousePos[0], y - mousePos[1] };

	switch (controlState)
	{
		// translate the landscape
	case TRANSLATE:
		if (leftMouseButton)
		{
			// control x,y translation via the left mouse button
			landTranslate[0] += mousePosDelta[0] * 0.01f;
			landTranslate[1] -= mousePosDelta[1] * 0.01f;
		}
		if (rightMouseButton)
		{
			// control z translation via the right mouse button
			landTranslate[2] += mousePosDelta[1] * 0.01f;
		}
		break;

		// rotate the landscape
	case ROTATE:
		if (leftMouseButton)
		{
			// control x,y rotation via the left mouse button
			landRotate[0] += mousePosDelta[1];
			landRotate[1] += mousePosDelta[0];
		}
		if (rightMouseButton)
		{
			// control z rotation via the right mouse button
			landRotate[2] += mousePosDelta[1];
		}
		break;

		// scale the landscape
	case SCALE:
		if (leftMouseButton)
		{
			// control x,y scaling via the left mouse button
			landScale[0] *= 1.0f + mousePosDelta[0] * 0.01f;
			landScale[1] *= 1.0f - mousePosDelta[1] * 0.01f;
		}
		if (rightMouseButton)
		{
			// control z scaling via the right mouse button
			landScale[2] *= 1.0f - mousePosDelta[1] * 0.01f;
		}
		break;
	}

	// store the new mouse position
	mousePos[0] = x;
	mousePos[1] = y;
}

void mouseMotionFunc(int x, int y) {
	// mouse has moved
	// store the new mouse position
	mousePos[0] = x;
	mousePos[1] = y;
}

void mouseButtonFunc(int button, int state, int x, int y) {
	// a mouse button has has been pressed or depressed
	switch (button)
	{
	case GLUT_LEFT_BUTTON:
		leftMouseButton = (state == GLUT_DOWN);
		break;

	case GLUT_MIDDLE_BUTTON:
		middleMouseButton = (state == GLUT_DOWN);
		break;

	case GLUT_RIGHT_BUTTON:
		rightMouseButton = (state == GLUT_DOWN);
		break;
	}

	// keep track of whether CTRL and SHIFT keys are pressed
	switch (glutGetModifiers())
	{
	case GLUT_ACTIVE_CTRL:
		controlState = TRANSLATE;
		break;

	case GLUT_ACTIVE_SHIFT:
		controlState = SCALE;
		break;

		// if CTRL and SHIFT are not pressed, we are in rotate mode
	default:
		controlState = ROTATE;
		break;
	}

	// store the new mouse position
	mousePos[0] = x;
	mousePos[1] = y;
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
		oneFrame = false;
		interrupt = !interrupt;
		break;

	case 'f':
		oneFrame = true;
		interrupt = false;
		break;
	}
}
