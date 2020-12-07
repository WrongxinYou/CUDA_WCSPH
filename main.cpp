#include "ShaderProgram.h"
#include "SPHSystem.h"
#include "sph_host.cuh"
#include "Global.h"
#include <gl/GL.h>

#include "cuda_gl_interop.h"
#include "cuda_runtime.h"

using namespace std;

typedef enum { ROTATE, TRANSLATE, SCALE } CONTROL_STATE;


///////////////////////////////////////////////////////////////////////////////
// Dispaly Settings
///////////////////////////////////////////////////////////////////////////////
int res[2] = { 512,512 };
CONTROL_STATE controlState = TRANSLATE;
int mousePos[2];
bool leftMouseButton = false;
bool middleMouseButton = false;
bool rightMouseButton = false;

// state of the world
float landRotate[3] = { 0.0f, 0.0f, 0.0f };
float landTranslate[3] = { 0.0f, 0.0f, 0.0f };
float landScale[3] = { 1.0f, 1.0f, 1.0f };

// gl buffers
ShaderProgram boxProgram, particleProgram;
unsigned int box_vbo, box_ebo, box_vao;
unsigned int particle_vbo, color_vbo;

// gl functions
void initScene(SPHSystem* sys);
void displayFunc();
void idleFunc();
void mouseMotionDragFunc(int x, int y);
void mouseMotionFunc(int x, int y);
void mouseButtonFunc(int button, int state, int x, int y);
void reshapeFunc(int w, int h);
void keyboardFunc(unsigned char key, int x, int y);

// box and board
float box_size = 1.0;
float board_pos = 0.5f;

/// TODO
// 1. do not allow scaling
// 2. change default operation = translate


///////////////////////////////////////////////////////////////////////////////
// SPH Settings
///////////////////////////////////////////////////////////////////////////////
SPHSystem* sphSys;


void initFluidSystem() {
	sphSys = new SPHSystem();
	sphSys->Initialize();
	InitDeviceSystem(sphSys);
}


int main(int argc, char* argv[]) {
	cout << "Initializing GLUT..." << endl;
	glutInit(&argc, argv);


	cout << "Initializing OpenGL..." << endl;
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_STENCIL);

	glutInitWindowSize(res[0], res[1]);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("CUDA SPH");

	cout << "OpenGL Version: " << glGetString(GL_VERSION) << endl;
	cout << "OpenGL Renderer: " << glGetString(GL_RENDERER) << endl;
	cout << "Shading Language Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;

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
		cout << "error: " << glewGetErrorString(result) << endl;
		exit(EXIT_FAILURE);
	}

	initFluidSystem();
	initScene(sphSys);
	glutMainLoop();
}

void createCudaVBO(GLuint* vbo, const unsigned int size) {
	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	glBufferData(GL_ARRAY_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register buffer object with CUDA
	checkCudaErrors(cudaGLRegisterBufferObject(*vbo));
}

void deleteCudaVBO(GLuint* vbo) {
	glGenBuffers(1, vbo);
	glDeleteBuffers(1, vbo);

	checkCudaErrors(cudaGLUnregisterBufferObject(*vbo));
	//CUDA_CHECK(cudaGLUnregisterBufferObject(*vbo));
	*vbo = NULL;
}

void initScene(SPHSystem* sys) {

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


	// init particle
	int size = sys->particle_num;
	// create VBOs
	createCudaVBO(&particle_vbo, sizeof(float3) * size);
	createCudaVBO(&color_vbo, sizeof(float3) * size);
	// initiate shader program
	particleProgram.Init("shader/particle.vs", "shader/particle.fs");
}

void drawParticles() {
	float3* cuda_particle_vbo;
	float3* cuda_color_vbo;

	checkCudaErrors(cudaGLMapBufferObject((void**)&cuda_particle_vbo, particle_vbo));
	checkCudaErrors(cudaGLMapBufferObject((void**)&cuda_color_vbo, color_vbo));

	/*CUDA_CHECK(cudaGLMapBufferObject((void**)cuda_particle_vbo, particle_vbo));
	CUDA_CHECK(cudaGLMapBufferObject((void**)cuda_color_vbo, color_vbo));*/

	// update color
	getNextFrame(cuda_particle_vbo, cuda_color_vbo, sphSys);

	checkCudaErrors(cudaGLUnmapBufferObject(particle_vbo));
	checkCudaErrors(cudaGLUnmapBufferObject(color_vbo));
	/*CUDA_CHECK(cudaGLUnmapBufferObject(particle_vbo));
	CUDA_CHECK(cudaGLUnmapBufferObject(color_vbo));*/

	glBindBuffer(GL_ARRAY_BUFFER, particle_vbo);
	glVertexPointer(3, GL_FLOAT, 0, nullptr);
	glEnableClientState(GL_VERTEX_ARRAY);

	glBindBuffer(GL_ARRAY_BUFFER, color_vbo);
	glColorPointer(3, GL_FLOAT, 0, nullptr);
	glEnableClientState(GL_COLOR_ARRAY);

	glDrawArrays(GL_POINTS, 0, sphSys->particle_num);

	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);

}

void displayFunc() {

	//glutSolidSphere

	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	float m[16];
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(landTranslate[0], landTranslate[1], landTranslate[2]);
	glRotatef(landRotate[0], 1.0, 0.0, 0.0);
	glRotatef(landRotate[1], 0.0, 1.0, 0.0);
	glRotatef(landRotate[2], 0.0, 0.0, 1.0);
	glScalef(landScale[0], landScale[1], landScale[2]);
	gluLookAt(box_size / 2.0f, box_size / 2.0f, box_size / 2.0f, box_size * 1.4, box_size, box_size, 0, 1, 0);
	glGetFloatv(GL_MODELVIEW_MATRIX, m);
	boxProgram.SetModelViewMatrix(m);

	float p[16];
	glMatrixMode(GL_PROJECTION);
	glGetFloatv(GL_PROJECTION_MATRIX, p);
	boxProgram.SetProjectionMatrix(p);


	// draw box
	boxProgram.Bind();
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
	glPushMatrix();
	drawParticles();
	glPopMatrix();

	glPopMatrix();

	glutSwapBuffers();
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

void closeWindow() {
	FreeDeviceSystem(sphSys);
	deleteCudaVBO(&particle_vbo);
	deleteCudaVBO(&color_vbo);
	checkCudaErrors(cudaDeviceReset());
	//CUDA_CHECK(cudaDeviceReset());
	exit(0);
}

void keyboardFunc(unsigned char key, int x, int y) {
	switch (key)
	{
	case 27: // ESC key
		closeWindow();
		break;

	case ' ':
		cout << "You pressed the spacebar." << endl;
		break;

	case 'x':
		// take a screenshot
		//saveScreenshot("screenshot.jpg");
		break;
	}
}