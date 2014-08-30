#include "Config.h"
#include "Camera.h"
#include "Light.h"
#include "Scene.h"
#include "domain.h"
#include "common.h"
#include "Barrier.h"

#include "space_col.h"

#include <math.h>
#include <memory.h>
#include <time.h>
#include "GL/glut.h"
#include "GL/glui.h"

//
//	Global

Camera	cam;
Light	light;
Scene	scene;
Domain	dmn;

///
///	For Information Display

void renderBitmapString(float x, float y, void *font, char *string) 
{  
	char *c;
	glRasterPos2f(x,y);
	for (c = string; *c != '\0'; c++) 
	{
		glutBitmapCharacter(font, *c);
	}
}

void info_display(char *info, int x, int y)
{
	glPushMatrix();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(-10, 10, -10, 10);

	glDisable(GL_LIGHTING);
	renderBitmapString(x, y, GLUT_BITMAP_TIMES_ROMAN_24, info);
	glEnable(GL_LIGHTING);

	glPopMatrix();
}

void Destroy()
{
#if GPU_SPEED_UP
	cudaFree(d_mat);
	cudaFree(d_internodes);
	cudaFree(d_optVec);
	cudaFree(d_shadow);
	cudaFree(d_barriers);
#endif
	free(h_internodes);
	free(h_shadow);
	delete [] optVecs;
	dmn.Destroy();
}

void Init()
{
#if GPU_SPEED_UP

	printf("GPU Memory Init...");

	optVecs = new OptVec[MAX_INTERNODES];

	//	CUDA Init
	cudaMalloc((void **)&d_optVec, sizeof(struct OptVec) * MAX_INTERNODES);
	cudaMemset((void *)d_optVec, 0, sizeof(struct OptVec) * MAX_INTERNODES);

	cudaMalloc((void **)&d_mat, X_Dim * Y_Dim * Z_Dim * sizeof(struct Cell));   
	cudaMemset((void *)d_mat, 0, X_Dim * Y_Dim * Z_Dim * sizeof(struct Cell));

	cudaMalloc((void **)&d_internodes, sizeof(InternodeDelegate) * MAX_INTERNODES);
	cudaMemset((void *)d_internodes, 0, sizeof(InternodeDelegate) * MAX_INTERNODES);
	h_internodes = (InternodeDelegate*) malloc(sizeof(InternodeDelegate) * MAX_INTERNODES);
	memset(h_internodes, 0, sizeof(InternodeDelegate) * MAX_INTERNODES);

	//	Barrier Memory
	cudaMalloc((void **)&d_barriers, sizeof(Barrier) * BARRIER_COUNT);
	cudaMemset((void *)d_barriers, 0, sizeof(Barrier) * BARRIER_COUNT);

	//	Shadow
	size_t nShadowBufSize = pow(SCENESIZE*2/SpaceGranularity, 2);
	h_shadow = (float*)malloc(sizeof(float)*nShadowBufSize);
	cudaMalloc((void **)&d_shadow, sizeof(float) * nShadowBufSize);

	//	Init
	gpuInit<<<BLOCK_DIM_1D, THREAD_DIM_1D>>>(d_mat);
	cudaThreadSynchronize();  
	cudaError_t eid = cudaGetLastError();
	if(eid != cudaSuccess)
	{
		const char *errStr = cudaGetErrorString(eid);
		printf("\n%s\n", errStr);
	}
	printf(" Done\n");
#endif

	//	
	srand(clock());

	glClearColor(26/255.f, 19/255.f, 100/255.f, 1);

	//	Camera Setup
	cam.Set(90,  50, 90, 
			0, 5, 0, 
			-1,  2, -1);
	
	//	Depth
	glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_TEST);

	//	Lighting
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);	
	
	GLfloat light_ambient[] = { 0.2, 0.2, 0.2, 0 }; 
	GLfloat light_diffuse[] = { 0.2, 0.2, 0.2, 0 };
	GLfloat light_specular[] = { 0.3, 0.3, 0.3, 0 };
	float sun_vec[3] = SUN_VEC;
	GLfloat light_position[] = { sun_vec[0], sun_vec[1], sun_vec[2], 0 };

	light.Set(light_ambient, light_diffuse, light_specular,	light_position);
	light.Put();

	//	Line Smooth
	glEnable(GL_LINE_SMOOTH); 
	glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE);

	//	Domain Init
	dmn.Init();
}

long nLastTick = 0;

void Display(void)
{
	if(nLastTick == 0)
	{
		nLastTick = clock();
	}

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//	Set Camera
	//
	cam.Put();

	//	Render 
	//
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
		
	scene.Render();
	dmn.Render();
	
	//	Display Frame Rate	
	//
	long nCurrTick = clock();

	const size_t BufSize = 20;
	char buf[BufSize] = {0};
	sprintf(buf, "%.2f",  1000.f / (nCurrTick - nLastTick));
	glColor3f(0, 1, 0);
	info_display(buf, 8, 9);
	nLastTick = nCurrTick;

	//	Display current generation stage
	//
	memset(buf, 0, sizeof(char)*BufSize);
	sprintf(buf, "Stage %d \\ %d", GrownStep, GEN_COUNT - 1 );
	glColor3f(0, 0, 1);
	info_display(buf, -10, 9);

	//	Flush
	glutSwapBuffers();    

	////	Reset everything for next frame
	////
	//printf("Resetting...\n");
	//gpuInit<<<BLOCK_DIM_1D, THREAD_DIM_1D>>>(d_mat);
	//cudaThreadSynchronize();  
	//cudaError_t eid = cudaGetLastError();
	//if(eid != cudaSuccess)
	//{
	//	const char *errStr = cudaGetErrorString(eid);
	//	printf("\n%s\n", errStr);
	//}
	//tree.internodes.clear();
	//printf("Resetting Done...\n");
}

void Reshape(int w, int h)
{
	glViewport(0,0,w, h);       
}

void Idle()
{
	glutPostRedisplay();
}

void Kbd(unsigned char a, int x, int y)
{

	switch(a)
	{
	case ' ':
		// growth
		if( GrownStep < GEN_COUNT)
		{
			GrownStep ++;
			printf("Growth Step: %d\n", GrownStep);
			glutPostRedisplay();
		}
		break;

	case 'w':
	case 'W':
		cam.ZoomIn();
		glutPostRedisplay();
		break;

	case 's':
	case 'S':
		cam.ZoomOut();
		glutPostRedisplay();
		break;

	case 'a':
	case 'A':
		cam.Rotate(5);
		glutPostRedisplay();
		break;

	case 'd':
	case 'D':
		cam.Rotate(-5);
		glutPostRedisplay();
		break;		

    case 27 : 		  
	    Destroy(); 		  
	    exit(0);

	    break;
	}	
}

int main(int argc, char* argv[])
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE);
	glutInitWindowSize(WindowWidth, WindowHeight);
	int iWinId = glutCreateWindow("Signal in Plants - Tony Zhang");
	           
	glutDisplayFunc(Display);	
	glutReshapeFunc(Reshape);
	//glutIdleFunc(Idle);

	glutKeyboardFunc(Kbd); 

	Init();

	//	GLUI
	GLUI *glui = GLUI_Master.create_glui( "Param Control", 0, WindowWidth + 20, 10 );
	
	glui->add_checkbox("Render Buds", &iRenderBud );
	glui->add_checkbox("Render Ground Shadow", &iRenderGroundShadow );
	glui->add_checkbox("Show Space Status", &iShowCellSpaceStatus );
	glui->add_checkbox("Show Shadow Status", &iShowCellShadow );
	   
    glui->set_main_gfx_window( iWinId );

	//	GO! 
	glutMainLoop();

	Destroy();

	return 0;
}

