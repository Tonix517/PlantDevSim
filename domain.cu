#include "domain.h"

#include <algorithm>
#include <assert.h>
#include <stdlib.h>

#include "Config.h"
#include "GL/glut.h"
#include "Common.h"
#include "space_col.h"
#include "Bud.h"
#include "Barrier.h"
#include "util.h"
#include "shadow.h"

///
float *h_shadow;
float *d_shadow;
///
///

void Domain::makeInternode( Internode &rNode, int nParentId, 
						    Vect3d &vNodePos, Vect3d &vNodeVec, 
							Vect3d &vTermBudVec,
							bool bHasLeaf, bool bHasAnBud)
{		
	// meta info
	rNode.id = nInternodeCount;
	rNode.ancestorId = nParentId;
	nInternodeCount ++;

	//	geometry
	rNode.pos.Set( vNodePos.x(), vNodePos.y(), vNodePos.z());
	rNode.vec.Set( vNodeVec.x(), vNodeVec.y(), vNodeVec.z());

	//	the Bud for the Internode	
	Bud *pNewBud = new Bud;
	pNewBud->pos.Set(rNode.pos + rNode.vec);
	pNewBud->ort.Set(vTermBudVec);
	rNode.pTerminalBud.reset(pNewBud);

	float xi, yi, zi;
	if( bHasLeaf || bHasAnBud)
	{		
		while( (xi = rand() % 1000) == 0);
		while( (yi = rand() % 1000) == 0);
		float a2 = rNode.vec.x();
		float b2 = rNode.vec.y();
		float c2 = rNode.vec.z();
		
		if(c2 != 0)
		{
			zi = -(xi*a2 + yi*b2)/c2;			
		}
		else
		{
			while( (zi = rand() % 1000) == 0);

			if(a2 != 0)
			{				
				xi = -yi*b2/a2;				
			}
			else if(b2 != 0)
			{
				yi = -xi*a2/b2;				
			}
			else
			{
				assert("Dude!");
			}

		}
	}

	//	Leaf
	if(bHasLeaf)
	{
		Leaf *pNewLeaf = new Leaf;
		pNewLeaf->pos.Set(rNode.pos + rNode.vec);
		
		float ret = xi * pNewLeaf->vec.x() + 	
					yi * pNewLeaf->vec.y() + 
					zi * pNewLeaf->vec.z() ;
		assert(ret == 0);

		pNewLeaf->vec.Set(xi, yi, zi);
		pNewLeaf->vec.Normalize();
		pNewLeaf->vec *= 1.5;
		rNode.pLeaf.reset(pNewLeaf);
	}

	if(bHasAnBud)
	{
		//	Anxillary Bud for the Internode	
		Bud *pNewAnBud = new Bud;
		
		//	Get Anxillary Bud Vec.
		Vect3d anVec, tmp1, tmp2;
		tmp1.Set(rNode.pTerminalBud->ort); tmp1.Normalize();
		tmp2.Set(xi, yi, zi); tmp2.Normalize();
		anVec += tmp1;
		float w = rand() % 20 / 100 + 1;
		anVec += tmp2 * w;
		anVec.Normalize();

		pNewAnBud->pos.Set(rNode.pos + rNode.vec + anVec * 0.1);
		pNewAnBud->ort = anVec;
		rNode.pAnxillaryBud.reset(pNewAnBud);
	}
}

void Domain::addInternode( Internode &rNode)
{
	rNode.pTerminalBud->nId = tree.internodes.size(); 	
	tree.internodes.push_back(rNode);
}

//
void Domain::RenderFrame()
{
	glColor3f(1, 228.f/255, 196.f/255);

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glDisable(GL_LIGHTING);

	//	Top
	glBegin(GL_QUADS);
		glVertex3f(-SCENESIZE, SCENEHEIGHT, -SCENESIZE);
		glVertex3f( SCENESIZE, SCENEHEIGHT, -SCENESIZE);
		glVertex3f( SCENESIZE, SCENEHEIGHT,  SCENESIZE);
		glVertex3f(-SCENESIZE, SCENEHEIGHT,  SCENESIZE);
	glEnd();

	//	Front
	glBegin(GL_QUADS);
		glVertex3f(-SCENESIZE, SCENEHEIGHT, -SCENESIZE);
		glVertex3f( SCENESIZE, SCENEHEIGHT, -SCENESIZE);
		glVertex3f( SCENESIZE, SCENEHEIGHT + DOMAIN_HEIGHT,  -SCENESIZE);
		glVertex3f(-SCENESIZE, SCENEHEIGHT + DOMAIN_HEIGHT,  -SCENESIZE);
	glEnd();

	//	Back
	glBegin(GL_QUADS);
		glVertex3f( SCENESIZE, SCENEHEIGHT, SCENESIZE);
		glVertex3f(-SCENESIZE, SCENEHEIGHT, SCENESIZE);
		glVertex3f(-SCENESIZE, SCENEHEIGHT + DOMAIN_HEIGHT,  SCENESIZE);
		glVertex3f( SCENESIZE, SCENEHEIGHT + DOMAIN_HEIGHT,  SCENESIZE);
	glEnd();

	// Bottom
	glBegin(GL_QUADS);		
		glVertex3f(-SCENESIZE, SCENEHEIGHT + DOMAIN_HEIGHT, -SCENESIZE);
		glVertex3f( SCENESIZE, SCENEHEIGHT + DOMAIN_HEIGHT, -SCENESIZE);
		glVertex3f( SCENESIZE, SCENEHEIGHT + DOMAIN_HEIGHT,  SCENESIZE);
		glVertex3f(-SCENESIZE, SCENEHEIGHT + DOMAIN_HEIGHT,  SCENESIZE);
	glEnd();

	glEnable(GL_LIGHTING);
}

void Domain::RenderTree()
{
	std::vector<Internode>::iterator iter = tree.internodes.begin();
	for(; iter != tree.internodes.end(); iter++)
	{
		iter->Render();
		//printf(" %d -> %d \n", iter->id, iter->ancestorId);
	}
}

struct Cell *pMat = NULL;

void Domain::Render()
{
	//RenderFrame();
	if( (GrownStep < GEN_COUNT) && (GrownStep != _nLastGen) )
	{
		_nLastGen = GrownStep;

		printf("Handling Space Colonization...\n");
		gpu_SpaceColonization();	

		printf("Shadow Propagation...\n");
		shadow_calculation_on_voxels<<<BLOCK_DIM_1D, THREAD_DIM_1D>>>(d_internodes, tree.internodes.size());
		cudaError_t eid = cudaGetLastError();
		if(eid != cudaSuccess)
		{
			const char *errStr = cudaGetErrorString(eid);
			printf("\n%s\n", errStr);
		}
		
		//	TODO
		//	Calculating Light received here
		//

		printf("Growing %d...\n", GrownStep);
		grow();
		if(iRenderGroundShadow)
		{
			printf("Calculating Shadow...");
				CalcGroundShadow();
			printf(" Done.\n");
		}
	}
	
	RenderTree();
	RenderBarriers();

	if(iRenderGroundShadow)
	{
		RenderGroundShadow();
	}

#if 1
	//	checking everything and render sth.
	//
	pMat = (struct Cell*)malloc(X_Dim * Y_Dim * Z_Dim * sizeof(struct Cell));
	cudaMemcpy(pMat, d_mat, X_Dim * Y_Dim * Z_Dim * sizeof(struct Cell), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	if(iShowCellSpaceStatus != 0)
	{
		checkCells();
	}

	if(iShowCellShadow != 0)
	{
		checkShadow();
	}

	free(pMat);
#endif
}

void Domain::checkShadow()
{
	glDisable(GL_LIGHTING);
	
	//	Cell Matrix Initialization
	for(int i = 0; i < X_Dim; i++)
	for(int j = 0; j < Y_Dim; j++)
	for(int k = 0; k < Z_Dim; k++)
	{		
		float fIllumValue = (pMat + i * Y_Dim * Z_Dim + j * Z_Dim + k)->fIllum;
		if( fIllumValue < 1)
		{
			float point[3];
			point[0] = i * SpaceGranularity + (-SCENESIZE);
			point[1] = j * SpaceGranularity + (SCENEHEIGHT);
			point[2] = k * SpaceGranularity + (-SCENESIZE);

			glColor3f(  fIllumValue, 
						fIllumValue, 
						fIllumValue );
			glPushMatrix();
				glTranslatef(point[0], point[1], point[2]);
				glutSolidCube(SpaceGranularity);
			glPopMatrix();
		}
	}
	glEnd();
	glEnable(GL_LIGHTING);	
}

void Domain::checkCells()
{	
	
	glDisable(GL_LIGHTING);
	glBegin(GL_POINTS);
	//	Cell Matrix Initialization
	for(int i = 0; i < X_Dim; i++)
	for(int j = 0; j < Y_Dim; j++)
	for(int k = 0; k < Z_Dim; k++)
	{
		float point[3];
		point[0] = i * SpaceGranularity + (-SCENESIZE);
		point[1] = j * SpaceGranularity + (SCENEHEIGHT);
		point[2] = k * SpaceGranularity + (-SCENESIZE);

		if((pMat + i * Y_Dim * Z_Dim + j * Z_Dim + k)->bOccupied)
		{
			glColor3f(1, 0, 0);
			glVertex3fv( point );
		}
		else if((pMat + i * Y_Dim * Z_Dim + j * Z_Dim + k)->nBudId != -1)
		{
			glColor3f(0, 1, 0);
			glVertex3fv( point );
		}
	}
	glEnd();
	glEnable(GL_LIGHTING);	
}

void Domain::Init()
{
//#if GPU_SPEED_UP
//#else
//	//	Cell Matrix Initialization
//	for(int i = 0; i < X_Dim; i++)
//	for(int j = 0; j < Y_Dim; j++)
//	for(int k = 0; k < Z_Dim; k++)
//	{
//		mat[i][j][k].bOccupied = false;
//		mat[i][j][k].fIllum = 1;
//	}
//#endif
	//	original shadow
	unsigned shadowSize = GroundDim * GroundDim * sizeof(MARK);
	h_ground_shadow = (MARK *)malloc(shadowSize);
	cudaMalloc((void **)&d_ground_shadow, shadowSize );
	
	cudaMemset(d_shadow, 0, pow(SCENESIZE*2/SpaceGranularity, 2));

	//	smoothed
	unsigned smoothedShadowSize = pow(SCENESIZE * 2 / SMOOTH_GRAN, 2) * sizeof(MARK);
	h_smoothed_shadow = (MARK *)malloc(smoothedShadowSize);
	cudaMalloc((void **)&d_smoothed_shadow, smoothedShadowSize);
	cudaError_t eid = cudaGetLastError();
	if(eid != cudaSuccess)
	{
		const char *errStr = cudaGetErrorString(eid);
		printf("\n%s\n", errStr);
	}
	cudaMemset(d_smoothed_shadow, 0, smoothedShadowSize);

#if 0
	Internode iNode1;
	makeInternode( iNode1,
				   Vect3d(-2, 8, 0), Vect3d(1, 1, 0), 
				   Vect3d(0, 1, 0), true, true);
	addInternode( iNode1);
	
	Internode iNode2;
	makeInternode( iNode2,
				   Vect3d(1, 8, 0), Vect3d(-1, 1, 0), 
				   Vect3d(0, 1, 0));
	addInternode( iNode2 );

#else

	//	Node
	Internode iNode;
	makeInternode( iNode, -1, 
				   Vect3d(0, 0, 0), Vect3d(0, 2, 0), 
				   Vect3d(0, 2, 0));
	addInternode( iNode );

	//	Barriers
	barriers[0].eType = SPHERE;
	barriers[0].fSize = 1;
	float barPos[3] = {1.5, 6, 0};
	barriers[0].center[0] = barPos[0];
	barriers[0].center[1] = barPos[1];
	barriers[0].center[2] = barPos[2];
	nBarrierCount = 1;

	printf("Processing Barriers... ");
	copyBarriersToGPU();
	gpu_process_barriers<<<BLOCK_DIM_1D, THREAD_DIM_1D>>>(d_barriers, 1, d_mat);
	printf("Done\n");

#endif
}

void Domain::Destroy()
{
	free(h_ground_shadow);
	cudaFree(d_ground_shadow);

	free(h_smoothed_shadow);
	cudaFree(d_smoothed_shadow);
}


//	for performance consideration
void Domain::gpu_SpaceColonization()
{
	copy_internodes();

	cudaMemcpy(	d_internodes,
				h_internodes,
				tree.internodes.size() * sizeof(InternodeDelegate),
				cudaMemcpyHostToDevice);

	//	1. Process Voxels (Space & Lights)
	printf("-- processing voxels..");
	space_colonization_on_voxels<<<BLOCK_DIM_1D, THREAD_DIM_1D>>>(d_internodes, tree.internodes.size());
	cudaThreadSynchronize();	
	cudaError_t eid = cudaGetLastError();
	if(eid != cudaSuccess)
	{
		const char *errStr = cudaGetErrorString(eid);
		printf("\n%s\n", errStr);
	}
	printf(" Done\n");

	//	2. Generate optimized vectors by #1
	printf("-- calculating optimized vectors..");
	get_optimized_bud_vector<<<BLOCK_DIM_1D, THREAD_DIM_1D>>>(d_internodes, d_optVec);
	cudaMemcpy(optVecs, d_optVec, sizeof(struct OptVec)*MAX_INTERNODES, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();	
	eid = cudaGetLastError();
	if(eid != cudaSuccess)
	{
		const char *errStr = cudaGetErrorString(eid);
		printf("\n%s\n", errStr);
	}

	//	Normalize All the vectors got
	for(size_t i = 0; i< tree.internodes.size(); i ++)
	{
		normalize(optVecs[i].vec);
	}
	printf(" Done\n");

}

std::vector<size_t> vOldInx;

void Domain::grow()
{
	//std::vector<Internode> &rNodes = tree.internodes;

	size_t nNodeNum = tree.internodes.size();
	printf("--node num: %d\n", nNodeNum);
	if(nNodeNum == 0)
	{
		printf("~~~~~No internodes, dude~~~~~\n");
		return;
	}

	///	Calculate Light factor here
	///
	float sun_vec[] = SUN_VEC;	

	for(size_t i = 0; i< nNodeNum; i++)
	{
		//	Processed already?
		if(std::find (vOldInx.begin(), vOldInx.end(), i) != vOldInx.end())
		{
			//printf("~~ %d processed already\n", i);
			continue;
		}

		//	Root Internode?
		if(tree.internodes[i].ancestorId == -1)
		{
			//printf("~~ It's root.\n");
			vOldInx.push_back(i);
			continue;
		}	

		//	Find the counter Internode who shares the same parent
		//	NOTE: since in my simplified model, one node only has
		//		  one anxillary bud, so only other one will match.
		size_t i2 = 0xFFFF;
		for(size_t j = 0; j< nNodeNum; j++)
		{
			if(std::find (vOldInx.begin(), vOldInx.end(), j) != vOldInx.end())
			{
				//printf("~~ %d processed already\n", i);
				continue;
			}

			if( (tree.internodes[j].ancestorId == tree.internodes[i].ancestorId) && 
				(tree.internodes[j].id != tree.internodes[i].id) )
			{
				i2 = j;
				break;
			}			
		}

		//	Only itself?
		if(i2 == 0xFFFF)
		{
			if(tree.internodes[i].pTerminalBud != NULL)
			{
				//	Calculate Light received
				float light_rec = Bud::getLightReceived(tree.internodes[i].pTerminalBud->ort, sun_vec);
				tree.internodes[i].pTerminalBud->fLightFactor = light_rec;

				vOldInx.push_back(i);
				//printf("~~ %d is the only child\n", i);
				break;
			}
		}
		else
		{
			//	a) First Pass
			float light1 = 0, light2 = 0;
			if(tree.internodes[i].pTerminalBud != NULL)
			{
				//	Calculate Light received
				light1 = Bud::getLightReceived(tree.internodes[i].pTerminalBud->ort, sun_vec);			
				light1 = light1 > 0 ? light1 : 0;
			}
			if(tree.internodes[i2].pTerminalBud != NULL)
			{
				//	Calculate Light received
				light2 = Bud::getLightReceived(tree.internodes[i].pTerminalBud->ort, sun_vec);	
				light2 = light2 > 0 ? light2 : 0;
			}

			//	b) Second Pass
			float totalLight = light1 + light2;
			if(totalLight == 0)
			{
				continue;
			}
			if(tree.internodes[i].bFromAnBud)
			{
				float total = LAMDA * light2 + (1 - LAMDA) * light1;
				tree.internodes[i].pTerminalBud->fLightFactor = (1- LAMDA)*light1 / total * totalLight;
				tree.internodes[i2].pTerminalBud->fLightFactor = LAMDA*light2 / total * totalLight;
			}
			else
			{
				float total = LAMDA * light1 + (1 - LAMDA) * light2;
				tree.internodes[i].pTerminalBud->fLightFactor = LAMDA * light2 / total;
				tree.internodes[i2].pTerminalBud->fLightFactor = (1 - LAMDA) * light1 / total;
			}

			vOldInx.push_back(i);
			vOldInx.push_back(i2);

			//printf("~~ Pair %d and %d processed\n", i, i2);
		}

		//// NOTE: as a much simplified model, only terminal bud 
		////	   are considered for signals
		////
		//if(tree.internodes[i].pAnxillaryBud != NULL)
		//{
		//	//	Calculate Light received
		//	float light_rec = Bud::getLightReceived(tree.internodes[i].pAnxillaryBud->ort, sun_vec);
		//	printf(" %d A-Bud : %f\n", i, light_rec);//	Calculate Light received

		//}
	}

	///	Make Growth here
	///

	Vect3d vLight( -sun_vec[0], -sun_vec[1], -sun_vec[2]);

	{
	for(size_t i = 0; i< nNodeNum; i++)
	{

		//	Gravity Effects
		Vect3d vGrav(0, -0.01, 0);
		//float fGravityFactor = pow(1.6f, tree.internodes[i].nGrowthStep*1.f)/2;
		float fGravityFactor = 0.3f * tree.internodes[i].nGrowthStep;

		//	1. Terminal Bud
		float fLightWeight = 0.02;
		if(tree.internodes[i].pTerminalBud != NULL)
		{
			unsigned nBudId = tree.internodes[i].pTerminalBud->nId;
			OptVec &rOptVec = optVecs[nBudId];			

			//	Current Internode Light Factor
			float fLightFactor = tree.internodes[i].pTerminalBud->fLightFactor * fLightWeight;
			
			//	a. Add new Internode
			Internode iNode;
			Vect3d optVec(rOptVec.vec[0], rOptVec.vec[1], rOptVec.vec[2]);
			makeInternode(iNode, tree.internodes[i].id, 
						 tree.internodes[i].pTerminalBud->pos, 
						 optVec + vGrav * fGravityFactor + vLight * fLightFactor, 
						 optVec + vGrav * fGravityFactor + vLight * fLightFactor, 
						 false, true );
			iNode.nGrowthStep = GrownStep;
			iNode.bFromAnBud = false;
			addInternode(iNode);

			//	b. remove the terminal bud
			tree.internodes[i].pTerminalBud.reset(0);
		}

		//	2. Anxillary Bud
		if(tree.internodes[i].pAnxillaryBud != NULL)
		{				
			float fLightFactor = tree.internodes[i].pAnxillaryBud->fLightFactor * fLightWeight;

			//	a. Add new Internode
			Internode iNewNode;
			makeInternode( iNewNode, tree.internodes[i].id, 
						   tree.internodes[i].pAnxillaryBud->pos, 
						   tree.internodes[i].pAnxillaryBud->ort + vGrav * fGravityFactor + vLight * fLightFactor, 
						   tree.internodes[i].pAnxillaryBud->ort + vGrav * fGravityFactor + vLight * fLightFactor,
						   false, true);
			iNewNode.nGrowthStep = GrownStep;
			iNewNode.bFromAnBud = true;
			addInternode(iNewNode);

			//	b. remove this Anxillary Bud
			tree.internodes[i].pAnxillaryBud.reset(0);
		}
	}
	}
}

void Domain::RenderShadow()
{
	//GLfloat vSun[3] = {};

	//gpu_shadowGeneration<<<>>>(float vSunX, float vSunY, float vSunZ);
	//cudaMemcpy(,cudaMemcpyDeviceToHost);
	//cudeThreadSynchronize();

	//glBegin(GL_POINTS);
	//for(float i = -SCENESIZE; i < SCENESIZE; i+= )
	//for(float j = -SCENESIZE; j < SCENESIZE; j+= )
	//{
	//	glVertex3f();
	//}
	//glEnd();
}

void Domain::RenderBarriers()
{
	for(unsigned i = 0; i< nBarrierCount; i++)
	{
		barriers[i].Render();
	}
}

void Domain::CalcGroundShadow()
{
	//	reset shadow buffer
	reset_ground_shadow();

	//	get original shadow value
	gpu_calc_shadow<<<GroundDim, GroundDim>>>( d_ground_shadow, d_mat );
	cudaMemcpy(h_ground_shadow, d_ground_shadow, sizeof(MARK)*GroundDim*GroundDim, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	//	smooth the shadow
	unsigned nSDim = SCENESIZE * 2 / SMOOTH_GRAN;
	gpu_smooth_shadow<<<SMOOTH_BLOCK_DIM, SMOOTH_THREAD_DIM>>>( d_ground_shadow, d_smoothed_shadow);
	cudaThreadSynchronize();
	cudaMemcpy(h_smoothed_shadow, d_smoothed_shadow, sizeof(MARK) * nSDim * nSDim, cudaMemcpyDeviceToHost);
}

void Domain::RenderGroundShadow()
{
#if 0

	int nVerticalCount = (DOMAIN_HEIGHT - SCENEHEIGHT) / SpaceGranularity;
	int nHorizontalCount = (SCENESIZE * 2) / SpaceGranularity;

	//	Check Casting Lines
	glDisable(GL_LIGHTING);
	glColor3f(0, 1, 0);
	glBegin(GL_LINES);
	for(int xi = 0; xi < nHorizontalCount; xi += 2)
	for(int zi = 0; zi < nHorizontalCount; zi += 2)
	{
		//	Get From-Cell index
		int fromInx[3] = {xi, 0, zi};

		//	View Vec
		float sun_vec[3] = SUN_VEC;
		float view_vec[3] = { -sun_vec[0], -sun_vec[1], -sun_vec[2] };
		//assert(view_vec[1] >= 0);

		//	Call To-Cell index
		int toInx[3] = {0};
		toInx[1] = nVerticalCount - 1;
		//	WARNING: the X\Z index may be over the available cube.
		//			 this will be checked in the dda_ray_casting()
		//
		toInx[0] = xi + (nVerticalCount - 1) * (view_vec[0] * 1.f / view_vec[1]);
		toInx[2] = zi + (nVerticalCount - 1) * (view_vec[2] * 1.f / view_vec[1]);

		//	Draw Casting Line
		glVertex3f( SpaceGranularity / 2 + xi * SpaceGranularity - SCENESIZE, SCENEHEIGHT, 
					SpaceGranularity / 2 + zi * SpaceGranularity - SCENESIZE);
		glVertex3f( SpaceGranularity / 2 + toInx[0] * SpaceGranularity - SCENESIZE, SCENEHEIGHT + DOMAIN_HEIGHT , 
					SpaceGranularity / 2 + toInx[2] * SpaceGranularity - SCENESIZE);
	}
	glEnd();
	glEnable(GL_LIGHTING);

#endif

#if 0
	//	render.
	float y = SCENEHEIGHT + 0.01;
	glDisable(GL_LIGHTING);
	glBegin(GL_QUADS);		
		for(int i = 0; i < GroundDim; i ++)
		for(int j = 0; j < GroundDim; j ++)
		{
			float fColor = 1 - *(h_ground_shadow + i * GroundDim + j);
			glColor3f(fColor, fColor, fColor);
			glVertex3f( -SCENESIZE + i * SpaceGranularity, y, -SCENESIZE + j * SpaceGranularity); 			
			glVertex3f( -SCENESIZE + i * SpaceGranularity, y, -SCENESIZE + (j + 1) * SpaceGranularity); 
			glVertex3f( -SCENESIZE + (i + 1) * SpaceGranularity, y, -SCENESIZE + (j+1) * SpaceGranularity); 
			glVertex3f( -SCENESIZE + (i + 1) * SpaceGranularity, y, -SCENESIZE + (j) * SpaceGranularity); 
		}
	glEnd();
	glEnable(GL_LIGHTING);

#else
	//	Render smoothed shadow
	unsigned nSmoothDim = SCENESIZE * 2 / SMOOTH_GRAN;
	float y = SCENEHEIGHT + 0.01;
	glDisable(GL_LIGHTING);
	glBegin(GL_QUADS);		
		for(int i = 0; i < nSmoothDim; i ++)
		for(int j = 0; j < nSmoothDim; j ++)
		{
			float fColor = 1 - *(h_smoothed_shadow + i * nSmoothDim + j);
			//if(fColor < 1)
			{
				glColor3f(fColor, fColor, fColor);
				glVertex3f( -SCENESIZE + i * SMOOTH_GRAN, y, -SCENESIZE + j * SMOOTH_GRAN); 			
				glVertex3f( -SCENESIZE + i * SMOOTH_GRAN, y, -SCENESIZE + (j + 1) * SMOOTH_GRAN); 
				glVertex3f( -SCENESIZE + (i + 1) * SMOOTH_GRAN, y, -SCENESIZE + (j + 1) * SMOOTH_GRAN); 
				glVertex3f( -SCENESIZE + (i + 1) * SMOOTH_GRAN, y, -SCENESIZE + (j) *SMOOTH_GRAN); 
			}
		}
	glEnd();
	glEnable(GL_LIGHTING);

#endif
}

