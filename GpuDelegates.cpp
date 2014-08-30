#include "GpuDelegates.h"
#include "common.h"

InternodeDelegate *d_internodes;
InternodeDelegate *h_internodes;

void copy_internodes()
{
	
	for(size_t i = 0; i < tree.internodes.size(); i++)
	{
		h_internodes[i].pos[0] = tree.internodes[i].pos.x();
		h_internodes[i].pos[1] = tree.internodes[i].pos.y();
		h_internodes[i].pos[2] = tree.internodes[i].pos.z();

		h_internodes[i].vec[0] = tree.internodes[i].vec.x();
		h_internodes[i].vec[1] = tree.internodes[i].vec.y();
		h_internodes[i].vec[2] = tree.internodes[i].vec.z();
		
		if(tree.internodes[i].pAnxillaryBud != NULL)
		{
			h_internodes[i].anBudPos[0] = tree.internodes[i].pos.x();
			h_internodes[i].anBudPos[1] = tree.internodes[i].pos.y();
			h_internodes[i].anBudPos[2] = tree.internodes[i].pos.z();
		}
		else
		{
			h_internodes[i].anBudPos[0] = INVALID_POS;
			h_internodes[i].anBudPos[1] = INVALID_POS;
			h_internodes[i].anBudPos[2] = INVALID_POS;	
		}

		if(tree.internodes[i].pLeaf != NULL)
		{
			h_internodes[i].leafPos[0] = tree.internodes[i].pLeaf->pos.x();
			h_internodes[i].leafPos[1] = tree.internodes[i].pLeaf->pos.y();
			h_internodes[i].leafPos[2] = tree.internodes[i].pLeaf->pos.z();
		}
		else
		{
			h_internodes[i].leafPos[0] = INVALID_POS;
			h_internodes[i].leafPos[1] = INVALID_POS;
			h_internodes[i].leafPos[2] = INVALID_POS;
		}

		if(tree.internodes[i].pTerminalBud != NULL)
		{
			h_internodes[i].budPos[0] =tree.internodes[i].pTerminalBud->pos.x();
			h_internodes[i].budPos[1] = tree.internodes[i].pTerminalBud->pos.y();
			h_internodes[i].budPos[2] = tree.internodes[i].pTerminalBud->pos.z();

			h_internodes[i].budOrt[0] =tree.internodes[i].pTerminalBud->ort.x();
			h_internodes[i].budOrt[1] = tree.internodes[i].pTerminalBud->ort.y();
			h_internodes[i].budOrt[2] = tree.internodes[i].pTerminalBud->ort.z();
		}		
		else
		{
			h_internodes[i].budPos[0] = INVALID_POS;
			h_internodes[i].budPos[1] = INVALID_POS;
			h_internodes[i].budPos[2] = INVALID_POS;

			h_internodes[i].budOrt[0] = INVALID_POS;
			h_internodes[i].budOrt[1] = INVALID_POS;
			h_internodes[i].budOrt[2] = INVALID_POS;
		}
	}

}