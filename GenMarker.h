#ifndef GENMARKER_H
#define GENMARKER_H

//	it records which generation it belongs to
struct GenMarker
{
	GenMarker() 
		: nGenNum(0)
	{ }

	unsigned nGenNum;
};

#endif