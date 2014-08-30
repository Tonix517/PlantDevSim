#ifndef LIGHT_H
#define LIGHT_H

class Light
{
public:
	
	Light();
	
	void Set(float *pAmb, float *pDiff, float *pSpec, float *pPos);
	void Put();

private:
	float light_ambient[4];
	float light_diffuse[4];
	float light_specular[4];
	float light_position[4];

};

#endif