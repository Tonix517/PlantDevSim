#ifndef SCENE_H
#define SCENE_H

class Scene
{
public:

	void Render()
	{
		_RenderSky();
		_RenderGround();
	}

private:

	void _RenderSky();
	void _RenderGround();
};

#endif