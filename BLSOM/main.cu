#include<iostream>
#include"BLSOM.h"
#include"SelectGPU.h"

#define VEC_DIM 3
#define MAP_WIDTH 10
#define MAP_HEIGHT 20

int main(int argc, char** argv) {
	int device;
	int vec_dim = VEC_DIM;
	int map_width = MAP_WIDTH;
	int map_height = MAP_HEIGHT;
	float map_wight[MAP_WIDTH*MAP_HEIGHT] = {};
	float ave_vec[VEC_DIM] = {};
	float rot1[VEC_DIM] = {};
	float rot2[VEC_DIM] = {};
	float sdev1 = 3;
	float sdev2 = 2;

	//std::shared_ptr<float> ptr;
	//ptr = cuda_shared_ptr<float>(10);
	/*--- Select GPU for BLSOM ---*/
	//SelectGPU(&device);

	/*--- make BLSOM Object ---*/
	//BLSOM test = BLSOM( vec_dim, map_width, map_height, device, true );
	BLSOM test = BLSOM(vec_dim, map_width);


	test.Init(sdev1, sdev2, rot1, rot2, ave_vec);
	test.check_mapWeight();

	/*--- Select GPU for BLSOM ---*/
	SelectGPU(&device);

	//test.GetMapWeight
	
	return 0;
}