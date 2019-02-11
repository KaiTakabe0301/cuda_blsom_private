
#include "BLSOM.h"
#include "SelectGPU.h"

using namespace std;

#define CHECK(call)														\
{																		\
	const cudaError_t error = call;										\
	if(error!=cudaSuccess){												\
		printf("Error %s:%d \t",__FILE__,__LINE__);						\
		printf("code:%d, reason:%s\n",error,cudaGetErrorString(error));	\
		exit(1);														\
	}																	\
}		

bool checkAllocatedMemory(void* pointer) {
	if (pointer != NULL) {
		return true;
	}
	else {
		return false;
	}
}

BLSOM::BLSOM(int vec_dim, int map_width) :iAlfa(0.5), iBeta(40), t_alfa(30), t_beta(20),
										  vec_dim(vec_dim), map_width(map_width), flg_gpu(true) {
	int device;
	
	this->map_height = 0;
	CHECK(SelectBestGPU(&device));

	if (flg_gpu) {
		CHECK(cudaSetDevice(device));
	}
}

BLSOM::BLSOM(int vec_dim, int map_width, int map_height) :iAlfa(0.5), iBeta(40), t_alfa(30), t_beta(20),
														  vec_dim(vec_dim), map_width(map_width), map_height(map_height), flg_gpu(true) {
	int device;
	CHECK(SelectBestGPU(&device));

	if (flg_gpu) {
		CHECK(cudaSetDevice(device));
	}
}

BLSOM::BLSOM(int vec_dim, int map_width, int map_height,int device):iAlfa(0.5), iBeta(40), t_alfa(30), t_beta(20), 
																    vec_dim(vec_dim), map_width(map_width), map_height(map_height),flg_gpu(true) {
	if (flg_gpu) {
		CHECK(cudaSetDevice(device));
	}
}

BLSOM::BLSOM(int vec_dim, int map_width, int map_height, int device, int gpuFlag) : iAlfa(0.5), iBeta(40), t_alfa(30), t_beta(20), 
																					vec_dim(vec_dim),map_width(map_width),map_height(map_height),flg_gpu(gpuFlag) {
	
	if (gpuFlag) {
		CHECK(cudaSetDevice(device));
	}
}

BLSOM::~BLSOM() {
	
}

void BLSOM::Init(const float sdev1, const float sdev2, const float* rot1, const float* rot2, const float *aveVec) {

	if (map_height == 0) {
		this->map_height = (sdev2 / sdev1)*this->map_width;
	}

	if (flg_gpu) {
		
		this->d_mapWeight = thrust::device_vector<float>(map_width*map_height*vec_dim);
		this->d_node = thrust::device_vector<float>(map_width*map_height);
		this->d_rot1 = thrust::device_vector<float>(vec_dim);
		this->d_rot2 = thrust::device_vector<float>(vec_dim);
		this->d_aveVec = thrust::device_vector<float>(vec_dim);
		this->d_train = thrust::device_vector<float>(vec_dim);
		this->d_sdev = thrust::device_vector<float>(2);
		this->d_bmuPos = thrust::device_vector<int>(2);


		cudaMemcpy(thrust::raw_pointer_cast(this->d_rot1.data()), rot1, this->vec_dim * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(thrust::raw_pointer_cast(this->d_rot2.data()), rot2, this->vec_dim * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(thrust::raw_pointer_cast(this->d_aveVec.data()), aveVec, this->vec_dim * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(thrust::raw_pointer_cast(this->d_sdev.data()), &sdev1, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(thrust::raw_pointer_cast(this->d_sdev.data()+1), &sdev2, sizeof(float), cudaMemcpyHostToDevice);
		
	}

	this->h_mapWeight = thrust::host_vector<float>(this->map_width*this->map_height*this->vec_dim);
	this->h_node = thrust::host_vector<float>(this->map_width*this->map_height);
	this->h_rot1 = thrust::host_vector<float>(this->vec_dim);
	this->h_rot2 = thrust::host_vector<float>(this->vec_dim);
	this->h_aveVec = thrust::host_vector<float>(this->vec_dim);
	this->h_train = thrust::host_vector<float>(this->vec_dim);
	this->h_sdev = thrust::host_vector<float>(2);
	this->h_bmuPos = thrust::host_vector<int>(2);

	memcpy(thrust::raw_pointer_cast(this->h_rot1.data()), rot1, this->vec_dim * sizeof(float));
	memcpy(thrust::raw_pointer_cast(this->h_rot2.data()), rot2, this->vec_dim * sizeof(float));
	memcpy(thrust::raw_pointer_cast(this->h_aveVec.data()), aveVec, this->vec_dim * sizeof(float));
	memcpy(thrust::raw_pointer_cast(this->h_sdev.data()), &sdev1, sizeof(float));
	memcpy(thrust::raw_pointer_cast(this->h_sdev.data()+1), &sdev2, sizeof(float));

	InitMapWeight();
}

void BLSOM::SetTrainingData(const float* train, const int train_num, const int epoc_num=1) {
	this->epoc_num = epoc_num;
	this->train_num = train_num;

	this->h_trains = thrust::host_vector<float>(epoc_num*train_num*this->vec_dim);
	this->d_trains = thrust::device_vector<float>(epoc_num*train_num*this->vec_dim);
	//this->h_trains = (float*)malloc(sizeof(float)*epoc_num*train_num*this->vec_dim);
	//cudaMalloc(&(this->d_trains), sizeof(float)*epoc_num*train_num*this->vec_dim);
	
	memcpy(thrust::raw_pointer_cast(this->h_trains.data()), train, epoc_num*train_num*this->vec_dim);
	cudaMemcpy(thrust::raw_pointer_cast(this->d_trains.data()), thrust::raw_pointer_cast(this->h_trains.data()), epoc_num*train_num*this->vec_dim, cudaMemcpyHostToDevice);

	//memcpy(this->h_trains, train, sizeof(float)*epoc_num*train_num*this->vec_dim);
	//cudaMemcpy(d_trains, h_trains, sizeof(float)*epoc_num*train_num*this->vec_dim, cudaMemcpyHostToDevice);
}

void BLSOM::check_mapWeight() {
	cudaMemcpy(thrust::raw_pointer_cast(this->h_mapWeight.data()), thrust::raw_pointer_cast(this->d_mapWeight.data()), sizeof(float)*this->map_width*this->map_height*this->vec_dim, cudaMemcpyDeviceToHost);

	for (int idx = 0; idx < map_width; idx++) {
		for (int idy = 0; idy < map_height; idy++) {
			printf("%d %d ",idx, idy);
			for (int idz = 0; idz < vec_dim; idz++) {
				printf("%f ", this->h_mapWeight[idx*idy + idz]);
			}
			printf("\n");
		}
	}
}

__global__ void InitMapWeightFromGPU(float* mapWeight) {
	int idx =  blockIdx.x*blockDim.x + threadIdx.x;
	mapWeight[idx] = 1;
}

void BLSOM::InitMapWeight() {
	dim3 block(this->vec_dim);
	dim3 grid(this->map_height*this->map_width);

	InitMapWeightFromGPU <<<grid, block >>> (thrust::raw_pointer_cast(this->d_mapWeight.data()));
}
