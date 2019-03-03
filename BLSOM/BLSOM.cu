
#include "BLSOM.h"
#include "SelectGPU.h"

using namespace std;

#ifndef DIST
#define DIST(bx,by,x,y) ((bx-x)*(bx-x)+(by-y)*(by-y))
#endif // !DIST


#ifndef MAX
#define MAX( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

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
		this->d_weightS = thrust::device_vector<float>(map_width*map_height* (vec_dim + 1));
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
	this->h_weightS = thrust::host_vector<float>(this->map_width*this->map_height* (this->vec_dim + 1));
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
	
	memcpy(thrust::raw_pointer_cast(this->h_trains.data()), train, epoc_num*train_num*this->vec_dim);
	cudaMemcpy(thrust::raw_pointer_cast(this->d_trains.data()), thrust::raw_pointer_cast(this->h_trains.data()), epoc_num*train_num*this->vec_dim, cudaMemcpyHostToDevice);
}

void BLSOM::check_mapWeight() {
	cudaMemcpy(thrust::raw_pointer_cast(this->h_mapWeight.data()), thrust::raw_pointer_cast(this->d_mapWeight.data()), sizeof(float)*this->map_width*this->map_height*this->vec_dim, cudaMemcpyDeviceToHost);

	for (int idy = 0; idy < map_height; idy++) {
		for (int idx = 0; idx < map_width; idx++) {
			//printf("%d %d \n",idy, idx);
			printf("%d", map_width*idy + idx);
			//printf("%d", map_width*vec_dim*idy + vec_dim*idx);
			/*
			for (int idz = 0; idz < vec_dim; idz++) {
				printf("%d :", map_width*vec_dim*idy + vec_dim*idx + idz);
				printf("%f ", this->h_mapWeight[map_width*vec_dim*idy + vec_dim*idx + idz]);
				printf("\n");
			}*/
			printf("\n");
		}
	}
}

__global__ void InitMapWeightFromGPU(float* mapWeight,const int map_width, const int vec_dim) {
	int ix = blockIdx.x*blockDim.x;
	int iy = blockIdx.y*blockDim.y;
	int idx = map_width*vec_dim*iy + vec_dim*ix + threadIdx.z;
	mapWeight[idx] = 0;
}

void BLSOM::InitMapWeight() {
	dim3 block(1,1,this->vec_dim);
	dim3 grid(this->map_width, this->map_height);

	InitMapWeightFromGPU <<<grid, block >>> (thrust::raw_pointer_cast(this->d_mapWeight.data()),this->map_width,this->vec_dim);
}

__global__ void InitNodeFromGPU(float* node,const int map_width) {
	int ix = blockIdx.x*blockDim.x;
	int iy = blockIdx.y*blockDim.y;
	int idx = map_width*iy + ix;
	node[idx] = 0;
}

__global__ void BMUFromGPU(float* input_xk, float* node, float* mapWeight, const int map_width, const int vec_dim) {
	int ix = blockIdx.x*blockDim.x;
	int iy = blockIdx.y*blockDim.y;
	int node_idx = map_width*iy + ix;
	int map_idx = map_width*vec_dim*iy + vec_dim*ix + threadIdx.z;
	
	node[node_idx] += (mapWeight[map_idx]-input_xk[threadIdx.z])*(mapWeight[map_idx] - input_xk[threadIdx.z]);
	
}

int BLSOM::getBMUIndex() {
	thrust::device_vector<float>::iterator bgn_itr = d_node.begin();
	thrust::device_vector<float>::iterator bmu_itr = thrust::min_element(thrust::device, d_node.begin(), d_node.end());
	return thrust::distance(bgn_itr, bmu_itr);
}

void BLSOM::setBMUPosition() {
	int bmu_index = getBMUIndex();
	this->h_bmuPos[0] = bmu_index % (this->map_width);	//xÀ•WŒvŽZ
	this->h_bmuPos[1] = bmu_index / (this->map_width);	//yÀ•WŒvŽZ
	this->d_bmuPos = this->h_bmuPos;
}

__global__ void CalcWeightSFromGPU(float* input_xk, int* bmuPos, float* weightS,
								   const int map_width, const int vec_dim,
								   const double iBeta, const double tBeta, const int lnum) {

	int ix = blockIdx.x*blockDim.x;
	int iy = blockIdx.y*blockDim.y;
	int weiS_idx = map_width*vec_dim*iy + vec_dim*ix + threadIdx.z;
	int weiS_eIdx = map_width*vec_dim*iy + vec_dim*ix + vec_dim;	//weightS[ix][iy][vec_dim]


	float dist = DIST(bmuPos[0], bmuPos[1], ix, iy);
	float Beta = MAX(0, (iBeta*(1 - (lnum / tBeta))));

	if ((Beta*Beta - dist) >= 0) {
		weightS[weiS_idx] += input_xk[threadIdx.z];
		weightS[weiS_eIdx]++;
	}

}

__global__ void UpdateMapWeightFromGPU(float* mapWeight, float* weightS,
									   const int map_width, const int vec_dim,
									   const double iAlfa, const double tAlfa, const int lnum) {
	int ix = blockIdx.x*blockDim.x;
	int iy = blockIdx.y*blockDim.y;
	int map_idx = map_width*vec_dim*iy + vec_dim*ix + threadIdx.z;
	int weiS_eIdx = map_width*vec_dim*iy + vec_dim*ix + vec_dim;	//weightS[ix][iy][vec_dim]

	float alfaFunc = MAX(0.01, (iAlfa*(1.0 - (lnum / tAlfa))));

	if (weightS[weiS_eIdx] > 0) {
		weightS[map_idx] /= weightS[weiS_eIdx];
		weightS[map_idx] -= mapWeight[map_idx];
		weightS[map_idx] *= alfaFunc;
		mapWeight[map_idx] += weightS[map_idx];
	}
}

void BLSOM::BMU(float* input_xk) {
	dim3 block(1, 1, this->vec_dim);
	dim3 grid(this->map_height, this->map_width);

	InitNodeFromGPU <<< grid, 1 >>> (thrust::raw_pointer_cast(this->d_node.data()),this->map_width);
	BMUFromGPU <<< grid,block >>>(input_xk, thrust::raw_pointer_cast(this->d_node.data()), thrust::raw_pointer_cast(this->d_mapWeight.data()), this->map_width, this->vec_dim);
	setBMUPosition();
	
}

void BLSOM::CalcWeightS(float* input_xk, int Lnum) {
	dim3 block(1, 1, this->vec_dim);
	dim3 grid(this->map_height, this->map_width);

	CalcWeightSFromGPU <<<grid, block >>> (input_xk,
										   thrust::raw_pointer_cast(this->d_bmuPos.data()),
										   thrust::raw_pointer_cast(this->d_weightS.data()),
										   this->map_width,
										   this->vec_dim,
										   this->iBeta,
										   this->t_beta,
										   Lnum);
											
}

void BLSOM::UpdateMapWeight(int Lnum) {
	dim3 block(1, 1, this->vec_dim);
	dim3 grid(this->map_height, this->map_width);

	UpdateMapWeightFromGPU <<<grid,block>>> (thrust::raw_pointer_cast(this->d_mapWeight.data()),
											 thrust::raw_pointer_cast(this->d_weightS.data()),
											 this->map_width,
											 this->vec_dim,
											 this->iAlfa,
											 this->t_alfa,
											 Lnum);
}

__global__ void InitWeighSFromGPU(float* weightS) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	weightS[idx] = 0;
}

void BLSOM::Learning(int Lnum) {
	std::cout << "Learning Start" << std::endl;

	dim3 weightS_block(this->vec_dim + 1);
	dim3 weightS_grid(this->map_height*this->map_width);

	for (int l = 0; l < Lnum; l++) {
		for (int i = 0; i < this->epoc_num; i++) {
			InitWeighSFromGPU <<< weightS_grid, weightS_block >>> (thrust::raw_pointer_cast(this->d_weightS.data()));

			for (int j = 0; j < this->train_num; j++) {
				this->BMU(thrust::raw_pointer_cast(&(this->d_trains[i * (this->train_num) * (this->vec_dim) + j*(this->vec_dim)]))); //“Y‚¦Žš‚ðC³
				this->CalcWeightS(thrust::raw_pointer_cast(&(this->d_trains[i * (this->train_num) * (this->vec_dim) + j*(this->vec_dim)])), l);
			}
			this->UpdateMapWeight(l);
		}
	}

	std::cout << "Learning Finish" << std::endl;
}
