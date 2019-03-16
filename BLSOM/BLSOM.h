#pragma once

#include<iostream>
#include<string>
#include<cmath>
#include<vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include<thrust\device_vector.h>
#include<thrust\host_vector.h>
#include<thrust\extrema.h>
#include <thrust/execution_policy.h>
#include<memory>

class BLSOM {
private:

	/*--- �}�b�v�̊�{��� ---*/
	int epoc_num;	//�G�|�b�N��
	int train_num;	//�w�K�f�[�^�̐�
	int vec_dim;	//�Ώۃf�[�^�̎���
	int map_width;	//�}�b�v�̉���I
	int map_height;	//�}�b�v�̍���J

	/*--- �w�K�W���ƋߖT�֐���parameter ---*/
	double t_alfa;	//�w�K�W��
	double t_beta;	//�ߖT�֐�
	double iAlfa;	//�����l��
	double iBeta;	//�����l��

	/*--- GPU�p�v�Z�ϐ��������� ---*/
	///BLSOM�ɗ��p
	bool flg_gpu;								//GPU���g�p���邩�ۂ�
	int   d_Acs;							//�g�p����GPU
	thrust::device_vector<float> d_mapWeight;						//J�~I�s��̃}�b�v�̗̈�m�ۂɗ��p
	thrust::device_vector<float> d_weightS;							//��\�x�N�g��Wij 
	thrust::device_vector<float> d_node;							//node
	thrust::device_vector<int> d_bmuPos;							//d_bmuPos[0]=x,d_bmuPos[1]=y
	thrust::device_vector<float> d_train;							//�w�K�f�[�^
	thrust::device_vector<float> d_trains;						//trains[�G�|�b�N��][���̓f�[�^��](�f�[�^�̎�����)
	thrust::device_vector<float> d_sdev;							//�W���΍���1(d_sdev[0]),��2(d_sdev[1])
	thrust::device_vector<float> d_rot1, d_rot2;					//���/���rotation
	thrust::device_vector<float> d_aveVec;						//���σx�N�g��
	thrust::device_vector<float> d_umat;							
	/*--- GPU�p�v�Z�ϐ������܂� ---*/

	/*--- CPU�p�v�Z�ϐ��������� ---*/
	thrust::host_vector<float> h_mapWeight;						//J�~I�s��̃}�b�v�̗̈�m�ۂɗ��p
	thrust::host_vector<float> h_weightS;						//��\�x�N�g��Wij 
	thrust::host_vector<float> h_node;							//node
	thrust::host_vector<int> h_bmuPos;							//d_bmuPos[0]=x,d_bmuPos[1]=y
	thrust::host_vector<float> h_train;							//�w�K�f�[�^
	thrust::host_vector<float> h_trains;						//trains[�G�|�b�N��][���̓f�[�^��](�f�[�^�̎�����)
	thrust::host_vector<float> h_sdev;							//�W���΍���1(d_sdev[0]),��2(d_sdev[1])
	thrust::host_vector<float> h_rot1, h_rot2;					//���/���rotation
	thrust::host_vector<float> h_aveVec;							//���σx�N�g��
	/*--- CPU�p�v�Z�ϐ������܂� ---*/
	
	int getBMUIndex();		//
	void setBMUPosition();	//

	float dist();	//���[�N���b�h�����̌v�Z
	void BMU(float* input_xk);
	void CalcWeightS(float* input_xk, int Lnum);
	void UpdateMapWeight(int Lnum);

	/*--- GPU���p�֐� ---*/
	void  InitMapWeight();	//�����}�b�v�̍쐬
	void  InitRandWeightFromGPU();//�����_���ɏ��������s��
	void  searchBMUFromGPU(int epoc_num,int data_size);		//epoc_num * data_size + vec_dim�ō��W������




public:
	BLSOM(int vec_dim, int map_width);
	BLSOM(int vec_dim, int map_width, int map_height);
	BLSOM(int vec_dim, int map_width, int map_height, int device);
	BLSOM(int vec_dim, int map_width, int map_height, int device,int gpuFlag);
	~BLSOM();
	void Init(const float sdev1, const float sdev2, const float* rot1, const float* rot2, const float *aveVec);
	
	void Learning(int Lnum);
	void Classification();	///Learning���s��A�G�|�b�N���̍ŋߖT�m�[�h������U��

	float* GetSOMMap();

	/*--- �v�Z�ɕK�v�ȕϐ����i�[����֐��������� ---*/
	void SetHyperParameter(double initAlfa, double initBeta, double timeAlfa, double timeBeta);		///initAlfa=�w�K�W�� initBeta�w�K=���a timeAlfa=���萔 timeBeta=���萔
	void SetStandardDeviation(float sdev1, float sdev2);
	void SetRotation(float* rot1, float *rot2);
	void SetAverageVecter(float *aveVec);
	void SetTrainingData(const float* train,const int train_num, const int epoc_num);
	/*--- �v�Z�ɕK�v�ȕϐ����i�[����֐������܂� ---*/

	void Test();

	/*--- getter ---*/
	int MapHeight() {
		return this->map_height;
	}
	int MapWidth() {
		return this->map_width;
	}
	int VectorDim() {
		return this->vec_dim;
	}

	/*--- setter ---*/
	void SetMapHeight(int height) {
		this->map_height = height;
	}
	void SetMapWidth(int width) {
		this->map_width = width;
	}
	void SetVecDim(int vecDim) {
		this->vec_dim = vecDim;
	}
	std::vector<std::vector<std::vector<double> > > GetMapWeight();
	std::vector<std::vector<double> >GetUMatrix();


	void check_mapWeight();
};