//==============================================================================
//
//  Copyright (c) 2015-2018 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
//
// This file contains an example application that loads and executes a neural
// network using the SNPE C++ API and saves the layer output to a file.
// Inputs to and outputs from the network are conveyed in binary form as single
// precision floating point values.
//

#include <iostream>
#include <getopt.h>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <string>
#include <iterator>
#include <unordered_map>
#include <numeric>

//#include "main.hpp"
#include "CheckRuntime.hpp"
#include "LoadContainer.hpp"
#include "SetBuilderOptions.hpp"
#include "LoadInputTensor.hpp"
#include "udlExample.hpp"
#include "CreateUserBuffer.hpp"
#include "PreprocessInput.hpp"
#include "SaveOutputTensor.hpp"
#include "Util.hpp"
#ifdef ANDROID
#include <GLES2/gl2.h>
#include "CreateGLBuffer.hpp"
#endif

#include "DlSystem/UserBufferMap.hpp"
#include "DlSystem/UDLFunc.hpp"
#include "DlSystem/IUserBuffer.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPE.hpp"
#include "DiagLog/IDiagLog.hpp"

#include <ctime>
#include <sys/time.h> 
#include <unistd.h> 
#include <sys/syscall.h> 
#include <dirent.h>
#include <algorithm>
#include <thread>

using namespace std; 

const char* alexnet_inputFile = "/data/local/tmp/alexnet/target_raw_list_320.txt";
std::string alexnet_OutputDir = "/data/local/tmp/test/part_exper/alexnet_output_part";
std::string alexnet_layerPath = "/data/local/tmp/test/part_exper/alexnet_";

const char* vgg_inputFile = "/data/local/tmp/vgg/target_raw_list_20.txt";
std::string vgg_OutputDir = "/data/local/tmp/test/part_exper/vgg_output_part";
std::string vgg_layerPath = "/data/local/tmp/test/part_exper/vgg_";

const char* pos_inputFile = "/data/local/tmp/pos/pos_raw_list_500.txt";
std::string pos_OutputDir = "/data/local/tmp/test/part_exper/pos_output_part";
std::string pos_layerPath = "/data/local/tmp/test/part_exper/pos_";

const char* mnist_inputFile = "/data/local/tmp/mnist/target_list_320.txt";
std::string mnist_OutputDir = "/data/local/tmp/test/part_exper/mnist_output_part";
std::string mnist_layerPath = "/data/local/tmp/test/part_exper/mnist_";

const char* googlenet_inputFile = "/data/local/tmp/vgg/target_raw_list_160.txt"; 
std::string googlenet_OutputDir = "/data/local/tmp/test/part_exper/googlenet_output_part";
std::string googlenet_layerPath = "/data/local/tmp/test/part_exper/googlenet_";

const char* resnet_inputFile = "/data/local/tmp/vgg/target_raw_list_20.txt";
std::string resnet_OutputDir = "/data/local/tmp/test/part_exper/resnet_output_part";
std::string resnet_layerPath = "/data/local/tmp/test/part_exper/resnet_";

const char* mobilenet_inputFile = "/data/local/tmp/vgg/target_raw_list_20.txt";
std::string mobilenet_OutputDir = "/data/local/tmp/test/part_exper/mobilenet_output_part";
std::string mobilenet_layerPath = "/data/local/tmp/test/part_exper/mobilenet_";

const char* squeezenet_inputFile = "/data/local/tmp/alexnet/target_raw_list_320.txt";
std::string squeezenet_OutputDir = "/data/local/tmp/test/part_exper/squeezenet_output_part";
std::string squeezenet_layerPath = "/data/local/tmp/test/part_exper/squeezenet_";

const char* yolov2_inputFile = "/data/local/tmp/yolov2/target_raw_list_160.txt";
std::string yolov2_OutputDir = "/data/local/tmp/test/part_exper/yolov2_output_part";
std::string yolov2_layerPath = "/data/local/tmp/test/part_exper/yolov2_";

const char* frcnn_inputFile = "/data/local/tmp/alexnet/target_raw_list_320.txt";
std::string frcnn_OutputDir = "/data/local/tmp/test/part_exper/frcnn_output_part";
std::string frcnn_layerPath = "/data/local/tmp/test/part_exper/frcnn_";

std::vector<std::vector<float>> alexnet_inputs;
std::vector<std::vector<float>> vgg_inputs;
std::vector<std::vector<float>> pos_inputs;
std::vector<std::vector<float>> mnist_inputs;
std::vector<std::vector<float>> googlenet_inputs;
std::vector<std::vector<float>> resnet_inputs;
std::vector<std::vector<float>> mobilenet_inputs;
std::vector<std::vector<float>> squeezenet_inputs;
std::vector<std::vector<float>> yolov2_inputs;
std::vector<std::vector<float>> frcnn_inputs;

vector<std::unique_ptr<zdl::DlSystem::ITensor>> alexnet_inputTensor;
vector<std::unique_ptr<zdl::DlSystem::ITensor>> vgg_inputTensor;
vector<std::unique_ptr<zdl::DlSystem::ITensor>> pos_inputTensor;
vector<std::unique_ptr<zdl::DlSystem::ITensor>> mnist_inputTensor;
vector<std::unique_ptr<zdl::DlSystem::ITensor>> googlenet_inputTensor;
vector<std::unique_ptr<zdl::DlSystem::ITensor>> resnet_inputTensor;
vector<std::unique_ptr<zdl::DlSystem::ITensor>> mobilenet_inputTensor;
vector<std::unique_ptr<zdl::DlSystem::ITensor>> squeezenet_inputTensor;
vector<std::unique_ptr<zdl::DlSystem::ITensor>> yolov2_inputTensor;
vector<std::unique_ptr<zdl::DlSystem::ITensor>> frcnn_inputTensor;

zdl::DlSystem::TensorMap midTensorMap_alexnet;
zdl::DlSystem::TensorMap midTensorMap_vgg;
zdl::DlSystem::TensorMap midTensorMap_vgg2;
zdl::DlSystem::TensorMap midTensorMap_vgg3;
zdl::DlSystem::TensorMap midTensorMap_pos;
zdl::DlSystem::TensorMap midTensorMap_mnist;
zdl::DlSystem::TensorMap midTensorMap_googlenet;
zdl::DlSystem::TensorMap midTensorMap_resnet;
zdl::DlSystem::TensorMap midTensorMap_mobilenet;
zdl::DlSystem::TensorMap midTensorMap_squeezenet;
zdl::DlSystem::TensorMap midTensorMap_yolov2;
zdl::DlSystem::TensorMap midTensorMap_yolov22;
zdl::DlSystem::TensorMap midTensorMap_yolov23;
zdl::DlSystem::TensorMap midTensorMap_frcnn;

std::vector<std::unique_ptr<zdl::SNPE::SNPE>> SNPE_alexnet;
std::vector<std::unique_ptr<zdl::SNPE::SNPE>> SNPE_vgg;
std::vector<std::unique_ptr<zdl::SNPE::SNPE>> SNPE_pos;
std::vector<std::unique_ptr<zdl::SNPE::SNPE>> SNPE_mnist;
std::vector<std::unique_ptr<zdl::SNPE::SNPE>> SNPE_googlenet;
std::vector<std::unique_ptr<zdl::SNPE::SNPE>> SNPE_resnet;
std::vector<std::unique_ptr<zdl::SNPE::SNPE>> SNPE_mobilenet;
std::vector<std::unique_ptr<zdl::SNPE::SNPE>> SNPE_squeezenet;
std::vector<std::unique_ptr<zdl::SNPE::SNPE>> SNPE_yolov2;
std::vector<std::unique_ptr<zdl::SNPE::SNPE>> SNPE_frcnn;

// **** ALL values are average
float alexnet_big = 106.0;
float alexnet_gpu = 20.35;
float alexnet_dsp = 23.0;
float vgg_big = 1150.3;
float vgg_gpu = 264.5;
float vgg_dsp = 100.1;
float lenet_big = 5.0;
float lenet_gpu = 5.3;
float lenet_dsp = 7.0;
float googlenet_big = 330.1;
float googlenet_gpu = 28.6;
float googlenet_dsp = 18.9;
float resnet_big = 501.8;
float resnet_gpu = 58.4;
float resnet_dsp = 36.8;
float mobilenet_big = 742.4;
float mobilenet_gpu = 13.2;
float mobilenet_dsp = 13.1;
float squeezenet_big = 182.4;
float squeezenet_gpu = 16.9;
float squeezenet_dsp = 12.9;
float yolov2_big = 253.9;
float yolov2_gpu = 48.1;
float yolov2_dsp = 48.9;
float frcnn_big = 105.4;
float frcnn_gpu = 20.2;
float frcnn_dsp = 20.5;


class Model_Parameter {
public:
	char id;	
	int batch;
	int num_layers;
	int SNPE_start_index;
	string device;
	string ver;
	int deadline;
	
	float exp_runtime; // expected runtime
	float exp_latency; // expected latency

	float BIG_runtime[4];	
	float GPU_runtime[4];	
	float DSP_runtime[4];	

        Model_Parameter(char _id, int _batch, int _num_layers, string _device, string _ver, int _deadline){
		id = _id;	
		batch = _batch;
		num_layers = _num_layers;
		device = _device;
		ver = _ver;
		deadline = _deadline;
        }

	void SetStartSnpeIndex(int _SNPE_start_index) {
		SNPE_start_index = _SNPE_start_index;
	}
	
	void InitModelProfiledRuntime(char _id, string _device, int _num_layers) {
		// init runtime
		for(int i = 0; i < _num_layers; i++) {
			BIG_runtime[i] = 999999;
			GPU_runtime[i] = 999999;
			DSP_runtime[i] = 999999;
		}

		// vgg 4 layers for DSP
		if(_id == 'v' && _device[0] == 'D' && _num_layers == 4) {
			DSP_runtime[0] = 32;
			DSP_runtime[1] = 29;
			DSP_runtime[2] = 24;
			DSP_runtime[3] = 20;
		}
		// vgg 4 layers for GPU
		else if(_id == 'v' && _device[0] == 'G' && _num_layers == 4) {
			GPU_runtime[0] = 88;
			GPU_runtime[1] = 60;
			GPU_runtime[2] = 49;
			GPU_runtime[3] = 97;
		}
		// yolov2tiny 4 layers for DSP
		else if(_id == 'y' && _device[0] == 'D' && _num_layers == 4) {
			DSP_runtime[0] = 16;
			DSP_runtime[1] = 8;
			DSP_runtime[2] = 11;
			DSP_runtime[3] = 28;
		}
		// yolov2tiny 4 layers for GPU
		else if(_id == 'y' && _device[0] == 'G' && _num_layers == 4) {
			GPU_runtime[0] = 11;
			GPU_runtime[1] = 12;
			GPU_runtime[2] = 20;
			GPU_runtime[3] = 18;
		}
		else if(_id == 'a' && _num_layers == 1) { // Alexnet 
			if(_device[0] == 'B') BIG_runtime[0] = alexnet_big;
			else if(_device[0] == 'G') GPU_runtime[0] = alexnet_gpu;
			else if(_device[0] == 'D') DSP_runtime[0] = alexnet_dsp;
		}
		else if(_id == 'v' && _num_layers == 1) { // VGG 
			if(_device[0] == 'B') BIG_runtime[0] = vgg_big;
			else if(_device[0] == 'G') GPU_runtime[0] = vgg_gpu;
			else if(_device[0] == 'D') DSP_runtime[0] = vgg_dsp;
		}
		else if(_id == 'l' && _num_layers == 1) { // lenet
			if(_device[0] == 'B') BIG_runtime[0] = lenet_big;
			else if(_device[0] == 'G') GPU_runtime[0] = lenet_gpu;
			else if(_device[0] == 'D') DSP_runtime[0] = lenet_dsp;
		}
		else if(_id == 'g' && _num_layers == 1) { // googlenet
			if(_device[0] == 'B') BIG_runtime[0] = googlenet_big;
			else if(_device[0] == 'G') GPU_runtime[0] = googlenet_gpu;
			else if(_device[0] == 'D') DSP_runtime[0] = googlenet_dsp;
		}
		else if(_id == 'r' && _num_layers == 1) { // resnet50
			if(_device[0] == 'B') BIG_runtime[0] = resnet_big;
			else if(_device[0] == 'G') GPU_runtime[0] = resnet_gpu;
			else if(_device[0] == 'D') DSP_runtime[0] = resnet_dsp;
		}
		else if(_id == 'm' && _num_layers == 1) { // mobilenet
			if(_device[0] == 'B') BIG_runtime[0] = mobilenet_big;
			else if(_device[0] == 'G') GPU_runtime[0] = mobilenet_gpu;
			else if(_device[0] == 'D') DSP_runtime[0] = mobilenet_dsp;
		}
		else if(_id == 's' && _num_layers == 1) { // squeezenet
			if(_device[0] == 'B') BIG_runtime[0] = squeezenet_big;
			else if(_device[0] == 'G') GPU_runtime[0] = squeezenet_gpu;
			else if(_device[0] == 'D') DSP_runtime[0] = squeezenet_dsp;
		}
		else if(_id == 'y' && _num_layers == 1) { // yolov2
			if(_device[0] == 'B') BIG_runtime[0] = yolov2_big;
			else if(_device[0] == 'G') GPU_runtime[0] = yolov2_gpu;
			else if(_device[0] == 'D') DSP_runtime[0] = yolov2_dsp;
		}
		else if(_id == 'f' && _num_layers == 1) { // faster rcnn
			if(_device[0] == 'B') BIG_runtime[0] = frcnn_big;
			else if(_device[0] == 'G') GPU_runtime[0] = frcnn_gpu;
			else if(_device[0] == 'D') DSP_runtime[0] = frcnn_dsp;
		}

	}
	
	void PrintParameters() {
		cout << "App id: " << id << endl;	
		cout << "Batch: " << batch << endl;
		cout << "Num layers: " << num_layers << endl; 
		cout << "Snpe index: " << SNPE_start_index << endl;
		cout << "Device: " << device  << endl;
		cout << "Version: " << ver  << endl;
		cout << "Deadline: " << deadline  << endl;

		for(int i = 0; i < num_layers; i++) {
			cout << "CPU " << i << " layer Runtime: " << BIG_runtime[i] << endl;
			cout << "GPU " << i << " layer Runtime: " << GPU_runtime[i] << endl;
			cout << "DSP " << i << " layer Runtime: " << DSP_runtime[i] << endl;
		}
	}
};

class Task {
public:
	char id; // "Alexnet: a", "VGG: v", "Mnist: m", "Googlenet: g"
	int arrival_time; // defined arrival_time

	int layer_num;	
	int batch_size;
	int SNPE_index; // SNPE queue index
	char dev; // BIG, GPU, DSP
	int task_idx; // for Parent_queue
	int deadline; // deadline
	int emergency;
	int total_layer_num;
	float runtime;
	float wruntime;
	float rruntime; // remaining time

	float exp_runtime; // expected runtime
	float exp_latency; // expected latency
	int is_vio;

	long int batch_enqueue_time; 
	long int after_task_scheduler_time;

	int wait_time; // wait_time

	//int real_arrival_time; // real arrival time
	int real_latency;
	int real_runtime;

	int wait_queue_length; // waiting tasks in queue

        Task(char _id, int _arrival_time){
		id = _id;
		arrival_time = _arrival_time;
		rruntime = 0;
	}
};

typedef struct Combi_R {
	int idx;		
	int sum_latency;
	int SLO_satisfied;	
	int SLO_vio_runtime;	
	int sum_energy;	
	int gpu_idx;
} Combi_R;


// Global constant variable
enum {UNKNOWN, USERBUFFER_FLOAT, USERBUFFER_TF8, ITENSOR};
const int FAILURE = 1;
const int SUCCESS = 0;
bool execStatus = false;
bool usingInitCaching = false;
std::string bufferTypeStr = "ITENSOR";
std::string userBufferSourceStr = "CPUBUFFER";

vector<Model_Parameter> Model_Par_List;
vector<Task> Batch_queue;
vector<Task> BIG_queue;
vector<Task> GPU_queue;
vector<Task> DSP_queue;

vector<int> Parent_queue; // parent who have child finish check (get task_idx)
vector<int> PParent_queue; // parent who have child finish check (get task_idx)
vector<int> PPParent_queue; // parent who have child finish check (get task_idx)

int BIG_RUN = 0;
int GPU_RUN = 0;
int DSP_RUN = 0;
int Finished_BIG = 0; 
int Finished_GPU = 0; 
int Finished_DSP = 0; 

int vgg_slice_exist = 0;
int num_VGG_SLICE_gpu = 0;
int num_VGG_SLICE_dsp = 0;
int num_YOLO_SLICE_gpu = 0;
int num_YOLO_SLICE_dsp = 0;

#define EMERGENCY_ON 1
#define EMERGENCY_OFF 0
float SLO_alpha = 1.5;
int Task_index = 0;

// Global variable 
ofstream Write_file;
ofstream Write_file_BIG;
ofstream Write_file_GPU;
ofstream Write_file_DSP;


void InitGlobalState() {
	Model_Par_List.clear();
	Batch_queue.clear();
	BIG_queue.clear();
	GPU_queue.clear();
	DSP_queue.clear();
	Task_index = 0;
	BIG_RUN = 0;
	GPU_RUN = 0;
	DSP_RUN = 0;
	Finished_BIG = 0; 
	Finished_GPU = 0; 
	Finished_DSP = 0; 

	vgg_slice_exist = 0;
	num_VGG_SLICE_gpu = 0;
	num_VGG_SLICE_dsp = 0;
	num_YOLO_SLICE_gpu = 0;
	num_YOLO_SLICE_dsp = 0;
}

void ReadDirectory(const std::string& name, vector<string>& v)
{
    DIR* dirp = opendir(name.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
	if(dp->d_name[0] == 'I')
       		v.push_back(dp->d_name);
    }
    closedir(dirp);
}

std::string GetAppList(std::string req_inputfile)
{
     size_t pos = req_inputfile.find("+"); 
     string app_list = req_inputfile.substr(pos+1, req_inputfile.size());
     return app_list;
}

vector<vector<int> > Cartesian( vector<vector<int> >& v ) { 
  auto product = []( long long a, vector<int>& b ) { return a*b.size(); };
  const long long N = accumulate( v.begin(), v.end(), 1LL, product );
  vector<vector<int> > all; 
  vector<int> u(v.size());
  for( long long n=0 ; n<N ; ++n ) { 
    lldiv_t q { n, 0 };
    for( long long i=v.size()-1 ; 0<=i ; --i ) { 
      q = div( q.quot, v[i].size() );
      u[i] = v[i][q.rem];
    }   
    all.push_back(u);
  }
  return all;
}

int DeadlineCheck(int task_deadline, int vRuntime){
	if(task_deadline >= vRuntime * SLO_alpha) 
		return 1;
	//return task_deadline - vRuntime * SLO_alpha; 
	return -(100.0 * vRuntime * SLO_alpha / task_deadline);
}



void SettingModelParameters(string algo_cmd, string app_list, int deadlineN) {
	
	const char* app_inputFile; 
	std::string app_OutputDir;
	std::string app_layerPath;

	ssize_t pos; // app check

	int num_layers[] = {1, 1, 1};
	string ver_set[] = {"0", "0", "0"};
	string devices_set[] = {"D", "G", "B"};

	// App dependent
	int deadline = -1;
	//int snpe_start_index = -1;	

	Model_Parameter* model_par;

	// non-sliced layers
	for(int i = 0; i < app_list.size(); i++) {
		// app dependent
		if(app_list[i] == 'a') { // alexnet
    			deadline = 20 * deadlineN;	
			if(algo_cmd.compare("slo_div") != 0) {
				// Model_id, batch, num_layers, devices, version, deadline
				model_par = new Model_Parameter('a', 1, 1, "D", "0", deadline); Model_Par_List.push_back(*model_par);
				model_par = new Model_Parameter('a', 1, 1, "G", "0", deadline); Model_Par_List.push_back(*model_par);
				model_par = new Model_Parameter('a', 1, 1, "B", "0", deadline); Model_Par_List.push_back(*model_par);
			}
			else {
				model_par = new Model_Parameter('a', 1, 1, "D", "0", deadline); Model_Par_List.push_back(*model_par);
				model_par = new Model_Parameter('a', 1, 1, "G", "0", deadline); Model_Par_List.push_back(*model_par);
				model_par = new Model_Parameter('a', 1, 4, "BBBB", "0", deadline); Model_Par_List.push_back(*model_par);
			}
		}
		else if(app_list[i] == 'v') { // vgg
			vgg_slice_exist = 1;
    			deadline = 100 * deadlineN;	
			if(algo_cmd.compare("slo_div") != 0) {
				model_par = new Model_Parameter('v', 1, 1, "D", "0", deadline); Model_Par_List.push_back(*model_par);
				model_par = new Model_Parameter('v', 1, 1, "G", "0", deadline); Model_Par_List.push_back(*model_par);
			}
			else {
				model_par = new Model_Parameter('v', 1, 1, "D", "0", deadline); Model_Par_List.push_back(*model_par);
				model_par = new Model_Parameter('v', 1, 1, "G", "0", deadline); Model_Par_List.push_back(*model_par);
				model_par = new Model_Parameter('v', 1, 4, "GGGG", "0", deadline); Model_Par_List.push_back(*model_par);
				model_par = new Model_Parameter('v', 1, 4, "DDDD", "0", deadline); Model_Par_List.push_back(*model_par);
			}
		}
		else if(app_list[i] == 'l') { // lenet
    			deadline = 5 * deadlineN;	
			model_par = new Model_Parameter('l', 1, 1, "D", "0", deadline); Model_Par_List.push_back(*model_par);
			model_par = new Model_Parameter('l', 1, 1, "G", "0", deadline); Model_Par_List.push_back(*model_par);
			model_par = new Model_Parameter('l', 1, 1, "B", "0", deadline); Model_Par_List.push_back(*model_par);
		}
		else if(app_list[i] == 'g') { // googlenet
    			//deadline = 16 * deadlineN;	
    			deadline = 18.9 * deadlineN;	
			model_par = new Model_Parameter('g', 1, 1, "D", "0", deadline); Model_Par_List.push_back(*model_par);
			model_par = new Model_Parameter('g', 1, 1, "G", "0", deadline); Model_Par_List.push_back(*model_par);
			model_par = new Model_Parameter('g', 1, 1, "B", "0", deadline); Model_Par_List.push_back(*model_par);
		}
		else if(app_list[i] == 'r') { // resnet50
    			deadline = 37 * deadlineN;	
			model_par = new Model_Parameter('r', 1, 1, "D", "0", deadline); Model_Par_List.push_back(*model_par);
			model_par = new Model_Parameter('r', 1, 1, "G", "0", deadline); Model_Par_List.push_back(*model_par);
			model_par = new Model_Parameter('r', 1, 1, "B", "0", deadline); Model_Par_List.push_back(*model_par);
		}
		else if(app_list[i] == 'm') { // mobileNet
    			deadline = 13.1 * deadlineN;	
			model_par = new Model_Parameter('m', 1, 1, "D", "0", deadline); Model_Par_List.push_back(*model_par);
			model_par = new Model_Parameter('m', 1, 1, "G", "0", deadline); Model_Par_List.push_back(*model_par);
			model_par = new Model_Parameter('m', 1, 1, "B", "0", deadline); Model_Par_List.push_back(*model_par);
		}
		else if(app_list[i] == 's') { // SqueezeNet
    			deadline = 12.9 * deadlineN;	
			model_par = new Model_Parameter('s', 1, 1, "D", "0", deadline); Model_Par_List.push_back(*model_par);
			model_par = new Model_Parameter('s', 1, 1, "G", "0", deadline); Model_Par_List.push_back(*model_par);
			model_par = new Model_Parameter('s', 1, 1, "B", "0", deadline); Model_Par_List.push_back(*model_par);
		}
		else if(app_list[i] == 'y') { // yoloV2tiny
    			deadline = 48.1 * deadlineN;	
			if(algo_cmd.compare("slo_div") != 0) {
				model_par = new Model_Parameter('y', 1, 1, "D", "0", deadline); Model_Par_List.push_back(*model_par);
				model_par = new Model_Parameter('y', 1, 1, "G", "0", deadline); Model_Par_List.push_back(*model_par);
				model_par = new Model_Parameter('y', 1, 1, "B", "0", deadline); Model_Par_List.push_back(*model_par);
			}
			else {
				model_par = new Model_Parameter('y', 1, 1, "D", "0", deadline); Model_Par_List.push_back(*model_par);
				model_par = new Model_Parameter('y', 1, 1, "G", "0", deadline); Model_Par_List.push_back(*model_par);
				model_par = new Model_Parameter('y', 1, 1, "B", "0", deadline); Model_Par_List.push_back(*model_par);
				model_par = new Model_Parameter('y', 1, 4, "GGGG", "0", deadline); Model_Par_List.push_back(*model_par);
				model_par = new Model_Parameter('y', 1, 4, "DDDD", "0", deadline); Model_Par_List.push_back(*model_par);
			}
		}
		else if(app_list[i] == 'f') { // faster RCNN
    			deadline = 20.2 * deadlineN;	
			model_par = new Model_Parameter('f', 1, 1, "D", "0", deadline); Model_Par_List.push_back(*model_par);
			model_par = new Model_Parameter('f', 1, 1, "G", "0", deadline); Model_Par_List.push_back(*model_par);
			model_par = new Model_Parameter('f', 1, 1, "B", "0", deadline); Model_Par_List.push_back(*model_par);
		}
	}

	int snpe_start_index = 0;	
	char prev_id;
	for(int i = 0; i < Model_Par_List.size(); i++) {
		Model_Parameter* mp = &Model_Par_List[i];
		if(mp->id != prev_id) {
			snpe_start_index = 0;	
			prev_id = mp->id;
		}
		snpe_start_index += mp->num_layers;	
		mp->SetStartSnpeIndex(snpe_start_index - mp->num_layers);
		mp->InitModelProfiledRuntime(mp->id, mp->device, mp->num_layers); 
	}
	
	cout << "=========================================================" << endl;
	cout << "Total Model list: " << Model_Par_List.size() << endl;
	for(int i = 0; i < Model_Par_List.size(); i++) {
		Model_Par_List[i].PrintParameters();
		cout << "=========================================================" << endl;
	}
}

std::unique_ptr<zdl::SNPE::SNPE> BuildDNNModel(std::string dlc, std::string OutputDir, std::string bufferTypeStr, std::string userBufferSourceStr, std::string device, int batchSize) {

    // Check if given arguments represent valid files
    std::ifstream dlcFile(dlc);
    //std::ifstream inputList(inputFile);
    if (!dlcFile) {
        std::cout << "Input list or dlc file not valid. Please ensure that you have provided a valid input list and dlc for processing. Run snpe-sample with the -h flag for more details" << std::endl;
        std::exit(FAILURE);
    }

    // Check if given buffer type is valid
    int bufferType;
    if (bufferTypeStr == "USERBUFFER_FLOAT")
    {
        bufferType = USERBUFFER_FLOAT;
    }
    else if (bufferTypeStr == "USERBUFFER_TF8")
    {
        bufferType = USERBUFFER_TF8;
    }
    else if (bufferTypeStr == "ITENSOR")
    {
        bufferType = ITENSOR;
    }
    else
    {
        std::cout << "Buffer type is not valid. Please run snpe-sample with the -h flag for more details" << std::endl;
        std::exit(FAILURE);
    }

    // Open the DL container that contains the network to execute.
    // Create an instance of the SNPE network from the now opened container.
    // The factory functions provided by SNPE allow for the specification
    // of which layers of the network should be returned as output and also
    // if the network should be run on the CPU or GPU.
    // The runtime availability API allows for runtime support to be queried.
    // If a selected runtime is not available, we will issue a warning and continue,
    // expecting the invalid configuration to be caught at SNPE network creation.
    zdl::DlSystem::UDLFactoryFunc udlFunc = sample::MyUDLFactory;
    zdl::DlSystem::UDLBundle udlBundle; udlBundle.cookie = (void*)0xdeadbeaf, udlBundle.func = udlFunc; // 0xdeadbeaf to test cookie

    zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;

    if (device[0] == 'g') 
        runtime = zdl::DlSystem::Runtime_t::GPU;
    else if (device[0] == 'd') 
        runtime = zdl::DlSystem::Runtime_t::DSP;
    else if (device[0] == 'c') 
        runtime = zdl::DlSystem::Runtime_t::CPU;
    runtime = checkRuntime(runtime);
    std::unique_ptr<zdl::DlContainer::IDlContainer> container = loadContainerFromFile(dlc);
    if (container == nullptr)
    {
       std::cerr << "Error while opening the container file." << std::endl;
       std::exit(FAILURE);
    }

    bool useUserSuppliedBuffers = (bufferType == USERBUFFER_FLOAT || bufferType == USERBUFFER_TF8);

    zdl::DlSystem::PlatformConfig platformConfig;

    std::unique_ptr<zdl::SNPE::SNPE> snpe_final = setBuilderOptions(container, runtime, udlBundle, useUserSuppliedBuffers, platformConfig, usingInitCaching);

    if (snpe_final == nullptr)
    {
       std::cerr << "Error while building SNPE object." << std::endl;
       std::exit(FAILURE);
    }
    if (usingInitCaching)
    {
       if (container->save(dlc))
       {
          std::cout << "Saved container into archive successfully" << std::endl;
       }
       else
       {
          std::cout << "Failed to save container into archive" << std::endl;
       }
    }

    // Configure logging output and start logging. The snpe-diagview
    // executable can be used to read the content of this diagnostics file
    auto logger_opt = snpe_final->getDiagLogInterface();
    if (!logger_opt) throw std::runtime_error("SNPE failed to obtain logging interface");
    auto logger = *logger_opt;
    auto opts = logger->getOptions();

    opts.LogFileDirectory = OutputDir;
    if(!logger->setOptions(opts)) {
        std::cerr << "Failed to set options" << std::endl;
        std::exit(FAILURE);
    }
    if (!logger->start()) {
        std::cerr << "Failed to start logger" << std::endl;
        std::exit(FAILURE);
    }
    return std::move(snpe_final);
}

int PrefetchInputFile(std::string OutputDir, vector<vector<float> > *model_inputs, const char* inputFile, int batchSize){

    std::ifstream inputList(inputFile);
    if (!inputList) {
        std::cout << "Input list or dlc file not valid. Please ensure that you have provided a valid input list and dlc for processing. Run snpe-sample with the -h flag for more details" << std::endl;
        std::exit(0); 
    }

    // Open the input file listing and group input files into batches
    std::vector<std::vector<std::string>> inputs = preprocessInput(inputFile, batchSize);

    for (size_t i = 0; i < inputs.size(); i++) {
    		std::vector<float> inputVec;
    		for(size_t j=0; j<inputs[i].size(); j++) {
       	 		std::string filePath(inputs[i][j]);
        		//std::cout << "Processing DNN Input: " << filePath << "\n";
        		std::vector<float> loadedFile = loadFloatDataFile(filePath);
        		inputVec.insert(inputVec.end(), loadedFile.begin(), loadedFile.end());
    		}
		model_inputs->push_back(inputVec);
		break; // for all batch sizes (even, odd, ...)
    }
    return SUCCESS;
}

void BuildModelAll(std::string app_OutputDir,std::string app_layerPath, std::string device_list, int batchSize, int num_input_layers, std::vector<std::unique_ptr<zdl::SNPE::SNPE>> &SNPE_vec) {
    std::string device = ""; 

    for(int i = 0; i < num_input_layers; i++) {
	stringstream part_num;
	part_num << i;
	std::string app_layerPath_full = app_layerPath + part_num.str() + ".dlc";

	if(device_list[i] == 'B') device = "cpu";
	else if(device_list[i] == 'G') device = "gpu";
	else if(device_list[i] == 'D') device = "dsp";
	
	std::unique_ptr<zdl::SNPE::SNPE> snpe = BuildDNNModel(app_layerPath_full, app_OutputDir, bufferTypeStr, userBufferSourceStr, device, batchSize);
	SNPE_vec.push_back(std::move(snpe));	

	cout << app_layerPath << ": "  << device << " finished" << endl; 
    }
}


std::unique_ptr<zdl::DlSystem::ITensor> GenerateInputTensor(std::vector<std::unique_ptr<zdl::SNPE::SNPE>>& model_ptr, std::vector<std::vector<float>>& input_ptr, int input_index, int start_index){
    auto& SNPE = model_ptr; 
    auto& inputs = input_ptr;

    // Check if given buffer type is valid
    int bufferType;
    if (bufferTypeStr == "USERBUFFER_FLOAT")
    {
        bufferType = USERBUFFER_FLOAT;
    }
    else if (bufferTypeStr == "USERBUFFER_TF8")
    {
        bufferType = USERBUFFER_TF8;
    }
    else if (bufferTypeStr == "ITENSOR")
    {
        bufferType = ITENSOR;
    }
    else
    {
        std::cout << "Buffer type is not valid. Please run snpe-sample with the -h flag for more details" << std::endl;
        std::exit(FAILURE);
    }

    if(bufferType == ITENSOR) { 
        zdl::DlSystem::TensorMap outputTensorMap;

        for (size_t i = input_index; i < inputs.size(); i++) {
   		std::unique_ptr<zdl::DlSystem::ITensor> input;
		const auto &strList_opt = SNPE.at(start_index)->getInputTensorNames();
    		if (!strList_opt) throw std::runtime_error("Error obtaining Input tensor names");
    		const auto &strList = *strList_opt;
    		assert (strList.size() == 1);

    		const auto &inputDims_opt = SNPE.at(start_index)->getInputDimensions(strList.at(0));
    		const auto &inputShape = *inputDims_opt;

    		input = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputShape);
	
    		if (input->getSize() != inputs[i].size()) {
       		 	std::cerr << "Size of input does not match network.\n"
       		           << "Expecting: " << input->getSize() << "\n"
       		           << "Got: " << inputs[i].size() << "\n";
       		 	std::exit(EXIT_FAILURE);
    		}
    		std::copy(inputs[i].begin(), inputs[i].end(), input->begin());
    		return std::move(input);   	
	}
    }
}

void BuildSnpeModelAll() {
    const char* app_inputFile; 
    string app_OutputDir;
    string app_layerPath;
    int model_num = 0;
    char prev_id;

    for(int i = 0; i < Model_Par_List.size(); i++) {
  	Model_Parameter* mp = &Model_Par_List[i];
	
	if(prev_id != mp->id) {
		prev_id = mp->id;	
		model_num = 0;	
	}
	
	// Alexnet
	if(mp->id == 'a') {
		app_layerPath = alexnet_layerPath + to_string(mp->num_layers) + "_dlc/part";
    		app_OutputDir = alexnet_OutputDir; 
		app_inputFile = alexnet_inputFile;

    		BuildModelAll(app_OutputDir, app_layerPath, mp->device, mp->batch, mp->num_layers, SNPE_alexnet);
		PrefetchInputFile(app_OutputDir, &alexnet_inputs, app_inputFile, mp->batch);
   		alexnet_inputTensor.push_back(GenerateInputTensor(SNPE_alexnet, alexnet_inputs, model_num++, mp->SNPE_start_index)); // i is for inputs, index is for SNPE array 
	}
	// VGG-16
	else if(mp->id == 'v') {
		app_layerPath = vgg_layerPath + to_string(mp->num_layers) + "_dlc/part";
    		app_OutputDir = vgg_OutputDir; 
		app_inputFile = vgg_inputFile;

    		BuildModelAll(app_OutputDir, app_layerPath, mp->device, mp->batch, mp->num_layers, SNPE_vgg);
   		PrefetchInputFile(app_OutputDir, &vgg_inputs, app_inputFile, mp->batch);
  		vgg_inputTensor.push_back(GenerateInputTensor(SNPE_vgg, vgg_inputs, model_num++, mp->SNPE_start_index)); // i is for inputs, index is for SNPE array 
	}
	// LeNet
	else if(mp->id == 'l') {
		app_layerPath = mnist_layerPath + to_string(mp->num_layers) + "_dlc/part";
    		app_OutputDir = mnist_OutputDir; 
		app_inputFile = mnist_inputFile;

    		BuildModelAll(app_OutputDir, app_layerPath, mp->device, mp->batch, mp->num_layers, SNPE_mnist);
 		PrefetchInputFile(app_OutputDir, &mnist_inputs, app_inputFile, mp->batch);
  		mnist_inputTensor.push_back(GenerateInputTensor(SNPE_mnist, mnist_inputs, model_num++, mp->SNPE_start_index)); // i is for inputs, index is for SNPE array 
	}

	// GoogleNet
	else if(mp->id == 'g') {
		app_layerPath = googlenet_layerPath + to_string(mp->num_layers) + "_dlc/part";
    		app_OutputDir = googlenet_OutputDir; 
		app_inputFile = googlenet_inputFile;
		
    		BuildModelAll(app_OutputDir, app_layerPath, mp->device, mp->batch, mp->num_layers, SNPE_googlenet);
  		PrefetchInputFile(app_OutputDir, &googlenet_inputs, app_inputFile, mp->batch);
  		googlenet_inputTensor.push_back(GenerateInputTensor(SNPE_googlenet, googlenet_inputs, model_num++, mp->SNPE_start_index)); // i is for inputs, index is for SNPE array 
	}

	// ResNet-50
	else if(mp->id == 'r') {
		app_layerPath = resnet_layerPath + to_string(mp->num_layers) + "_dlc/part";
    		app_OutputDir = resnet_OutputDir; 
		app_inputFile = resnet_inputFile;

    		BuildModelAll(app_OutputDir, app_layerPath, mp->device, mp->batch, mp->num_layers, SNPE_resnet);
		PrefetchInputFile(app_OutputDir, &resnet_inputs, app_inputFile, mp->batch);
   		resnet_inputTensor.push_back(GenerateInputTensor(SNPE_resnet, resnet_inputs, model_num++, mp->SNPE_start_index)); // i is for inputs, index is for SNPE array 
	}

	// MobileNet
	else if(mp->id == 'm') {
		app_layerPath = mobilenet_layerPath + to_string(mp->num_layers) + "_dlc/part";
    		app_OutputDir = mobilenet_OutputDir; 
		app_inputFile = mobilenet_inputFile;

    		BuildModelAll(app_OutputDir, app_layerPath, mp->device, mp->batch, mp->num_layers, SNPE_mobilenet);
 		PrefetchInputFile(app_OutputDir, &mobilenet_inputs, app_inputFile, mp->batch);
  		mobilenet_inputTensor.push_back(GenerateInputTensor(SNPE_mobilenet, mobilenet_inputs, model_num++, mp->SNPE_start_index)); // i is for inputs, index is for SNPE array 
	}

	// SqueezeNet
	else if(mp->id == 's') {
		app_layerPath = squeezenet_layerPath + to_string(mp->num_layers) + "_dlc/part";
    		app_OutputDir = squeezenet_OutputDir; 
		app_inputFile = squeezenet_inputFile;

    		BuildModelAll(app_OutputDir, app_layerPath, mp->device, mp->batch, mp->num_layers, SNPE_squeezenet);
		PrefetchInputFile(app_OutputDir, &squeezenet_inputs, app_inputFile, mp->batch);
   		squeezenet_inputTensor.push_back(GenerateInputTensor(SNPE_squeezenet, squeezenet_inputs, model_num++, mp->SNPE_start_index)); // i is for inputs, index is for SNPE array 
	}

	// YoloV2tiny
	else if(mp->id == 'y') {
		app_layerPath = yolov2_layerPath + to_string(mp->num_layers) + "_dlc/part";
    		app_OutputDir = yolov2_OutputDir; 
		app_inputFile = yolov2_inputFile;

    		BuildModelAll(app_OutputDir, app_layerPath, mp->device, mp->batch, mp->num_layers, SNPE_yolov2);
		PrefetchInputFile(app_OutputDir, &yolov2_inputs, app_inputFile, mp->batch);
   		yolov2_inputTensor.push_back(GenerateInputTensor(SNPE_yolov2, yolov2_inputs, model_num++, mp->SNPE_start_index)); // i is for inputs, index is for SNPE array 
	}

	// faster_rcnn
	else if(mp->id == 'f') {
		app_layerPath = frcnn_layerPath + to_string(mp->num_layers) + "_dlc/part";
    		app_OutputDir = frcnn_OutputDir; 
		app_inputFile = frcnn_inputFile;

    		BuildModelAll(app_OutputDir, app_layerPath, mp->device, mp->batch, mp->num_layers, SNPE_frcnn);
		PrefetchInputFile(app_OutputDir, &frcnn_inputs, app_inputFile, mp->batch);
   		frcnn_inputTensor.push_back(GenerateInputTensor(SNPE_frcnn, frcnn_inputs, model_num++, mp->SNPE_start_index)); // i is for inputs, index is for SNPE array 
	}

	}
}

bool ARRIVAL_CMP(const Task &p1, const Task &p2){
    if(p1.arrival_time < p2.arrival_time){
        return true;
    }
    else{
        return false;
    }
}

bool SLO_CMP(const Combi_R &a, const Combi_R &b){
	if(a.SLO_vio_runtime > b.SLO_vio_runtime)
		return true;
	else if(a.SLO_vio_runtime == b.SLO_vio_runtime)
		return a.sum_latency < b.sum_latency;		
	else
		return false;
}


void EnqueueTask(Model_Parameter* selected, Task* task, int emergency_on){

	// for slice version
	if(selected->num_layers > 1) {
		Parent_queue.push_back(Task_index);
		PParent_queue.push_back(Task_index);
		PPParent_queue.push_back(Task_index);
	}

	// Generate and Enqueue Task
	for(int i = 0; i < selected->num_layers; i++) {
		Task* new_task = new Task(task->id, task->arrival_time);

		new_task->layer_num = i;		
		new_task->batch_size = task->batch_size;
		new_task->SNPE_index = selected->SNPE_start_index + i;
		new_task->dev = selected->device[i];
		new_task->task_idx = Task_index;		
		new_task->deadline = selected->deadline;
		if(emergency_on)
			new_task->emergency = 1;
		else 
			new_task->emergency = 0;
		new_task->total_layer_num = selected->num_layers;
	
		new_task->exp_latency = selected->exp_latency;
		new_task->exp_runtime = selected->exp_runtime;
		new_task->is_vio = DeadlineCheck(selected->deadline, selected->exp_latency);

		new_task->batch_enqueue_time = task->batch_enqueue_time;
		new_task->after_task_scheduler_time = 0;

		new_task->wait_time = 0;		
		
		if(new_task->dev == 'B'){
			new_task->runtime = selected->BIG_runtime[i];
			new_task->wruntime = selected->BIG_runtime[i];
			for(int j = i; j < selected->num_layers; j++) 
				new_task->rruntime += selected->BIG_runtime[j];  
			new_task->wait_queue_length = BIG_queue.size();
			BIG_queue.push_back(*new_task);
		}
		else if(new_task->dev == 'G'){ 
			new_task->runtime = selected->GPU_runtime[i];
			new_task->wruntime = selected->GPU_runtime[i];
			for(int j = i; j < selected->num_layers; j++) 
				new_task->rruntime += selected->GPU_runtime[j];  
			new_task->wait_queue_length = GPU_queue.size();
			GPU_queue.push_back(*new_task);
		}
		else if(new_task->dev == 'D'){ 
			new_task->runtime = selected->DSP_runtime[i];
			new_task->wruntime = selected->DSP_runtime[i];
			for(int j = i; j < selected->num_layers; j++) 
				new_task->rruntime += selected->DSP_runtime[j];  
			new_task->wait_queue_length = DSP_queue.size();
			DSP_queue.push_back(*new_task);
		}
	}
 	Task_index++; // increase global task index
}



void GenerateRequestQueue(vector<Task>& Request_queue, string filepath){
	string filePath = filepath;
	ifstream openFile(filePath.data());

	if( openFile.is_open() ){
		string line;
	
		while(getline(openFile, line)){
			string delimiter = ":";
			string task_name = line.substr(0, line.find(delimiter));
			string arrival_time = line.substr(line.find(delimiter) + 1, line.size());

			Task* task = new Task(task_name[0], stoi(arrival_time));
			Request_queue.push_back(*task);
		}
		openFile.close();
	}
        sort(Request_queue.begin(), Request_queue.end(), ARRIVAL_CMP);

}

void MAEL(vector<Task>& Batch_queue, int *vBIG_runtime, int * vGPU_runtime, int* vDSP_runtime) {
	vector<vector<Model_Parameter> > candidate_set;

	// Prepare all candidate set
	for(int i = 0; i < Batch_queue.size(); i++) {
		vector<Model_Parameter> candidate;
		char app_id = Batch_queue[i].id;

		for(int j = 0; j < Model_Par_List.size(); j++) {
			if(Model_Par_List[j].id == app_id && Model_Par_List[j].num_layers == 1) 
			{
				candidate.push_back( Model_Par_List[j] ); 
			}
		}	
		candidate_set.push_back( candidate );	
	}
	vector<vector<int> > all_cand_idx; 	
	vector<vector<int> > all_combi; 	
		
	for(int i = 0; i < candidate_set.size(); i++) {
		vector<int> cand_idx;	
		vector<Model_Parameter> model_cand = candidate_set[i];	
		for(int j = 0; j < model_cand.size(); j++) 
			cand_idx.push_back(j);
		all_cand_idx.push_back(cand_idx);	
	}
	all_combi = Cartesian(all_cand_idx);	

	// Find MAEL (Min-Average-Estimated-Latency)
	vector<int> sum_latency_set;
	for(int i = 0; i < all_combi.size(); i++) {
		int idx;
		float sum_latency = 0;

		int vBIG_sum_tmp = *vBIG_runtime;
		int vGPU_sum_tmp = *vGPU_runtime;
		int vDSP_sum_tmp = *vDSP_runtime;

		for(int j = 0; j < all_combi[i].size(); j++) {
			vector<Model_Parameter> model_cand = candidate_set[j];	
			idx = all_combi[i][j]; 		

			// calculate if new task is run on each device
			if(model_cand[idx].device[0] == 'B') {
				vBIG_sum_tmp += model_cand[idx].BIG_runtime[0];
				sum_latency += vBIG_sum_tmp;	
			}
			else if(model_cand[idx].device[0] == 'G') {
				vGPU_sum_tmp += model_cand[idx].GPU_runtime[0];
				sum_latency += vGPU_sum_tmp;	
			}
			else if(model_cand[idx].device[0] == 'D') {
				vDSP_sum_tmp += model_cand[idx].DSP_runtime[0];
				sum_latency += vDSP_sum_tmp;	
			}
		}
		sum_latency_set.push_back(sum_latency);
	}
	int min = *min_element(sum_latency_set.begin(), sum_latency_set.end());
	vector<int>::iterator it = find(sum_latency_set.begin(), sum_latency_set.end(), min);
	int min_idx = distance(sum_latency_set.begin(), it);

	// Update Device Runtime and Generate Task 
	for(int i = 0; i < all_combi[min_idx].size(); i++)  {
		int idx = all_combi[min_idx][i];
		Model_Parameter* selected = &candidate_set[i][idx];			
	
		if(selected->device[0] == 'B') {
			selected->exp_runtime = selected->BIG_runtime[0]; // runtime
			(*vBIG_runtime) += selected->BIG_runtime[0];
			selected->exp_latency = (*vBIG_runtime); // latency
		}
		else if(selected->device[0] == 'G') {
			selected->exp_runtime = selected->GPU_runtime[0]; // runtime
			(*vGPU_runtime) += selected->GPU_runtime[0];
			selected->exp_latency = (*vGPU_runtime); // latency
		}
		else if(selected->device[0] == 'D') {
			selected->exp_runtime = selected->DSP_runtime[0]; // runtime
			(*vDSP_runtime) += selected->DSP_runtime[0];
			selected->exp_latency = (*vDSP_runtime); // latency
		}
		EnqueueTask(selected, &Batch_queue[i], EMERGENCY_OFF);
	}
}

void PSLO_MAEL(vector<Task> Batch_queue, int *vBIG_runtime, int * vGPU_runtime, int* vDSP_runtime) {
	vector<vector<Model_Parameter> > candidate_set;
	vector<Combi_R> All_combi;

	// Prepare all candidate set
	for(int i = 0; i < Batch_queue.size(); i++) {
		vector<Model_Parameter> candidate;
		char app_id = Batch_queue[i].id;

		for(int j = 0; j < Model_Par_List.size(); j++) {
			if(Model_Par_List[j].id == app_id && Model_Par_List[j].num_layers == 1) 
			{
				candidate.push_back( Model_Par_List[j] ); 
			}
		}	
		candidate_set.push_back( candidate );	
	}
	vector<vector<int> > all_cand_idx; 	
	vector<vector<int> > all_combi; 	
		
	for(int i = 0; i < candidate_set.size(); i++) {
		vector<int> cand_idx;	
		vector<Model_Parameter> model_cand = candidate_set[i];	
		for(int j = 0; j < model_cand.size(); j++) 
			cand_idx.push_back(j);
		all_cand_idx.push_back(cand_idx);	
	}
	all_combi = Cartesian(all_cand_idx);	


	for(int i = 0; i < all_combi.size(); i++) {
		int idx;
		float sum_latency = 0;
		int slo_satisfied = 0;
		int sum_vio_runtime = 0;

		int vBIG_sum_tmp = *vBIG_runtime;
		int vGPU_sum_tmp = *vGPU_runtime;
		int vDSP_sum_tmp = *vDSP_runtime;

		for(int j = 0; j < all_combi[i].size(); j++) {
			vector<Model_Parameter> model_cand = candidate_set[j];	
			idx = all_combi[i][j]; 		
			sum_vio_runtime = 0;

			// calculate if new task is run on each device
			if(model_cand[idx].device[0] == 'B') {
				vBIG_sum_tmp += model_cand[idx].BIG_runtime[0];
				sum_latency += vBIG_sum_tmp;	
				int vio_runtime = DeadlineCheck(model_cand[idx].deadline, vBIG_sum_tmp); 
				sum_vio_runtime += vio_runtime;
			}
			else if(model_cand[idx].device[0] == 'G') {
				vGPU_sum_tmp += model_cand[idx].GPU_runtime[0];
				sum_latency += vGPU_sum_tmp;	
				int vio_runtime = DeadlineCheck(model_cand[idx].deadline, vGPU_sum_tmp); 
				sum_vio_runtime += vio_runtime;
			}
			else if(model_cand[idx].device[0] == 'D') {
				vDSP_sum_tmp += model_cand[idx].DSP_runtime[0];
				sum_latency += vDSP_sum_tmp;	
				int vio_runtime = DeadlineCheck(model_cand[idx].deadline, vDSP_sum_tmp);
				sum_vio_runtime += vio_runtime;
			}

		}
		Combi_R* new_combi = (Combi_R *)malloc(sizeof(Combi_R));
		new_combi->idx = i;
		new_combi->sum_latency = sum_latency;
		new_combi->SLO_satisfied = slo_satisfied;
		new_combi->SLO_vio_runtime = sum_vio_runtime;
		All_combi.push_back(*new_combi);
	
	}

	sort(All_combi.begin(), All_combi.end(), SLO_CMP);
	int min_idx = All_combi[0].idx;

	// Update Device Runtime and Generate Task 
	for(int i = 0; i < all_combi[min_idx].size(); i++)  {
		int idx = all_combi[min_idx][i];
		Model_Parameter* selected = &candidate_set[i][idx];			

		// sliced task
		if(selected->id == 'v' && selected->device[0] == 'G' && num_VGG_SLICE_gpu < 1) {
				(*vGPU_runtime) += 88; // only first layer of VGG
				num_VGG_SLICE_gpu += 1;	

				for(int j = 0; j < Model_Par_List.size(); j++) {
					if(Model_Par_List[j].id == 'v' && Model_Par_List[j].num_layers == 4 && Model_Par_List[j].device[0] == 'G') {
						selected = &Model_Par_List[j]; 
					}
				}	
				EnqueueTask(selected, &Batch_queue[i], EMERGENCY_OFF);
		}
		else if(selected->id == 'v' && selected->device[0] == 'D' && num_VGG_SLICE_dsp < 1) {
				(*vDSP_runtime) += 32; // only first layer of VGG
				num_VGG_SLICE_dsp += 1;	

				for(int j = 0; j < Model_Par_List.size(); j++) {
					if(Model_Par_List[j].id == 'v' && Model_Par_List[j].num_layers == 4 && Model_Par_List[j].device[0] == 'D') {
						selected = &Model_Par_List[j]; 
					}
				}	
				EnqueueTask(selected, &Batch_queue[i], EMERGENCY_OFF);
		}

		else if(vgg_slice_exist == 0 && selected->id == 'y' && selected->device[0] == 'G' && num_YOLO_SLICE_gpu < 1){
				(*vGPU_runtime) += 15;  // only first layer of YoloV2tiny
				num_YOLO_SLICE_gpu += 1;

				for(int j = 0; j < Model_Par_List.size(); j++) {
					if(Model_Par_List[j].id == 'y' && Model_Par_List[j].num_layers == 4 && Model_Par_List[j].device[0] == 'G') {
						selected = &Model_Par_List[j]; 
					}
				}	
				EnqueueTask(selected, &Batch_queue[i], EMERGENCY_OFF);
		}

		else if(vgg_slice_exist == 0 && selected->id == 'y' && selected->device[0] == 'D' && num_YOLO_SLICE_dsp < 1){
				(*vDSP_runtime) += 15;  // only first layer of YoloV2tiny
				num_YOLO_SLICE_dsp += 1;

				for(int j = 0; j < Model_Par_List.size(); j++) {
					if(Model_Par_List[j].id == 'y' && Model_Par_List[j].num_layers == 4 && Model_Par_List[j].device[0] == 'D') {
						selected = &Model_Par_List[j]; 
					}
				}	
				EnqueueTask(selected, &Batch_queue[i], EMERGENCY_OFF);

		}


		else { // Non-sliced task
		
		if(selected->device[0] == 'B') {
			selected->exp_runtime = selected->BIG_runtime[0]; // runtime
			(*vBIG_runtime) += selected->BIG_runtime[0];
			selected->exp_latency = (*vBIG_runtime); // latency

			if(DeadlineCheck(selected->deadline, (*vBIG_runtime)) == 1)
				EnqueueTask(selected, &Batch_queue[i], EMERGENCY_OFF);
			else 
				EnqueueTask(selected, &Batch_queue[i], EMERGENCY_ON);
		}
		else if(selected->device[0] == 'G') {
			selected->exp_runtime = selected->GPU_runtime[0]; // runtime
			(*vGPU_runtime) += selected->GPU_runtime[0];
			selected->exp_latency = (*vGPU_runtime); // latency
			if(DeadlineCheck(selected->deadline, (*vGPU_runtime)) == 1)
				EnqueueTask(selected, &Batch_queue[i], EMERGENCY_OFF);
			else 
				EnqueueTask(selected, &Batch_queue[i], EMERGENCY_ON);
		}
		else if(selected->device[0] == 'D') {
			selected->exp_runtime = selected->DSP_runtime[0]; // runtime
			(*vDSP_runtime) += selected->DSP_runtime[0];
			selected->exp_latency = (*vDSP_runtime); // latency
			if(DeadlineCheck(selected->deadline, (*vDSP_runtime)) == 1)
				EnqueueTask(selected, &Batch_queue[i], EMERGENCY_OFF);
			else 
				EnqueueTask(selected, &Batch_queue[i], EMERGENCY_ON);
		}
		} // Non-sliced tasks end
	}

}


void SLO_MAEL(vector<Task>& Batch_queue, int *vBIG_runtime, int * vGPU_runtime, int* vDSP_runtime) {
	vector<vector<Model_Parameter> > candidate_set;
	vector<Combi_R> All_combi;

	// Prepare all candidate set
	for(int i = 0; i < Batch_queue.size(); i++) {
		vector<Model_Parameter> candidate;
		char app_id = Batch_queue[i].id;

		for(int j = 0; j < Model_Par_List.size(); j++) {
			if(Model_Par_List[j].id == app_id && Model_Par_List[j].num_layers == 1) 
			{
				candidate.push_back( Model_Par_List[j] ); 
			}
		}	
		candidate_set.push_back( candidate );	
	}
	vector<vector<int> > all_cand_idx; 	
	vector<vector<int> > all_combi; 	
		
	for(int i = 0; i < candidate_set.size(); i++) {
		vector<int> cand_idx;	
		vector<Model_Parameter> model_cand = candidate_set[i];	
		for(int j = 0; j < model_cand.size(); j++) 
			cand_idx.push_back(j);
		all_cand_idx.push_back(cand_idx);	
	}
	all_combi = Cartesian(all_cand_idx);	

	for(int i = 0; i < all_combi.size(); i++) {
		int idx;
		float sum_latency = 0;
		int slo_satisfied = 0;
		int sum_vio_runtime = 0;

		int vBIG_sum_tmp = *vBIG_runtime;
		int vGPU_sum_tmp = *vGPU_runtime;
		int vDSP_sum_tmp = *vDSP_runtime;

		for(int j = 0; j < all_combi[i].size(); j++) {
			vector<Model_Parameter> model_cand = candidate_set[j];	
			idx = all_combi[i][j]; 		
			sum_vio_runtime = 0;

			// calculate if new task is run on each device
			if(model_cand[idx].device[0] == 'B') {
				vBIG_sum_tmp += model_cand[idx].BIG_runtime[0];
				sum_latency += vBIG_sum_tmp;	
				int vio_runtime = DeadlineCheck(model_cand[idx].deadline, vBIG_sum_tmp); 
				sum_vio_runtime += vio_runtime;
			}
			else if(model_cand[idx].device[0] == 'G') {
				vGPU_sum_tmp += model_cand[idx].GPU_runtime[0];
				sum_latency += vGPU_sum_tmp;	
				int vio_runtime = DeadlineCheck(model_cand[idx].deadline, vGPU_sum_tmp); 
				sum_vio_runtime += vio_runtime;
			}
			else if(model_cand[idx].device[0] == 'D') {
				vDSP_sum_tmp += model_cand[idx].DSP_runtime[0];
				sum_latency += vDSP_sum_tmp;	
				int vio_runtime = DeadlineCheck(model_cand[idx].deadline, vDSP_sum_tmp);
				sum_vio_runtime += vio_runtime;
			}

		}

		Combi_R* new_combi = (Combi_R *)malloc(sizeof(Combi_R));
		new_combi->idx = i;
		new_combi->sum_latency = sum_latency;
		new_combi->SLO_satisfied = slo_satisfied;
		new_combi->SLO_vio_runtime = sum_vio_runtime;
		All_combi.push_back(*new_combi);
	
	}

	sort(All_combi.begin(), All_combi.end(), SLO_CMP);
	int min_idx = All_combi[0].idx;

	// Update Device Runtime and Generate Task 
	for(int i = 0; i < all_combi[min_idx].size(); i++)  {
		int idx = all_combi[min_idx][i];
		Model_Parameter* selected = &candidate_set[i][idx];			
		
		if(selected->device[0] == 'B') {
			selected->exp_runtime = selected->BIG_runtime[0]; // runtime
			(*vBIG_runtime) += selected->BIG_runtime[0];
			selected->exp_latency = (*vBIG_runtime); // latency

			if(DeadlineCheck(selected->deadline, (*vBIG_runtime)) == 1)
				EnqueueTask(selected, &Batch_queue[i], EMERGENCY_OFF);
			else 
				EnqueueTask(selected, &Batch_queue[i], EMERGENCY_ON);
		}
		else if(selected->device[0] == 'G') {
			selected->exp_runtime = selected->GPU_runtime[0]; // runtime
			(*vGPU_runtime) += selected->GPU_runtime[0];
			selected->exp_latency = (*vGPU_runtime); // latency
			if(DeadlineCheck(selected->deadline, (*vGPU_runtime)) == 1) EnqueueTask(selected, &Batch_queue[i], EMERGENCY_OFF);
			else 
				EnqueueTask(selected, &Batch_queue[i], EMERGENCY_ON);
		}
		else if(selected->device[0] == 'D') {
			selected->exp_runtime = selected->DSP_runtime[0]; // runtime
			(*vDSP_runtime) += selected->DSP_runtime[0];
			selected->exp_latency = (*vDSP_runtime); // latency
			if(DeadlineCheck(selected->deadline, (*vDSP_runtime)) == 1)
				EnqueueTask(selected, &Batch_queue[i], EMERGENCY_OFF);
			else 
				EnqueueTask(selected, &Batch_queue[i], EMERGENCY_ON);
		}
	}
}


void RequestManager(string algo_cmd, int batch_window, vector<Task> Request_queue) {
    	struct timeval tp; 
	long int cur_time;
	long int init_time;

        int cur_window = -1; 
        int iter_cnt = 0;
	int START_flag = 1;

	int vBIG_runtime = 0;
	int vGPU_runtime = 0;
	int vDSP_runtime = 0;
	
       	gettimeofday(&tp, NULL);  // Added
	init_time = tp.tv_sec * 1000 + tp.tv_usec / 1000; 

        while(Request_queue.size() > 0 || Batch_queue.size() > 0){ 
		if(vBIG_runtime - batch_window > 0) vBIG_runtime -= batch_window;
		else vBIG_runtime = 0;
		if(vGPU_runtime - batch_window > 0) vGPU_runtime -= batch_window;
		else vGPU_runtime = 0;
		if(vDSP_runtime - batch_window > 0) vDSP_runtime -= batch_window;
		else vDSP_runtime = 0;
	
                if(iter_cnt % batch_window == 0) {
			cur_window = iter_cnt;
       	        	while(Request_queue.size() > 0){ 
                       		if(Request_queue.front().arrival_time <= cur_window) {
                               		Task task = Request_queue.front();
			
					Task* new_task = new Task(task.id, task.arrival_time);

        				gettimeofday(&tp, NULL);  // Added
					new_task->batch_enqueue_time = tp.tv_sec * 1000 + tp.tv_usec / 1000; 
					//new_task->real_arrival_time = new_task->batch_enqueue_time - init_time; 
				
					if(START_flag == 1) { 
						Write_file << "START_TIME: " << new_task->batch_enqueue_time << endl;
						START_flag = 0;
					}
                                	//cout << "START_time: " <<  task.id << " " << task.arrival_time << " " << task.batch_enqueue_time - init_time << endl;
                                	//Write_file << "START_time: " <<  task.id << " " << task.arrival_time << " " << task.batch_enqueue_time << endl;

					Batch_queue.push_back(*new_task);
                                	Request_queue.erase(Request_queue.begin()); // erase first element
                        	}
                        	else
                                	break;
                	}
		}

		if(Batch_queue.size() > 0) {
			if(algo_cmd.compare("my") == 0) {
				MAEL(Batch_queue, &vBIG_runtime, &vGPU_runtime, &vDSP_runtime);
				Batch_queue.clear();
			}
			else if(algo_cmd.compare("slo") == 0) {
				SLO_MAEL(Batch_queue, &vBIG_runtime, &vGPU_runtime, &vDSP_runtime);
				Batch_queue.clear();
			}
			else if(algo_cmd.compare("slo_div") == 0) {
				PSLO_MAEL(Batch_queue, &vBIG_runtime, &vGPU_runtime, &vDSP_runtime);
				Batch_queue.clear();
			}
		}
		//cout << iter_cnt << ": " << vBIG_runtime << " " << vGPU_runtime << " " << vDSP_runtime << endl;

        	gettimeofday(&tp, NULL);  // Added
		cur_time = tp.tv_sec * 1000 + tp.tv_usec / 1000; 
		cur_time = cur_time - init_time;
                //cout << "Cur_time: " << cur_time << endl;
                //cout << "iter_cnt: " << iter_cnt << endl;

		//Write_file << "[ " << vBIG_runtime << " " << vGPU_runtime << " " << vDSP_runtime << " ]" << endl;
                //Write_file << "Req_time: " <<  iter_cnt << endl;
                //Write_file << "Cur_time: " << cur_time << endl;
	
                iter_cnt += batch_window; 

		if(iter_cnt - cur_time > 0)
                	usleep(1000 * (iter_cnt - cur_time) );
        }

}

// Layer Execution (start from input tensor)
zdl::DlSystem::TensorMap LayerExecution(std::vector<std::unique_ptr<zdl::SNPE::SNPE>>& model_ptr,std::unique_ptr<zdl::DlSystem::ITensor>& input_ptr, int index, char app_id, char dev) {
    	struct timeval tp;
        long int before_part, after_part;
        gettimeofday(&tp, NULL);  
   	before_part = tp.tv_sec * 1000 + tp.tv_usec / 1000; 

	auto& SNPE = model_ptr; 
	auto& input = input_ptr;
	bool execStatus = false;

        zdl::DlSystem::TensorMap outputTensorMap;

	execStatus = SNPE.at(index)->execute(input.get(), outputTensorMap); 
        gettimeofday(&tp, NULL);  
        after_part = tp.tv_sec * 1000 + tp.tv_usec / 1000;
        Write_file << app_id  << " " << dev << " Execute " << ": " << after_part - before_part << " ms" <<  std::endl;

	return outputTensorMap;	
}

// Layer Execution (start from mid tensor)
zdl::DlSystem::TensorMap LayerExecution(std::vector<std::unique_ptr<zdl::SNPE::SNPE>>& model_ptr,zdl::DlSystem::TensorMap& midTensorMap, int index, char app_id, char dev) {
    	struct timeval tp; 
        long int before_part, after_part; 
        gettimeofday(&tp, NULL); 
   	before_part = tp.tv_sec * 1000 + tp.tv_usec / 1000; 

	auto& SNPE = model_ptr; 
	bool execStatus = false;
        zdl::DlSystem::TensorMap outputTensorMap;

	execStatus = SNPE.at(index)->execute(midTensorMap, outputTensorMap); 

        gettimeofday(&tp, NULL);  
        after_part = tp.tv_sec * 1000 + tp.tv_usec / 1000; 
        Write_file << app_id  << " " << dev << " Execute " << ": " << after_part - before_part << " ms" <<  std::endl;

	return outputTensorMap;	
}

void RunTask(Task* task){

    struct timeval tp; // Added

    gettimeofday(&tp, NULL);  // Added
    long int before = tp.tv_sec * 1000 + tp.tv_usec / 1000; 
	
    if(task->id == 'a') {
    	midTensorMap_alexnet = LayerExecution(SNPE_alexnet, alexnet_inputTensor[task->SNPE_index], task->SNPE_index, task->id, task->dev);
    }
    else if(task->id == 'v'){
	
    	if(task->layer_num == 0) {
		if(task->dev == 'G')
       		 	midTensorMap_vgg = LayerExecution(SNPE_vgg, vgg_inputTensor[0], task->SNPE_index, task->id, task->dev);
		else if(task->dev == 'D')
       		 	midTensorMap_vgg = LayerExecution(SNPE_vgg, vgg_inputTensor[1], task->SNPE_index, task->id, task->dev);
	}
    	else if(task->layer_num == 1) 
 	   	midTensorMap_vgg2 = LayerExecution(SNPE_vgg, midTensorMap_vgg, task->SNPE_index, task->id, task->dev);
    	else if(task->layer_num == 2) 
    		midTensorMap_vgg3 = LayerExecution(SNPE_vgg, midTensorMap_vgg2, task->SNPE_index, task->id, task->dev);
    	else if(task->layer_num == 3) 
    		LayerExecution(SNPE_vgg, midTensorMap_vgg3, task->SNPE_index, task->id, task->dev);
    }
    else if(task->id == 'p'){
    	midTensorMap_pos = LayerExecution(SNPE_pos, pos_inputTensor[task->SNPE_index], task->SNPE_index, task->id, task->dev);
    }
    else if(task->id == 'l'){
    	midTensorMap_mnist = LayerExecution(SNPE_mnist, mnist_inputTensor[task->SNPE_index], task->SNPE_index, task->id, task->dev);
    }
    else if(task->id == 'g'){
    	midTensorMap_googlenet = LayerExecution(SNPE_googlenet, googlenet_inputTensor[task->SNPE_index], task->SNPE_index, task->id, task->dev);
    }
    else if(task->id == 'r'){
    	midTensorMap_resnet = LayerExecution(SNPE_resnet, resnet_inputTensor[task->SNPE_index], task->SNPE_index, task->id, task->dev);
    }
    else if(task->id == 'm'){
    	midTensorMap_mobilenet = LayerExecution(SNPE_mobilenet, mobilenet_inputTensor[task->SNPE_index], task->SNPE_index, task->id, task->dev);
    }
    else if(task->id == 's'){
    	midTensorMap_squeezenet = LayerExecution(SNPE_squeezenet, squeezenet_inputTensor[task->SNPE_index], task->SNPE_index, task->id, task->dev);
    }
    else if(task->id == 'y'){

    	if(task->layer_num == 0) {
		if(task->dev == 'G')
       		 	midTensorMap_yolov2 = LayerExecution(SNPE_yolov2, yolov2_inputTensor[0], task->SNPE_index, task->id, task->dev);
		else if(task->dev == 'D')
       		 	midTensorMap_yolov2 = LayerExecution(SNPE_yolov2, yolov2_inputTensor[1], task->SNPE_index, task->id, task->dev);
	}

    	else if(task->layer_num == 1) 
    		midTensorMap_yolov22 = LayerExecution(SNPE_yolov2, midTensorMap_yolov2, task->SNPE_index, task->id, task->dev);
    	else if(task->layer_num == 2) 
    		midTensorMap_yolov23 = LayerExecution(SNPE_yolov2, midTensorMap_yolov22, task->SNPE_index, task->id, task->dev);
    	else if(task->layer_num == 3) 
    		LayerExecution(SNPE_yolov2, midTensorMap_yolov23, task->SNPE_index, task->id, task->dev);
    }
    else if(task->id == 'f'){
    	midTensorMap_frcnn = LayerExecution(SNPE_frcnn, frcnn_inputTensor[task->SNPE_index], task->SNPE_index, task->id, task->dev);
    }

    gettimeofday(&tp, NULL);  // Added
    task->after_task_scheduler_time = tp.tv_sec * 1000 + tp.tv_usec / 1000; 

    int real_latency = task->after_task_scheduler_time - task->batch_enqueue_time;
    int real_runtime = task->after_task_scheduler_time - before;

    if(task->layer_num == 0) {
	int parent_idx = task->task_idx;
   	for(int i = 0; i < Parent_queue.size(); i++) {
 	if(parent_idx == Parent_queue[i]) {
			Parent_queue.erase(Parent_queue.begin() + i);
			break;
		}
    	}
    }
    else if(task->layer_num == 1) {
	int pparent_idx = task->task_idx;
    	for(int i = 0; i < PParent_queue.size(); i++) {
 		if(pparent_idx == PParent_queue[i]) {
			PParent_queue.erase(PParent_queue.begin() + i);
			break;
		}
	}
    }
    else if(task->layer_num == 2) {
    	int ppparent_idx = task->task_idx;
    	for(int i = 0; i < PPParent_queue.size(); i++) {
 		if(ppparent_idx == PPParent_queue[i]) {
			PPParent_queue.erase(PPParent_queue.begin() + i);
			break;
		}
    	}
    }


    std::cout << task->layer_num + 1 << "/" << task->total_layer_num << " "  << task->id << task->task_idx << " Latency: " << real_latency << " ms " << task->dev << " " << task->arrival_time <<  std::endl;
	
    if(task->dev == 'B'){
    	Write_file_BIG << task->id << task->task_idx << " Latency: " << real_latency << " ms " << task->dev << " " << task->arrival_time <<  std::endl;  

	if(task->total_layer_num == task->layer_num + 1) {
		Finished_BIG++;
	}
	BIG_RUN = 0; 
    }
		
    else if(task->dev == 'G'){
    	Write_file_GPU << task->id << task->task_idx << " Latency: " << real_latency << " ms " << task->dev << " " << task->arrival_time <<  std::endl; 
	if(task->total_layer_num == task->layer_num + 1) {
		Finished_GPU++;
		if(task->total_layer_num == 4) {
			if(task->id == 'y') num_YOLO_SLICE_gpu -= 1;
			else if(task->id == 'v') num_VGG_SLICE_gpu -= 1;
		}
	}
	GPU_RUN = 0;
    }
    else if(task->dev == 'D'){
    	Write_file_DSP << task->id << task->task_idx << " Latency: " << real_latency << " ms " << task->dev << " " << task->arrival_time <<  std::endl;
	if(task->total_layer_num == task->layer_num + 1) {
		Finished_DSP++;
		if(task->total_layer_num == 4) {
			if(task->id == 'y') num_YOLO_SLICE_dsp -= 1;
			else if(task->id == 'v') num_VGG_SLICE_dsp -= 1;
		}
	}
	DSP_RUN = 0;
    }
}

// Create New task for layer execution
Task* CreateNewTask(Task* taskinqueue) {
	Task* task = new Task(taskinqueue->id, taskinqueue->arrival_time);
	task->total_layer_num = taskinqueue->total_layer_num;
	task->layer_num = taskinqueue->layer_num;
	task->batch_size = taskinqueue->batch_size;
	task->SNPE_index = taskinqueue->SNPE_index;
	task->dev = taskinqueue->dev;
	task->task_idx = taskinqueue->task_idx;
	task->batch_enqueue_time = taskinqueue->batch_enqueue_time;
	task->deadline = taskinqueue->deadline;

	task->is_vio = taskinqueue->is_vio;
	task->exp_latency = taskinqueue->exp_latency;
	task->exp_runtime = taskinqueue->exp_runtime;

	return task;
}


float CalculateRemainingTime(Task* task, long int cur_time) {
	float remaining_time;
	
	if(task->total_layer_num == 1) 
		remaining_time = task->deadline - (cur_time - task->batch_enqueue_time + task->runtime);
	else if(task->total_layer_num == 4) 
		remaining_time = task->deadline - (cur_time - task->batch_enqueue_time + task->rruntime);
		
	return remaining_time;
}

int PSLO_MAEL_Scheduler(vector<Task> DeviceQueue) {
	vector<int> possible_index;	
	int min_runtime = 999999;

	int s_idx = -1;
	struct timeval tp; 
	long int cur_time;

	gettimeofday(&tp, NULL);  // Added
	cur_time = tp.tv_sec * 1000 + tp.tv_usec / 1000; 

	for(int i = 0; i < DeviceQueue.size(); i++) {
		DeviceQueue[i].wruntime = CalculateRemainingTime(&DeviceQueue[i], cur_time);
	}

	for(int i = 0; i < DeviceQueue.size(); i++) {
		if(DeviceQueue[i].layer_num == 0)  {
			s_idx = i;
			possible_index.push_back(s_idx);
		}

		else if(DeviceQueue[i].layer_num == 1) {
			int pid = DeviceQueue[i].task_idx;
			bool parent_finish = true;

			for(int j = 0; j < Parent_queue.size(); j++) {
				if(pid == Parent_queue[j]) {
					parent_finish = false;
					break;
				}
			}
			if(parent_finish == true) {
				s_idx = i;
				possible_index.push_back(s_idx);
			}
		}

		else if(DeviceQueue[i].layer_num == 2) {
			int pid = DeviceQueue[i].task_idx;
			bool parent_finish = true;

			for(int j = 0; j < PParent_queue.size(); j++) {
				if(pid == PParent_queue[j]) {
					parent_finish = false;
					break;
				}
			}
			if(parent_finish == true) {
				s_idx = i;
				possible_index.push_back(s_idx);
			}
		}

		else if(DeviceQueue[i].layer_num == 3) {
			int pid = DeviceQueue[i].task_idx;
			bool parent_finish = true;

			for(int j = 0; j < PPParent_queue.size(); j++) {
				if(pid == PPParent_queue[j]) {
					parent_finish = false;
					break;
				}
			}
			if(parent_finish == true) {
				s_idx = i;
				possible_index.push_back(s_idx);
			}
		}	
	}

	for(int i = 0; i < possible_index.size(); i++) {
		int index = possible_index[i]; 
		if(min_runtime > DeviceQueue[index].wruntime){ 
			min_runtime = DeviceQueue[index].wruntime;	
			s_idx = index;
		}
	}
	return s_idx;
}


void Task_scheduler_my(int Total_task, string algo_cmd) {

    int weight = 10;

    thread qthreads[3]; // BIG, GPU, DSP

    struct timeval tp; 
    long int cur_time;

    while(1) { 
    	if(BIG_queue.size() + GPU_queue.size() + DSP_queue.size() > 0) {

		if(BIG_queue.size() != 0 && BIG_RUN == 0) {
			int s_idx = -1; // selected idx
			int min_runtime = 999999;

			if(algo_cmd.compare("my") == 0) {
				for(int i = 0; i < BIG_queue.size(); i++) {
					if(min_runtime > BIG_queue[i].runtime){ 
						min_runtime = BIG_queue[i].runtime;	
						s_idx = i;
					}
				}
				for(int i = 0; i < BIG_queue.size(); i++)
					BIG_queue[i].runtime /= weight;	
			}
			else if(algo_cmd.compare("slo") == 0) {
      		 		gettimeofday(&tp, NULL);  // Added
				cur_time = tp.tv_sec * 1000 + tp.tv_usec / 1000; 
				// init remaining time
				for(int i = 0; i < BIG_queue.size(); i++) 
					BIG_queue[i].wruntime = BIG_queue[i].deadline - (cur_time - BIG_queue[i].batch_enqueue_time + BIG_queue[i].runtime);

				for(int i = 0; i < BIG_queue.size(); i++) {
					if(min_runtime > BIG_queue[i].wruntime){ 
						min_runtime = BIG_queue[i].wruntime;	
						s_idx = i;
					}
				}
			}
		
			if(s_idx == -1) // first element of the queue
				s_idx = 0;
	
			Task* task = CreateNewTask(&BIG_queue[s_idx]);
			BIG_RUN = 1;
			qthreads[0] = thread(RunTask, task);
			qthreads[0].detach();
			BIG_queue.erase(BIG_queue.begin() + s_idx);
		}
		if(GPU_queue.size() != 0 && GPU_RUN == 0) {
			int s_idx = -1; // selected idx
			int min_runtime = 999999;

			if(algo_cmd.compare("my") == 0) {
				for(int i = 0; i < GPU_queue.size(); i++) {
					if(min_runtime > GPU_queue[i].runtime){ 
						min_runtime = GPU_queue[i].runtime;	
						s_idx = i;
					}
				}
				for(int i = 0; i < GPU_queue.size(); i++) 
					GPU_queue[i].runtime /= weight;	
			}
			else if(algo_cmd.compare("slo") == 0) {
      		 		gettimeofday(&tp, NULL);  // Added
				cur_time = tp.tv_sec * 1000 + tp.tv_usec / 1000; 
				// init remaining time
				for(int i = 0; i < GPU_queue.size(); i++) 
					GPU_queue[i].wruntime = GPU_queue[i].deadline - (cur_time - GPU_queue[i].batch_enqueue_time + GPU_queue[i].runtime);

				for(int i = 0; i < GPU_queue.size(); i++) {
					if(min_runtime > GPU_queue[i].wruntime){ 
						min_runtime = GPU_queue[i].wruntime;	
						s_idx = i;
					}
				}
			}

			else if(algo_cmd.compare("slo_div") == 0) {
				s_idx = PSLO_MAEL_Scheduler(GPU_queue);
			} 
			
	
			if(s_idx == -1) // first elemnet of the queue
				s_idx = 0;

			Task* task = CreateNewTask(&GPU_queue[s_idx]);
			GPU_RUN = 1;
			qthreads[1] = thread(RunTask, task);
			qthreads[1].detach();
			GPU_queue.erase(GPU_queue.begin() + s_idx);
		}
		if(DSP_queue.size() != 0 && DSP_RUN == 0) {
			int s_idx = -1; // selected idx
			int min_runtime = 999999;
		
			if(algo_cmd.compare("my") == 0) {
				for(int i = 0; i < DSP_queue.size(); i++) {
					if(min_runtime > DSP_queue[i].runtime){ 
						min_runtime = DSP_queue[i].runtime;	
						s_idx = i;
					}
				}
				for(int i = 0; i < DSP_queue.size(); i++) 
					DSP_queue[i].runtime /= weight;	
			}
			else if(algo_cmd.compare("slo") == 0) {
      		 		gettimeofday(&tp, NULL);  // Added
				cur_time = tp.tv_sec * 1000 + tp.tv_usec / 1000; 
				// init remaining time
				for(int i = 0; i < DSP_queue.size(); i++) 
					DSP_queue[i].wruntime = DSP_queue[i].deadline - (cur_time - DSP_queue[i].batch_enqueue_time + DSP_queue[i].runtime);

				for(int i = 0; i < DSP_queue.size(); i++) {
					if(min_runtime > DSP_queue[i].wruntime){ 
						min_runtime = DSP_queue[i].wruntime;	
						s_idx = i;
					}
				}
			}

			else if(algo_cmd.compare("slo_div") == 0) {
				s_idx = PSLO_MAEL_Scheduler(DSP_queue);
			} 
		
			if(s_idx == -1) // first element of the queue
				s_idx = 0;

			Task* task = CreateNewTask(&DSP_queue[s_idx]);

			DSP_RUN = 1;
			qthreads[2] = thread(RunTask, task);
			qthreads[2].detach();
			DSP_queue.erase(DSP_queue.begin() + s_idx);
		}
    	} 
/*
	if(sch_end_time - sch_start_time > 10) {
		cout << "TIME OUT" << endl;	
    		Write_file << "TIME_OUT" << endl;
		sleep(1);
		break;
	}
*/
   	//cout << "Finished: " << Finished_BIG << " " <<  Finished_GPU << " " <<  Finished_DSP  << " = " << Finished_BIG + Finished_GPU + Finished_DSP  << " / " << Total_task << endl ;

 	if(Finished_BIG + Finished_GPU + Finished_DSP >= Total_task) { 
		sleep(1);
      	 	gettimeofday(&tp, NULL);  // Added
		long int end_time = tp.tv_sec * 1000 + tp.tv_usec / 1000; 
    		Write_file << "END_TIME: " << end_time  << endl;
		break;
	}
		
    } // while loop

    if (qthreads[0].joinable()) qthreads[0].join();
    if (qthreads[1].joinable()) qthreads[1].join();
    if (qthreads[2].joinable()) qthreads[2].join();
}




int main(int argc, char** argv)
{
    cout << "Usage: snpe-sample.. <algo_cmd> <input_name> <deadlineN> <batch_window>" << endl;
    cout << "Example: snpe-sample my poisson 10 10" << endl;

    string algo_cmd = argv[1];
    string input_name = argv[2];
    int deadline_n = stoi(argv[3]); 
    int batch_window = stoi(argv[4]); 

//    string in_dir_name = "/home/wonik/Downloads/snpe-1.25.1.310/exper_result/ATC20/Inputfiles/poisson_avlg/";
//   string out_dir_name = "/home/wonik/Downloads/snpe-1.25.1.310/exper_result/ATC20/Output/poisson_avlg/";
    string in_dir_name = "/data/local/tmp/request_file/" + input_name  +"I/";
    string out_dir_name = "/data/local/tmp/request_file/" + input_name + algo_cmd + "_O/";
    string in_filepath;
    string out_filepath;
    string app_list;
    vector<string> req_inputfiles;

    int trial = 1; 

    ReadDirectory(in_dir_name, req_inputfiles);

    for(int i = 0; i < req_inputfiles.size(); i++) {
	// Init global state
	InitGlobalState();

	// set full in/out path 
	in_filepath = in_dir_name + req_inputfiles[i];
	out_filepath = out_dir_name + "O" + req_inputfiles[i].substr(1, req_inputfiles[i].size());
	// Build Section
	// get App list from request input file name 
	app_list = GetAppList(req_inputfiles[i]);
	SettingModelParameters(algo_cmd, app_list, deadline_n);
	BuildSnpeModelAll();
	cout << "BUILD finished" << endl;
	// Build Section end

	Write_file << "App_list: " << app_list << endl;
	Write_file << "deadlineN: " << deadline_n << endl;
	Write_file << "scheduling_window: " << batch_window << endl;

	sleep(2);
	cout << "Sleep...(2)" << endl;

	Write_file.open(out_filepath+"ALL", ios::out);	
	Write_file_BIG.open(out_filepath + "C", ios::out);	
	Write_file_GPU.open(out_filepath + "G", ios::out);	
	Write_file_DSP.open(out_filepath + "D", ios::out);	
	for(int j = 0; j < trial; j++) {
		vector<Task> Request_queue;

		Write_file << "Trial: " << j+1 << endl;
		Write_file_BIG << "Trial: " << j+1 << endl;
		Write_file_GPU << "Trial: " << j+1 << endl;
		Write_file_DSP << "Trial: " << j+1 << endl;
   		GenerateRequestQueue(Request_queue, in_filepath);
		
		thread RequestManagerThread(RequestManager, algo_cmd, batch_window, Request_queue );
		thread SchedulerManagerThread(Task_scheduler_my, Request_queue.size(), algo_cmd);

		RequestManagerThread.join();
		SchedulerManagerThread.join();
		
	}	

	Write_file.close();
	Write_file_BIG.close();
	Write_file_GPU.close();
	Write_file_DSP.close();
    }

    return 0;
}
