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

using namespace std; 


class Model_Parameter {
public:
	char id;	
	int batch;
	int num_layers;
	int snpe_index;
	string device;
	string ver;
	int deadline;

        Model_Parameter(char _id, int _batch, int _num_layers, string _device, string _ver, int _deadline){
		id = _id;	
		batch = _batch;
		num_layers = _num_layers;
		device = _device;
		ver = _ver;
		deadline = _deadline;
        }

	void SetSnpeIndex(int _snpe_index) {
		snpe_index = _snpe_index;
	}
	
	void PrintParameters() {
		cout << "App id: " << id << endl;	
		cout << "Batch: " << batch << endl;
		cout << "Num layers: " << num_layers << endl; 
		cout << "Snpe index: " << snpe_index << endl;
		cout << "Device: " << device  << endl;
		cout << "Version: " << ver  << endl;
		cout << "Deadline: " << deadline  << endl;
	}
};


// Global constant variable
enum {UNKNOWN, USERBUFFER_FLOAT, USERBUFFER_TF8, ITENSOR};
const int FAILURE = 1;
const int SUCCESS = 0;
bool execStatus = false;
bool usingInitCaching = false;
std::string bufferTypeStr = "ITENSOR";
std::string userBufferSourceStr = "CPUBUFFER";

// Global variable 
ofstream Write_file;
ofstream Write_file_BIG;
ofstream Write_file_GPU;
ofstream Write_file_DSP;

vector<Model_Parameter> Model_Par_List;


void InitGlobalState() {
	Model_Par_List.clear();
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
    			deadline = 91 * deadlineN;	
			if(algo_cmd.compare("slo_div") != 0) {
				model_par = new Model_Parameter('v', 1, 1, "D", "0", deadline); Model_Par_List.push_back(*model_par);
				model_par = new Model_Parameter('v', 1, 1, "G", "0", deadline); Model_Par_List.push_back(*model_par);
			}
			else {
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
    			deadline = 27 * deadlineN;	
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
    			//deadline = 12 * deadlineN;	
    			deadline = 21 * deadlineN;	
			model_par = new Model_Parameter('m', 1, 1, "D", "0", deadline); Model_Par_List.push_back(*model_par);
			model_par = new Model_Parameter('m', 1, 1, "G", "0", deadline); Model_Par_List.push_back(*model_par);
			model_par = new Model_Parameter('m', 1, 1, "B", "0", deadline); Model_Par_List.push_back(*model_par);
		}
		else if(app_list[i] == 's') { // SqueezeNet
    			//deadline = 12 * deadlineN;	
    			deadline = 19 * deadlineN;	
			model_par = new Model_Parameter('s', 1, 1, "D", "0", deadline); Model_Par_List.push_back(*model_par);
			model_par = new Model_Parameter('s', 1, 1, "G", "0", deadline); Model_Par_List.push_back(*model_par);
			model_par = new Model_Parameter('s', 1, 1, "B", "0", deadline); Model_Par_List.push_back(*model_par);
		}
		else if(app_list[i] == 'y') { // yoloV2tiny
    			deadline = 47 * deadlineN;	
			if(algo_cmd.compare("slo_div") != 0) {
				model_par = new Model_Parameter('y', 1, 1, "D", "0", deadline); Model_Par_List.push_back(*model_par);
				model_par = new Model_Parameter('y', 1, 1, "G", "0", deadline); Model_Par_List.push_back(*model_par);
				model_par = new Model_Parameter('y', 1, 1, "B", "0", deadline); Model_Par_List.push_back(*model_par);
			}
			else {
				model_par = new Model_Parameter('y', 1, 1, "D", "0", deadline); Model_Par_List.push_back(*model_par);
				model_par = new Model_Parameter('y', 1, 1, "G", "0", deadline); Model_Par_List.push_back(*model_par);
				model_par = new Model_Parameter('y', 1, 1, "B", "0", deadline); Model_Par_List.push_back(*model_par);
				//model_par = new Model_Parameter('y', 1, 4, "GGGG", "0", deadline); Model_Par_List.push_back(*model_par);
			}
		}
		else if(app_list[i] == 'f') { // faster RCNN
    			//deadline = 20 * deadlineN;	
    			deadline = 23 * deadlineN;	
			model_par = new Model_Parameter('f', 1, 1, "D", "0", deadline); Model_Par_List.push_back(*model_par);
			model_par = new Model_Parameter('f', 1, 1, "G", "0", deadline); Model_Par_List.push_back(*model_par);
			model_par = new Model_Parameter('f', 1, 1, "B", "0", deadline); Model_Par_List.push_back(*model_par);
		}
	}

	// setting snpe start index 
	int snpe_start_index = 0;	
	char prev_id;
	for(int i = 0; i < Model_Par_List.size(); i++) {
		Model_Parameter* mp = &Model_Par_List[i];
		if(mp->id != prev_id) {
			snpe_start_index = 0;	
			prev_id = mp->id;
		}
		snpe_start_index += mp->num_layers;	
		mp->SetSnpeIndex(snpe_start_index - mp->num_layers);
	}
	// end
	
	cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
	cout << Model_Par_List.size() << endl;
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
    {
	cout << "<<<<< GPU >>>>> " << endl;
        runtime = zdl::DlSystem::Runtime_t::GPU;
    }
    else if (device[0] == 'd') 
    {
	cout << "<<<<< DSP >>>>> " << endl;
        runtime = zdl::DlSystem::Runtime_t::DSP;
    }
    else if (device[0] == 'c') 
    {
	cout << "<<<<< CPU >>>>> " << endl;
        runtime = zdl::DlSystem::Runtime_t::CPU;
    }
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
    		if (strList.size() == 1) std::exit(FAILURE);

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


int main(int argc, char** argv)
{
    cout << "Usage: snpe-sample.. <algo_cmd> <input_name> <deadlineN> <batch_window>" << endl;
    cout << "Example: snpe-sample my poisson 10 10" << endl;

    string algo_cmd = argv[1];
    string input_name = argv[2];
    int deadline_n = stoi(argv[3]); 
    int batch_window = stoi(argv[4]); 

    string in_dir_name = "/home/wonik/Downloads/snpe-1.25.1.310/exper_result/ATC20/Inputfiles/poisson_avlg/";
    string out_dir_name = "/home/wonik/Downloads/snpe-1.25.1.310/exper_result/ATC20/Output/poisson_avlg/";
//    string in_dir_name = "/data/local/tmp/request_file/" + input_name  +"I/";
//    string out_dir_name = "/data/local/tmp/request_file/" + input_name + algo_cmd + "_O/";
    string in_filepath;
    string out_filepath;

    string app_list;
  
    vector<string> req_inputfiles;
    ReadDirectory(in_dir_name, req_inputfiles);

    for(int i = 0; i < req_inputfiles.size(); i++) {
	// set full in/out path 
	in_filepath = in_dir_name + req_inputfiles[i];
	out_filepath = out_dir_name + "O" + req_inputfiles[i].substr(1, req_inputfiles[i].size());
	// get App list from request input file name 
	app_list = GetAppList(req_inputfiles[i]);

	cout << in_filepath << endl;	
	cout << out_filepath << endl;
	cout << app_list << endl;

	Write_file.open(out_filepath+"ALL", ios::out);	
	Write_file_BIG.open(out_filepath + "C", ios::out);	
	Write_file_GPU.open(out_filepath + "G", ios::out);	
	Write_file_DSP.open(out_filepath + "D", ios::out);	

	InitGlobalState();
	SettingModelParameters(algo_cmd, app_list, deadline_n);

	Write_file.close();
	Write_file_BIG.close();
	Write_file_GPU.close();
	Write_file_DSP.close();
    }

    return 0;
}
