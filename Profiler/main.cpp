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

using namespace std;

const int FAILURE = 1;
const int SUCCESS = 0;

const char* vgg_inputFile320 = "/data/local/tmp/vgg/target_raw_list_320.txt";
const char* vgg_inputFile160 = "/data/local/tmp/vgg/target_raw_list_160.txt";

const char* alexnet_inputFile = "/data/local/tmp/alexnet/target_raw_list_320.txt";
std::string alexnet_OutputDir = "/data/local/tmp/test/part_exper/alexnet_output_part";
std::string alexnet_layerPath = "/data/local/tmp/test/part_exper/alexnet_";

std::string googlenet_OutputDir = "/data/local/tmp/test/part_exper/googlenet_output_part";
std::string googlenet_layerPath = "/data/local/tmp/test/part_exper/googlenet_";

std::string resnet_OutputDir = "/data/local/tmp/test/part_exper/resnet_output_part";
std::string resnet_layerPath = "/data/local/tmp/test/part_exper/resnet_";

const char* vgg_inputFile = "/data/local/tmp/vgg/target_raw_list_10.txt";
std::string vgg_OutputDir = "/data/local/tmp/test/part_exper/vgg_output_part";
std::string vgg_layerPath = "/data/local/tmp/test/part_exper/vgg_";

const char* pos_inputFile = "/data/local/tmp/pos/pos_raw_list_500.txt";
std::string pos_OutputDir = "/data/local/tmp/test/part_exper/pos_output_part";
std::string pos_layerPath = "/data/local/tmp/test/part_exper/pos_";

const char* mnist_inputFile = "/data/local/tmp/mnist/target_list_100.txt";
std::string mnist_OutputDir = "/data/local/tmp/test/part_exper/mnist_output_part";
std::string mnist_layerPath = "/data/local/tmp/test/part_exper/mnist_";

std::string mobilenet_OutputDir = "/data/local/tmp/test/part_exper/mobilenet_output_part";
std::string mobilenet_layerPath = "/data/local/tmp/test/part_exper/mobilenet_";

std::string squeezenet_OutputDir = "/data/local/tmp/test/part_exper/squeezenet_output_part";
std::string squeezenet_layerPath = "/data/local/tmp/test/part_exper/squeezenet_";

const char* yolov2_inputFile = "/data/local/tmp/yolov2/target_raw_list_160.txt";
std::string yolov2_OutputDir = "/data/local/tmp/test/part_exper/yolov2_output_part";
std::string yolov2_layerPath = "/data/local/tmp/test/part_exper/yolov2_";

std::string frcnn_OutputDir = "/data/local/tmp/test/part_exper/frcnn_output_part";
std::string frcnn_layerPath = "/data/local/tmp/test/part_exper/frcnn_";

// total number of layers 
int num_input_layers = -1;


int main(int argc, char** argv)
{
    const char* app_inputFile; 
    std::string app_OutputDir;
    std::string app_layerPath;

    cout << "Usage: snpe_profiler <app_name> <devices> <batch size> <version>" << endl; 
    cout << "Possible App_name: alexnet, vgg, pos, mnist, googlenet, resnet, mobilenet, squeezenet, yolov2, frcnn " << endl;  
    cout << "Possible Devices: 0, 1, 2" << endl;
    cout << "                : (0,CPU), (1,GPU), (2,DSP)" << endl;
    cout << "Possible Batch size: 1, 2, 4, ...." << endl;
    cout << "Possible Version: 1, 2, 4 " << endl;
	
    std::string mode_list = argv[2];
    int batchSize = 1;
    
    // set num input layers 
    num_input_layers = mode_list.size();

    // argv[1]: app_name
    // argv[2]: devices
    // argv[3]: batch size
    // argv[4]: version	

    // set batch size
    if(argv[3] != NULL){
	batchSize = atoi(argv[3]);
    }

    if(strcmp(argv[1], "alexnet") == 0) {
	std::cout << "Alexnet" << std::endl;
 	app_inputFile = alexnet_inputFile;
 	app_OutputDir = alexnet_OutputDir;

	if(num_input_layers == 1)
 		app_layerPath = alexnet_layerPath + "1" + "_dlc/part";
	else if(num_input_layers == 2)
 		app_layerPath = alexnet_layerPath + "2" + "_dlc_ver" + argv[4] + "/part";
	else if(num_input_layers == 4)
 		app_layerPath = alexnet_layerPath + "4" + "_dlc/part";
    }
    else if(strcmp(argv[1], "vgg") == 0) {
	std::cout << "VGG" << std::endl;
 	app_inputFile = vgg_inputFile;
 	app_OutputDir = vgg_OutputDir;

	if(num_input_layers== 1)
 		app_layerPath = vgg_layerPath + "1" + "_dlc/part";
	else if(num_input_layers== 2)
 		app_layerPath = vgg_layerPath + "2" + "_dlc_ver" + argv[4] + "/part";
	else if(num_input_layers== 4)
 		app_layerPath = vgg_layerPath + "4" + "_dlc/part";
    }
    else if(strcmp(argv[1], "pos") == 0) {
	std::cout << "POS" << std::endl;
 	app_inputFile = pos_inputFile;
 	app_OutputDir = pos_OutputDir;

	if(num_input_layers== 1)
 		app_layerPath = pos_layerPath + "1" + "_dlc/part";
	else
 		app_layerPath = pos_layerPath + "2" + "_dlc_ver" + argv[4] + "/part";
    }
    else if(strcmp(argv[1], "mnist") == 0) {
	std::cout << "Mnist" << std::endl;
 	app_inputFile = mnist_inputFile;
 	app_OutputDir = mnist_OutputDir;

	if(num_input_layers== 1)
 		app_layerPath = mnist_layerPath + "1" + "_dlc/part";
	else
 		app_layerPath = mnist_layerPath + "2" + "_dlc_ver" + argv[4] + "/part";
    }
    else if(strcmp(argv[1], "googlenet") == 0) {
	std::cout << "Googlenet" << std::endl;
	if(batchSize < 80) 
 		app_inputFile = vgg_inputFile160;
	else 
 		app_inputFile = vgg_inputFile320;
 	app_OutputDir = alexnet_OutputDir;

	if(num_input_layers == 1)
 		app_layerPath = googlenet_layerPath + "1" + "_dlc/part";
	else
 		app_layerPath = googlenet_layerPath + "2" + "_dlc_ver" + argv[4] + "/part";
    }
    else if(strcmp(argv[1], "resnet") == 0) {
	std::cout << "Resnet" << std::endl;
	if(batchSize < 80) 
 		app_inputFile = vgg_inputFile160;
	else 
 		app_inputFile = vgg_inputFile320;
 	app_OutputDir = alexnet_OutputDir;

	if(num_input_layers == 1)
 		app_layerPath = resnet_layerPath + "1" + "_dlc/part";
	else if(num_input_layers == 2)
 		app_layerPath = resnet_layerPath + "2" + "_dlc_ver" + argv[4] + "/part";
	else
 		app_layerPath = resnet_layerPath + "4" + "_dlc/part";
    }
    else if(strcmp(argv[1], "mobilenet") == 0) {
	std::cout << "Mobilenet" << std::endl;
	if(batchSize < 80) 
 		app_inputFile = vgg_inputFile160;
	else 
 		app_inputFile = vgg_inputFile320;
 	app_OutputDir = alexnet_OutputDir;

	if(num_input_layers == 1)
 		app_layerPath = mobilenet_layerPath + "1" + "_dlc/part";
	else
 		app_layerPath = mobilenet_layerPath + "2" + "_dlc_ver" + argv[4] + "/part";
    }
    else if(strcmp(argv[1], "squeezenet") == 0) {
	std::cout << "SqueezeNet" << std::endl;
 	app_inputFile = alexnet_inputFile;
 	app_OutputDir = alexnet_OutputDir;

	if(num_input_layers == 1)
 		app_layerPath = squeezenet_layerPath + "1" + "_dlc/part";
	else
 		app_layerPath = squeezenet_layerPath + "2" + "_dlc_ver" + argv[4] + "/part";
    }
    else if(strcmp(argv[1], "yolov2") == 0) {
	std::cout << "yoloV2tiny" << std::endl;
 	app_inputFile = yolov2_inputFile;
 	app_OutputDir = yolov2_OutputDir;

	if(num_input_layers == 1)
 		app_layerPath = yolov2_layerPath + "1" + "_dlc/part";
	else if(num_input_layers == 2)
 		app_layerPath = yolov2_layerPath + "2" + "_dlc_ver" + argv[4] + "/part";
	else
 		app_layerPath = yolov2_layerPath + "4" + "_dlc/part";
    }
    else if(strcmp(argv[1], "frcnn") == 0) {
	std::cout << "Faster_RCNN" << std::endl;
 	app_inputFile = alexnet_inputFile;
 	app_OutputDir = frcnn_OutputDir;

	if(num_input_layers == 1)
 		app_layerPath = frcnn_layerPath + "1" + "_dlc/part";
	else
 		app_layerPath = frcnn_layerPath + "2" + "_dlc_ver" + argv[4] + "/part";
    }



	return 0;
}
