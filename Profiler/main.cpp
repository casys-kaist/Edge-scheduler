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
#include <ctime> 
#include <sys/time.h> 
#include <unistd.h> 

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

// << Global variable >>
int num_input_layers = -1; // total number of layers 
enum {UNKNOWN, USERBUFFER_FLOAT, USERBUFFER_TF8, ITENSOR};
enum {CPUBUFFER, GLBUFFER};
bool execStatus = false;
bool usingInitCaching = false;
std::string bufferTypeStr = "USERBUFFER_FLOAT";
std::string userBufferSourceStr = "CPUBUFFER";
std::vector<std::unique_ptr<zdl::SNPE::SNPE>> SNPE;
vector<int> Batch_runtime;
// << Global variable >> 

int Execute(std::string OutputDir, const char* inputFile){

    struct timeval tp; // Added
    long int before_total, after_total; // Added
    long int before_all, after_all; // Added
    long int before_part, after_part; // Added 

    // Check the batch size for the container
    // SNPE 1.16.0 (and newer) assumes the first dimension of the tensor shape
    // is the batch size.

    std::ifstream inputList(inputFile);
    if (!inputList) {
        std::cout << "Input list or dlc file not valid. Please ensure that you have provided a valid input list and dlc for processing. Run snpe-sample with the -h flag for more details" << std::endl;
        std::exit(0); 
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

    //Check if given user buffer source type is valid
    int userBufferSourceType;
    // CPUBUFFER / GLBUFFER supported only for USERBUFFER_FLOAT
    if (bufferType == USERBUFFER_FLOAT)
    {
        if( userBufferSourceStr == "CPUBUFFER" )
        {
            userBufferSourceType = CPUBUFFER;
        }
        else if( userBufferSourceStr == "GLBUFFER" )
        {
#ifndef ANDROID
            std::cout << "GLBUFFER mode is only supported on Android OS" << std::endl;
            std::exit(FAILURE);
#endif
            userBufferSourceType = GLBUFFER;
        }
        else
        {
            std::cout
                  << "Source of user buffer type is not valid. Please run snpe-sample with the -h flag for more details"
                  << std::endl;
            std::exit(FAILURE);
        }
    }
    bool useUserSuppliedBuffers = (bufferType == USERBUFFER_FLOAT || bufferType == USERBUFFER_TF8);

    zdl::DlSystem::TensorShape tensorShape;
    tensorShape = SNPE.at(0)->getInputDimensions();
    size_t batchSize = tensorShape.getDimensions()[0];

#ifdef ANDROID
    size_t bufSize = 0;
    if (userBufferSourceType == GLBUFFER) {
        if(batchSize > 1) {
            std::cerr << "GL buffer source mode does not support batchsize larger than 1" << std::endl;
            std::exit(1);
        }
        bufSize = calcSizeFromDims(tensorShape.getDimensions(), tensorShape.rank(), sizeof(float));
    }
#endif
    //std::cout << "Batch size for the container is " << batchSize << std::endl;

    // Open the input file listing and group input files into batches
    std::vector<std::vector<std::string>> inputs = preprocessInput(inputFile, batchSize);

    // Load contents of input file batches ino a SNPE tensor or user buffer,
    // user buffer include cpu buffer and OpenGL buffer,
    // execute the network with the input and save each of the returned output to a file.
    if(useUserSuppliedBuffers)
    {
       // SNPE allows its input and output buffers that are fed to the network
       // to come from user-backed buffers. First, SNPE buffers are created from
       // user-backed storage. These SNPE buffers are then supplied to the network
       // and the results are stored in user-backed output buffers. This allows for
       // reusing the same buffers for multiple inputs and outputs.
       zdl::DlSystem::UserBufferMap inputMap, outputMap;
       zdl::DlSystem::UserBufferMap midTensorMap1;
       zdl::DlSystem::UserBufferMap midTensorMap2;
       zdl::DlSystem::UserBufferMap midTensorMap3;
       std::vector <std::unique_ptr<zdl::DlSystem::IUserBuffer>> snpeUserBackedInputBuffers, snpeUserBackedOutputBuffers;
       std::unordered_map <std::string, std::vector<uint8_t>> applicationOutputBuffers;
       std::unordered_map <std::string, std::vector<uint8_t>> applicationMidBuffers1;
       std::unordered_map <std::string, std::vector<uint8_t>> applicationMidBuffers2;
       std::unordered_map <std::string, std::vector<uint8_t>> applicationMidBuffers3;

/*
       if( bufferType == USERBUFFER_TF8 )
       {
          createOutputBufferMap(outputMap, applicationOutputBuffers, snpeUserBackedOutputBuffers, snpe, true);

          std::unordered_map <std::string, std::vector<uint8_t>> applicationInputBuffers;
          createInputBufferMap(inputMap, applicationInputBuffers, snpeUserBackedInputBuffers, snpe, true);

          for( size_t i = 0; i < inputs.size(); i++ )
          {
             // Load input user buffer(s) with values from file(s)
             if( batchSize > 1 )
                std::cout << "Batch " << i << ":" << std::endl;
             loadInputUserBufferTf8(applicationInputBuffers, snpe, inputs[i], inputMap);
             // Execute the input buffer map on the model with SNPE
             execStatus = snpe->execute(inputMap, outputMap);
             // Save the execution results only if successful
             if (execStatus == true)
             {
                saveOutput(outputMap, applicationOutputBuffers, OutputDir, i * batchSize, batchSize, true);
             }
             else
             {
                std::cerr << "Error while executing the network." << std::endl;
             }
          }
       }
*/
       if( bufferType == USERBUFFER_FLOAT && num_input_layers == 1)
       {
          createOutputBufferMap(outputMap, applicationOutputBuffers, snpeUserBackedOutputBuffers, SNPE.at(0), false);

          if( userBufferSourceType == CPUBUFFER )
          {
             std::unordered_map <std::string, std::vector<uint8_t>> applicationInputBuffers;
             createInputBufferMap(inputMap, applicationInputBuffers, snpeUserBackedInputBuffers, SNPE.at(0), false);

             for( size_t i = 0; i < inputs.size(); i++ )
             {
                // Load input user buffer(s) with values from file(s)
                if( batchSize > 1 )
          //         std::cout << "Batch " << i << ":" << std::endl;
                //loadInputUserBufferFloat(applicationInputBuffers, SNPE.at(0), inputs[i]); //// ******** original  ********************
                loadInputUserBufferFloat(applicationInputBuffers, SNPE.at(0), inputs[0]);
                // Execute the input buffer map on the model with SNPE
	    gettimeofday(&tp, NULL);  // Added
   	    before_all = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
                execStatus = SNPE.at(0)->execute(inputMap, outputMap);
            gettimeofday(&tp, NULL);  // Added
            after_all = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
            std::cout << "Execute Part Network "<< "0" << ": " << after_all - before_all << " ms" <<  std::endl; //Added 
		
			// warmup 160 / BatchSize
			/*
			if(i == 0) { 	
				for( size_t k = 0; k < 160 / batchSize; k++)
                			execStatus = SNPE.at(0)->execute(inputMap, outputMap);
			}
			*/
			//else {
			    	Batch_runtime.push_back(after_all - before_all);
			//}
			if(Batch_runtime.size() == 10) break;
	
		/*
                // Save the execution results only if successful
                if (execStatus == true)
                {
                   saveOutput(outputMap, applicationOutputBuffers, OutputDir, i * batchSize, batchSize, false);
                }
                else
                {
                   std::cerr << "Error while executing the network." << std::endl;
                }
		*/
             }
          }
       }
       else if( bufferType == USERBUFFER_FLOAT && num_input_layers == 4) {
          createOutputBufferMap(outputMap, applicationOutputBuffers, snpeUserBackedOutputBuffers, SNPE.at(3), false);
          createOutputBufferMap(midTensorMap1, applicationMidBuffers1, snpeUserBackedOutputBuffers, SNPE.at(0), false);
          createOutputBufferMap(midTensorMap2, applicationMidBuffers2, snpeUserBackedOutputBuffers, SNPE.at(1), false);
          createOutputBufferMap(midTensorMap3, applicationMidBuffers3, snpeUserBackedOutputBuffers, SNPE.at(2), false);

          if( userBufferSourceType == CPUBUFFER )
          {
             std::unordered_map <std::string, std::vector<uint8_t>> applicationInputBuffers;
             createInputBufferMap(inputMap, applicationInputBuffers, snpeUserBackedInputBuffers, SNPE.at(0), false);

             for( size_t i = 0; i < inputs.size(); i++ )
	    {
                // Load input user buffer(s) with values from file(s)
                if( batchSize > 1 )
                   std::cout << "Batch " << i << ":" << std::endl;
                loadInputUserBufferFloat(applicationInputBuffers, SNPE.at(0), inputs[i]);
                // Execute the input buffer map on the model with SNPE
	    gettimeofday(&tp, NULL);  // Added
   	    before_all = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
                execStatus = SNPE.at(0)->execute(inputMap, midTensorMap1);
            gettimeofday(&tp, NULL);  // Added
            after_all = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
            std::cout << "Execute Part Network "<< "0" << ": " << after_all - before_all << " ms" <<  std::endl; //Added 

	    gettimeofday(&tp, NULL);  // Added
   	    before_all = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
                execStatus = SNPE.at(1)->execute(midTensorMap1, midTensorMap2);
            gettimeofday(&tp, NULL);  // Added
            after_all = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
            std::cout << "Execute Part Network "<< "1" << ": " << after_all - before_all << " ms" <<  std::endl; //Added 
	    gettimeofday(&tp, NULL);  // Added
   	    before_all = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
                execStatus = SNPE.at(2)->execute(midTensorMap2, midTensorMap3);
            gettimeofday(&tp, NULL);  // Added
            after_all = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
            std::cout << "Execute Part Network "<< "2" << ": " << after_all - before_all << " ms" <<  std::endl; //Added 
	    gettimeofday(&tp, NULL);  // Added
   	    before_all = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
                execStatus = SNPE.at(3)->execute(midTensorMap3, outputMap);
            gettimeofday(&tp, NULL);  // Added
            after_all = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
            std::cout << "Execute Part Network "<< "3" << ": " << after_all - before_all << " ms" <<  std::endl; //Added 

                // Save the execution results only if successful
             }

	  }

	}
    else if( bufferType == USERBUFFER_FLOAT && num_input_layers == 2) {
          createOutputBufferMap(outputMap, applicationOutputBuffers, snpeUserBackedOutputBuffers, SNPE.at(1), false);
          createOutputBufferMap(midTensorMap1, applicationMidBuffers1, snpeUserBackedOutputBuffers, SNPE.at(0), false);

          if( userBufferSourceType == CPUBUFFER )
          {
             std::unordered_map <std::string, std::vector<uint8_t>> applicationInputBuffers;
             createInputBufferMap(inputMap, applicationInputBuffers, snpeUserBackedInputBuffers, SNPE.at(0), false);

             for( size_t i = 0; i < inputs.size(); i++ )
	    {
                // Load input user buffer(s) with values from file(s)
                if( batchSize > 1 )
                   std::cout << "Batch " << i << ":" << std::endl;
                loadInputUserBufferFloat(applicationInputBuffers, SNPE.at(0), inputs[i]);
                // Execute the input buffer map on the model with SNPE
	    gettimeofday(&tp, NULL);  // Added
   	    before_all = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
                execStatus = SNPE.at(0)->execute(inputMap, midTensorMap1);
            gettimeofday(&tp, NULL);  // Added
            after_all = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
            std::cout << "Execute Part Network "<< "0" << ": " << after_all - before_all << " ms" <<  std::endl; //Added 

	    gettimeofday(&tp, NULL);  // Added
   	    before_all = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
                execStatus = SNPE.at(1)->execute(midTensorMap1, outputMap);
            gettimeofday(&tp, NULL);  // Added
            after_all = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
            std::cout << "Execute Part Network "<< "1" << ": " << after_all - before_all << " ms" <<  std::endl; //Added 

                // Save the execution results only if successful
             }

	  }

	}
 
    }
    else if(bufferType == ITENSOR && num_input_layers == 1) { 
        zdl::DlSystem::TensorMap outputTensorMap;

	gettimeofday(&tp, NULL);  // Added
   	before_total = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added

        for (size_t i = 0; i < inputs.size(); i++) {

            // Load input/output buffers with ITensor
            if(batchSize > 1)
                std::cout << "Batch " << i << ":" << std::endl;
            std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = loadInputTensor(SNPE.at(0), inputs[i]);
	    //std::cout << "before" << std::endl;
	    //system("cat /sys/kernel/gpu/gpu_busy");
	    gettimeofday(&tp, NULL);  // Added
   	    before_all = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added

            execStatus = SNPE.at(0)->execute(inputTensor.get(), outputTensorMap); 

            gettimeofday(&tp, NULL);  // Added
            after_all = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
            std::cout << "Execute Part Network "<< "0" << ": " << after_all - before_all << " ms" <<  std::endl; //Added 
	    //std::cout << "after" << std::endl;
	    //system("cat /sys/kernel/gpu/gpu_busy");

            // Save the execution results if execution successful
            if (execStatus == true)
            {
               saveOutput(outputTensorMap, OutputDir, i * batchSize, batchSize);
            }
            else
            {
               std::cerr << "Error while executing the network." << std::endl;
            }

	    //std::cout << "Sleep(1)" << std::endl;
	    //sleep(1);
	}
		
	gettimeofday(&tp, NULL);  // Added
    	after_total = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
    	std::cout << "Execute Total Network: " << after_total - before_total << " ms" <<  std::endl; //Added 
	//std::cout << "Sleep(2)" << std::endl;
	//sleep(2);

    }
    else if(bufferType == ITENSOR && num_input_layers > 1)
    {
        // A tensor map for SNPE execution outputs
        zdl::DlSystem::TensorMap outputTensorMap;
        zdl::DlSystem::TensorMap midTensorMap1;
        zdl::DlSystem::TensorMap midTensorMap2;

	gettimeofday(&tp, NULL);  // Added
   	before_total = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
        
	int cnt = 0;
        int max_cnt = 20;

        for (size_t i = 0; i < inputs.size(); i++) {
	    gettimeofday(&tp, NULL);  // Added
   	    before_all = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added

            // Load input/output buffers with ITensor
            if(batchSize > 1)
                std::cout << "Batch " << i << ":" << std::endl;
            std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = loadInputTensor(SNPE.at(0), inputs[i]);

            gettimeofday(&tp, NULL);  // Added
   	    before_part = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
            // Execute the input tensor on the model with SNPE
	    //std::cout << "before" << std::endl;
	    //system("cat /sys/kernel/gpu/gpu_busy");
            execStatus = SNPE.at(0)->execute(inputTensor.get(), midTensorMap1); 
	    //std::cout << "after" << std::endl;
	    //system("cat /sys/kernel/gpu/gpu_busy");
            gettimeofday(&tp, NULL);  // Added
            after_part = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
            std::cout << "Execute Part Network "<< "0" << ": " << after_part - before_part << " ms" <<  std::endl; //Added 
	    //std::cout << "Sleep(1)" << std::endl;
	    //sleep(1);

	    // num_input_layers > 2
	    if(num_input_layers > 2) {

	  	 for(int j = 1; j < num_input_layers - 1; j++) {
        	    	gettimeofday(&tp, NULL);  // Added
   	    		before_part = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
	   		//std::cout << "before" << std::endl;
	 		//system("cat /sys/kernel/gpu/gpu_busy");
			if(j % 2 == 1)
            			execStatus = SNPE.at(j)->execute(midTensorMap1, midTensorMap2); 
	  		else	
          	  		execStatus = SNPE.at(j)->execute(midTensorMap2, midTensorMap1); 
	  	  	//std::cout << "after" << std::endl;
	    		//system("cat /sys/kernel/gpu/gpu_busy");
            		gettimeofday(&tp, NULL);  // Added
            		after_part = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
            		std::cout << "Execute Part Network "<< j << ": " << after_part - before_part << " ms" <<  std::endl; //Added 
		 	//std::cout << "Sleep(1)" << std::endl;
		 	//sleep(1);
	   	 }

          	 gettimeofday(&tp, NULL);  // Added
   	  	 before_part = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
	   	 //std::cout << "before" << std::endl;
	 	 //system("cat /sys/kernel/gpu/gpu_busy");
	   	 if((num_input_layers - 1) % 2 == 1)
           	 	execStatus = SNPE.at(num_input_layers - 1)->execute(midTensorMap1, outputTensorMap);
	   	 else
           	 	execStatus = SNPE.at(num_input_layers - 1)->execute(midTensorMap2, outputTensorMap);
	  	 //std::cout << "after" << std::endl;
	    	 //system("cat /sys/kernel/gpu/gpu_busy");

          	 gettimeofday(&tp, NULL);  // Added
        	 after_part = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
           	 std::cout << "Execute Part Network "<< num_input_layers - 1 << ": " << after_part - before_part << " ms" <<  std::endl; //Added 
	  	 //std::cout << "after" << std::endl;
	    	 //system("cat /sys/kernel/gpu/gpu_busy");
		 //std::cout << "Sleep(1)" << std::endl;
		 //sleep(1);

	    }
	    else {
          	gettimeofday(&tp, NULL);  // Added
   	  	before_part = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
            	execStatus = SNPE.at(num_input_layers - 1)->execute(midTensorMap1, outputTensorMap);
          	gettimeofday(&tp, NULL);  // Added
        	after_part = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
           	std::cout << "Execute Part Network "<< num_input_layers - 1 << ": " << after_part - before_part << " ms" <<  std::endl; //Added 
		 std::cout << "Sleep(1)" << std::endl;
		 sleep(1);
	    }

            // Save the execution results if execution successful
            if (execStatus == true)
            {
               saveOutput(outputTensorMap, OutputDir, i * batchSize, batchSize);
            }
            else
            {
               std::cerr << "Error while executing the network." << std::endl;
            }
    	    gettimeofday(&tp, NULL);  // Added
    	    after_all = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
    	    std::cout << "Execute All Network: " << after_all - before_all << " ms" <<  std::endl; //Added 
	    std::cout << "Sleep(2)" << std::endl;
	    sleep(2);
	
		// for only 20 iteraction 
	    cnt++;
	    if( cnt > max_cnt) 
		break;	
	  
        }

	gettimeofday(&tp, NULL);  // Added
    	after_total = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
    	std::cout << "Execute Total Network: " << after_total - before_total << " ms" <<  std::endl; //Added 
    }
    return SUCCESS;
}


std::unique_ptr<zdl::SNPE::SNPE> DNN_build(std::string dlc, std::string OutputDir, std::string bufferTypeStr, std::string userBufferSourceStr, std::string mode, int batchSize) {

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
                if (mode[0] == 'g') 
                {
			//cout << "<<<<< GPU >>>>> " << endl;
                    runtime = zdl::DlSystem::Runtime_t::GPU;
                }
                else if (mode[0] == 'd') 
                {
			//cout << "<<<<< DSP >>>>> " << endl;
                    runtime = zdl::DlSystem::Runtime_t::DSP;
                }
                else if (mode[0] == 'c') 
                {
			//cout << "<<<<< CPU >>>>> " << endl;
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

    std::unique_ptr<zdl::SNPE::SNPE> snpe = setBuilderOptions(container, runtime, udlBundle, useUserSuppliedBuffers, platformConfig, usingInitCaching);
    if (snpe == nullptr)
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

    // get Input tensor name
    const auto &strList_opt = snpe->getInputTensorNames();
    if (!strList_opt) throw std::runtime_error("Error obtaining Input tensor names");
    const auto &strList = *strList_opt;
  
    // get tensorShape
    zdl::DlSystem::TensorShape tensorShape;
    tensorShape = snpe->getInputDimensions();
    // set new batch size	
    tensorShape[0] = batchSize;

    // set TensorshapeMap
    zdl::DlSystem::TensorShapeMap tensorShapeMap;
    tensorShapeMap.add(strList.at(0), tensorShape);

  
    std::unique_ptr<zdl::SNPE::SNPE> snpe_final = setBuilderOptions(container, runtime, udlBundle, useUserSuppliedBuffers, platformConfig, tensorShapeMap, usingInitCaching);
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
    auto logger_opt = snpe->getDiagLogInterface();
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

void BuildSetup(std::string app_OutputDir,std::string app_layerPath, std::string mode_list, int batchSize){
    std::string mode = ""; 

    for(int i = 0; i < num_input_layers; i++) {
	stringstream part_num;
	part_num << i;
	std::string app_layerPath_full = app_layerPath + part_num.str() + ".dlc";

	if(mode_list[i] == '0') mode = "cpu";
	else if(mode_list[i] == '1') mode = "gpu";
	else if(mode_list[i] == '2') mode = "dsp";
	
	std::unique_ptr<zdl::SNPE::SNPE> snpe = DNN_build(app_layerPath_full, app_OutputDir, bufferTypeStr, userBufferSourceStr, mode, batchSize);
	SNPE.push_back(std::move(snpe));	
    }
}




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

    BuildSetup(app_OutputDir, app_layerPath, mode_list, batchSize);
    Execute(app_OutputDir, app_inputFile);
	
    int sum_of_elems = 0;
    for(std::vector<int>::iterator it = Batch_runtime.begin(); it != Batch_runtime.end(); ++it)
   	 sum_of_elems += *it;
    cout << batchSize << " " << sum_of_elems / float(Batch_runtime.size()) << endl;

    return 0;
}
