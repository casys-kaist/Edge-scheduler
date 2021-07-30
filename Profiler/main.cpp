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

enum {UNKNOWN, USERBUFFER_FLOAT, USERBUFFER_TF8, ITENSOR};
enum {CPUBUFFER, GLBUFFER};
std::vector<std::unique_ptr<zdl::SNPE::SNPE>> SNPE;
bool execStatus = false;
bool usingInitCaching = false;
//std::string bufferTypeStr = "ITENSOR";
std::string bufferTypeStr = "USERBUFFER_FLOAT";
std::string userBufferSourceStr = "CPUBUFFER";
int num_input_layers = -1;

const int FAILURE = 1;
const int SUCCESS = 0;


int DNN_execute(std::string OutputDir, const char* inputFile){

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
       std::vector <std::unique_ptr<zdl::DlSystem::IUserBuffer>> snpeUserBackedInputBuffers, snpeUserBackedOutputBuffers;
       std::unordered_map <std::string, std::vector<uint8_t>> applicationOutputBuffers;

       if( bufferType == USERBUFFER_FLOAT && num_input_layers == 1)
       {
          createOutputBufferMap(outputMap, applicationOutputBuffers, snpeUserBackedOutputBuffers, SNPE.at(0), false);

          if( userBufferSourceType == CPUBUFFER )
          {
             std::unordered_map <std::string, std::vector<uint8_t>> applicationInputBuffers;
             createInputBufferMap(inputMap, applicationInputBuffers, snpeUserBackedInputBuffers, SNPE.at(0), false);

             //for( size_t i = 0; i < inputs.size(); i++ )
             for( size_t i = 0; i < 1000; i++ )
             {
                // Load input user buffer(s) with values from file(s)
                if( batchSize > 1 )
                loadInputUserBufferFloat(applicationInputBuffers, SNPE.at(0), inputs[0]);

                // Execute the input buffer map on the model with SNPE
	    	//gettimeofday(&tp, NULL);  // Added
   	    	//before_all = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
                execStatus = SNPE.at(0)->execute(inputMap, outputMap);
            	//gettimeofday(&tp, NULL);  // Added
            	//after_all = tp.tv_sec * 1000 + tp.tv_usec / 1000; // Added
            	//std::cout << "Execute Part Network "<< "0" << ": " << after_all - before_all << " ms" <<  std::endl; //Added 
		
             }
          }
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
                    runtime = zdl::DlSystem::Runtime_t::GPU;
                }
                else if (mode[0] == 'd') 
                {
                    runtime = zdl::DlSystem::Runtime_t::DSP;
                }
                else if (mode[0] == 'c') 
                {
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


void build_setup(std::string app_OutputDir,std::string app_layerPath, std::string mode_list, int batchSize){
    std::string mode = ""; 

    for(int i = 0; i < mode_list.size(); i++) {
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
    std::string app_layerPath; // argv[1]
    const char* app_inputFile;  // argv[2]
    std::string app_OutputDir; // argv[3]
    std::string mode_list; // argv[4]
    int batchSize = 1; // argv[5]

    // Usage: snpe-trace <Model_dlc_path> <inputfile_path> <output_path> <DEVICEs..> <Batch>
    std::cout <<  "Usage: snpe-trace <Model_dlc_path> <inputfile_path> <output_path> <DEVICEs..> <Batch>" << std::endl;

    if(argv[5] != NULL){
	batchSize = atoi(argv[5]);
    }
	
    num_input_layers = mode_list.size();
    build_setup(app_OutputDir, app_layerPath, mode_list, batchSize);
    DNN_execute(app_OutputDir, app_inputFile);

}

/*
const int FAILURE = 1;
const int SUCCESS = 0;

int main(int argc, char** argv)
{
    enum {UNKNOWN, USERBUFFER_FLOAT, USERBUFFER_TF8, ITENSOR};
    enum {CPUBUFFER, GLBUFFER};

    // Command line arguments
    static std::string dlc = "";
    static std::string OutputDir = "./output/";
    const char* inputFile = "";
    std::string bufferTypeStr = "ITENSOR";
    std::string userBufferSourceStr = "CPUBUFFER";
    static zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
    bool execStatus = false;
    bool usingInitCaching = false;

    // Process command line arguments
    int opt = 0;
    while ((opt = getopt(argc, argv, "hi:d:o:b:s:r:z:c")) != -1)
    {
        switch (opt)
        {
            case 'h':
                std::cout
                        << "\nDESCRIPTION:\n"
                        << "------------\n"
                        << "Example application demonstrating how to load and execute a neural network\n"
                        << "using the SNPE C++ API.\n"
                        << "\n\n"
                        << "REQUIRED ARGUMENTS:\n"
                        << "-------------------\n"
                        << "  -d  <FILE>   Path to the DL container containing the network.\n"
                        << "  -i  <FILE>   Path to a file listing the inputs for the network.\n"
                        << "  -o  <PATH>   Path to directory to store output results.\n"
                        << "\n"
                        << "OPTIONAL ARGUMENTS:\n"
                        << "-------------------\n"
                        << "  -b  <TYPE>   Type of buffers to use [USERBUFFER_FLOAT, USERBUFFER_TF8, ITENSOR] (" << bufferTypeStr << " is default).\n"
                        << "  -r  <RUNTIME> The runtime to be used [gpu, dsp, cpu] (cpu is default). \n"
                        << "  -z  <NUMBER>  The maximum number that resizable dimensions can grow into. \n"
                        << "                Used as a hint to create UserBuffers for models with dynamic sized outputs. Should be a positive integer and is not applicable when using ITensor. \n"
#ifdef ANDROID
                        << "  -s  <TYPE>   Source of user buffers to use [GLBUFFER, CPUBUFFER] (" << userBufferSourceStr << " is default).\n"
                        << "               GL buffer is only supported on Android OS.\n"
#endif
                        << "  -c           Enable init caching to accelerate the initialization process of SNPE. Defaults to disable.\n"
                        << std::endl;

                std::exit(SUCCESS);
            case 'i':
                inputFile = optarg;
                break;
            case 'd':
                dlc = optarg;
                break;
            case 'o':
                OutputDir = optarg;
                break;
            case 'b':
                bufferTypeStr = optarg;
                break;
            case 's':
                userBufferSourceStr = optarg;
                break;
            case 'z':
                setResizableDim(atoi(optarg));
                break;
            case 'r':
                if (strcmp(optarg, "gpu") == 0)
                {
                    runtime = zdl::DlSystem::Runtime_t::GPU;
                }
                else if (strcmp(optarg, "dsp") == 0)
                {
                    runtime = zdl::DlSystem::Runtime_t::DSP;
                }
                else if (strcmp(optarg, "cpu") == 0)
                {
                   runtime = zdl::DlSystem::Runtime_t::CPU;
                }
                else
                {
                   std::cerr << "The runtime option provide is not valid. Defaulting to the CPU runtime." << std::endl;

                }
                break;
            case 'c':
               usingInitCaching = true;
               break;
            default:
                std::cout << "Invalid parameter specified. Please run snpe-sample with the -h flag to see required arguments" << std::endl;
                std::exit(FAILURE);
        }
    }

    // Check if given arguments represent valid files
    std::ifstream dlcFile(dlc);
    std::ifstream inputList(inputFile);
    if (!dlcFile || !inputList) {
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

    runtime = checkRuntime(runtime);
    std::unique_ptr<zdl::DlContainer::IDlContainer> container = loadContainerFromFile(dlc);
    if (container == nullptr)
    {
       std::cerr << "Error while opening the container file." << std::endl;
       std::exit(FAILURE);
    }

    bool useUserSuppliedBuffers = (bufferType == USERBUFFER_FLOAT || bufferType == USERBUFFER_TF8);

    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::DlSystem::PlatformConfig platformConfig;
#ifdef ANDROID
    CreateGLBuffer* glBuffer = nullptr;
    if (userBufferSourceType == GLBUFFER) {
        if(!checkGLCLInteropSupport()) {
            std::cerr << "Failed to get gl cl shared library" << std::endl;
            std::exit(1);
        }
        glBuffer = new CreateGLBuffer();
        glBuffer->setGPUPlatformConfig(platformConfig);
    }
#endif

    snpe = setBuilderOptions(container, runtime, udlBundle, useUserSuppliedBuffers, platformConfig, usingInitCaching);
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

    // Check the batch size for the container
    // SNPE 1.16.0 (and newer) assumes the first dimension of the tensor shape
    // is the batch size.
    zdl::DlSystem::TensorShape tensorShape;
    tensorShape = snpe->getInputDimensions();
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
    std::cout << "Batch size for the container is " << batchSize << std::endl;

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
       std::vector <std::unique_ptr<zdl::DlSystem::IUserBuffer>> snpeUserBackedInputBuffers, snpeUserBackedOutputBuffers;
       std::unordered_map <std::string, std::vector<uint8_t>> applicationOutputBuffers;

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
       else if( bufferType == USERBUFFER_FLOAT )
       {
          createOutputBufferMap(outputMap, applicationOutputBuffers, snpeUserBackedOutputBuffers, snpe, false);

          if( userBufferSourceType == CPUBUFFER )
          {
             std::unordered_map <std::string, std::vector<uint8_t>> applicationInputBuffers;
             createInputBufferMap(inputMap, applicationInputBuffers, snpeUserBackedInputBuffers, snpe, false);

             for( size_t i = 0; i < inputs.size(); i++ )
             {
                // Load input user buffer(s) with values from file(s)
                if( batchSize > 1 )
                   std::cout << "Batch " << i << ":" << std::endl;
                loadInputUserBufferFloat(applicationInputBuffers, snpe, inputs[i]);
                // Execute the input buffer map on the model with SNPE
                execStatus = snpe->execute(inputMap, outputMap);
                // Save the execution results only if successful
                if (execStatus == true)
                {
                   saveOutput(outputMap, applicationOutputBuffers, OutputDir, i * batchSize, batchSize, false);
                }
                else
                {
                   std::cerr << "Error while executing the network." << std::endl;
                }
             }
          }
#ifdef ANDROID
            if(userBufferSourceType  == GLBUFFER) {
                std::unordered_map<std::string, GLuint> applicationInputBuffers;
                createInputBufferMap(inputMap, applicationInputBuffers, snpeUserBackedInputBuffers, snpe);
                GLuint glBuffers = 0;
                for(size_t i = 0; i < inputs.size(); i++) {
                    // Load input GL buffer(s) with values from file(s)
                    glBuffers = glBuffer->convertImage2GLBuffer(inputs[i], bufSize);
                    loadInputUserBuffer(applicationInputBuffers, snpe, glBuffers);
                    // Execute the input buffer map on the model with SNPE
                    execStatus =  snpe->execute(inputMap, outputMap);
                    // Save the execution results only if successful
                    if (execStatus == true) {
                        saveOutput(outputMap, applicationOutputBuffers, OutputDir, i*batchSize, batchSize, false);
                    }
                    else
                    {
                        std::cerr << "Error while executing the network." << std::endl;
                    }
                    // Release the GL buffer(s)
                    glDeleteBuffers(1, &glBuffers);
                }
            }
#endif
       }
    }
    else if(bufferType == ITENSOR)
    {
        // A tensor map for SNPE execution outputs zdl::DlSystem::TensorMap outputTensorMap;

        for (size_t i = 0; i < inputs.size(); i++) {
            // Load input/output buffers with ITensor
            if(batchSize > 1)
                std::cout << "Batch " << i << ":" << std::endl;
            std::unique_ptr<zdl::DlSystem::ITensor> inputTensor = loadInputTensor(snpe, inputs[i]);
            // Execute the input tensor on the model with SNPE
            execStatus = snpe->execute(inputTensor.get(), outputTensorMap);
            // Save the execution results if execution successful
            if (execStatus == true)
            {
               saveOutput(outputTensorMap, OutputDir, i * batchSize, batchSize);
            }
            else
            {
               std::cerr << "Error while executing the network." << std::endl;
            }
        }
    }
    return SUCCESS;
}
*/
