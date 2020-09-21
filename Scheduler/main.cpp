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

int main(int argc, char** argv)
{
    cout << "Usage: snpe-sample.. <algo_cmd> <input_name> <deadlineN> <batch_window>" << endl;
    cout << "Example: snpe-sample my poisson 10 10" << endl;

    string algo_cmd = argv[1];
    string input_name = argv[2];
    int deadline_n = stoi(argv[3]); 
    int batch_window = stoi(argv[4]); 

    string in_dir_name = "/home/wonik/Downloads/snpe-1.25.1.310/exper_result/ATC20/Inputfiles/poisson_avlg/";
//    string in_dir_name = "/data/local/tmp/request_file/" + input_name  +"I/";
//    string out_dir_name = "/data/local/tmp/request_file/" + input_name + algo_cmd + "_O/";

    vector<string> req_inputfiles;
    ReadDirectory(in_dir_name, req_inputfiles);

    for(int i = 0; i < req_inputfiles.size(); i++) {
	cout << req_inputfiles[i] << endl;
    }

    return 0;
}
