import copy 
import itertools
import sys
import random
import os
import re
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import Counter


############### DNN MODEL INDEX ##################
#	 0: alexnet
#	 1: vgg
#	 2: lenet
#	 3: googlenet
#	 4: resnet
#	 5: mobilenet
#	 6: squeezenet
#	 7: yolov2
#	 8: fasterrcnn
####################################################

# min runtime
# alexnet, vgg, lenet, googlenet, resnet, mobilenet, squeezenet, yolov2, frcnn
Models_min_runtimes = [20, 91, 5, 16, 36.8, 12.2, 12.6, 47.2, 20.4]

# num_request_based
def GeneratePoissonRequest(req_queue, _lambda, num_requests, max_arr_time):
	
	last_arrival_time = 999999
	while max_arr_time < last_arrival_time:
		req_queue_tmp = []
		_arrival_time = 0
		for i in range(num_requests):
			#Get the next probability value from Uniform(0,1)
			p = random.random()
	
			#Plug it into the inverse of the CDF of Exponential(_lamnbda)
			_inter_arrival_time = -math.log(1.0 - p)/_lambda
	
			#Add the inter-arrival time to the running sum
			_arrival_time = _arrival_time + _inter_arrival_time
			
			req_queue_tmp.append(int(_arrival_time * 1000)) # sec -> ms

		last_arrival_time = req_queue_tmp[-1]	
	
	for i in range(0, len(req_queue_tmp)):
		req_queue.append(req_queue_tmp[i])
		
	req_queue.sort()
	
	return num_requests


def LambdaSetting(model_tot_req, max_arrtime):
	model_lambda = []
	for i in range(0, len(model_tot_req)):
		_lambda = 1000 * ( model_tot_req[i] / max_arrtime )
		model_lambda.append(_lambda)
	#print("lambda: ", model_lambda)
	return model_lambda
	
def GeneratePoissonMain(req_queue, model_tot_req, max_arrtime):
	tot_req = 0
	model_lambda = LambdaSetting(model_tot_req, max_arrtime)

	for i in range(0, len(model_tot_req)):
		tot_req += GeneratePoissonRequest(req_queue[i], model_lambda[i], model_tot_req[i], max_arrtime) 
	return tot_req 
		
	
def GetTotalRequest(selected_model_min_runtimes, max_arrtime, scenario, intensity):
	model_total_req = []
	ideal_req = []

	for i in range(0, len(selected_model_min_runtimes)):
		ideal_req.append(max_arrtime / selected_model_min_runtimes[i])

	for i in range(0, len(scenario)):
		model_total_req.append( int(intensity * (ideal_req[i] * float(scenario[i])/sum(scenario) )))

	return model_total_req
	
# For each iteration
def InitSetting(interval, model, base_lambda, intensity, max_arrtime, scenario):
	selected_model_min_runtimes = []
	req_queue = []

	for i in range(0, len(model)):
		selected_model_min_runtimes.append(Models_min_runtimes[model[i]])

	model_tot_req = GetTotalRequest(selected_model_min_runtimes, max_arrtime, scenario, intensity)

	for i in range(0, len(model)):	
		req_queue.append([])

	return req_queue, model_tot_req
	

def printConfigInfo(interval, model, base_lambda, intensity, max_arrtime):
	print("Interval: ", interval)
	print("Model index: ", model)
	print("Base Lambda: ", base_lambda) 
	print("Intensity: ", intensity)
	print("Max arrival time: ", max_arrtime)

def PrintInfo(interval, model, base_lambda, intensity, max_arrtime, scenario, req_queue):
	print("=======================================================================")
	print("Interval: ", interval)
	print("Model: ", model)
	print("Scenario ratio: ", scenario)
	print("Base Lambda: ", base_lambda) 
	print("Intensity: ", intensity)
	print("Max arrival time: ", max_arrtime)

	for i in range(0, len(req_queue)):
		model_name = ""
		if model[i] == 0: model_name += "alexnet"
		elif model[i] == 1: model_name += "vgg"
		elif model[i] == 2: model_name += "lenet"
		elif model[i] == 3: model_name += "googlenet"
		elif model[i] == 4: model_name += "resnet"
		elif model[i] == 5: model_name += "mobilenet"
		elif model[i] == 6: model_name += "squeezenet"
		elif model[i] == 7: model_name += "yolov2"
		elif model[i] == 8: model_name += "frcnn"
		
		sys.stdout.write(model_name + ": ")
		print(req_queue[i])
	print("=======================================================================")
	

def GetModelIndex(model_list):
	model = []

	for i in range(0, len(model_list)):
		model_name = model_list[i].replace(' ', '')
		if model_name == "alexnet": model.append(0)
		elif model_name == "vgg": model.append(1)
		elif model_name == "lenet": model.append(2)
		elif model_name == "googlenet": model.append(3)
		elif model_name == "resnet": model.append(4)
		elif model_name == "mobilenet": model.append(5)
		elif model_name == "squeezenet": model.append(6)
		elif model_name == "yolov2": model.append(7)
		elif model_name == "fasterrcnn": model.append(8)

	return model

def ReadInputConfigs(configFile):
	interval = [10]
	model = [] 
	base_lambda = [10]
	intensity = []
	max_arrtime = [2000] # 2 sec
	scenarios = [] # 2 sec

	File = open(configFile, "r")
	while True:
		line = File.readline()
		if not line: break
		
		if line.find("Interval") != -1:
			interval = re.findall("\d+", line)
			interval = [round(float(n), 1) for n in interval]
		elif line.find("Model index") != -1:
			model_list = line.rstrip("\n").split(":")[1].split(",")
			model = GetModelIndex(model_list)
		elif line.find("Base Lambda") != -1:
			base_lambda = re.findall("\d+", line)
			base_lambda = [round(float(n), 1) for n in base_lambda]
		elif line.find("Intensity") != -1:
			intensity = re.findall("\d+\.\d+", line)
			intensity = [round(float(n), 1) for n in intensity]
		elif line.find("Max arrival time") != -1:
			max_arrtime = re.findall("\d+", line)
			max_arrtime = [round(float(n), 1) for n in max_arrtime]
		elif line.find("Scenario") != -1:
			scenario = re.findall("\d+", line)
			scenarios.append([int(n) for n in scenario])

	return interval, model, base_lambda, intensity, max_arrtime, scenarios


def GenerateInputFileName(input_full_path, scenario_num, intensity, model, model_tot_req):
	input_file_name = input_full_path + "/I"
	input_file_name += "S" + str(scenario_num) + "_" + str(int(intensity*10)) + "L"
	for i in range(0, len(model_tot_req)):
		#if models_req_numbers[k] == 0: continue
		input_file_name += "_" + str(model_tot_req[i]) 
	#input_file_name += "_" + str(base_lambda)
	
	input_file_name += "+"
	for i in range(0, len(model)):
		if model[i] == 0: input_file_name += "a"
		elif model[i] == 1: input_file_name += "v"
		elif model[i] == 2: input_file_name += "l"
		elif model[i] == 3: input_file_name += "g"
		elif model[i] == 4: input_file_name += "r"
		elif model[i] == 5: input_file_name += "m"
		elif model[i] == 6: input_file_name += "s"
		elif model[i] == 7: input_file_name += "y"
		elif model[i] == 8: input_file_name += "f"
	#print(input_file_name)

	return input_file_name

def ExportFiles(input_file_name, model, req_queue):

	fw = open(input_file_name, "w")

	oldstdout = sys.stdout
	sys.stdout = fw	
	
	for i in range(0, len(req_queue)):
		for j in range(0, len(req_queue[i])):
			req_str = ""
			model_name = ""
			if model[i] == 0: model_name += "alexnet"
			elif model[i] == 1: model_name += "vgg"
			elif model[i] == 2: model_name += "lenet"
			elif model[i] == 3: model_name += "googlenet"
			elif model[i] == 4: model_name += "resnet"
			elif model[i] == 5: model_name += "mobilenet"
			elif model[i] == 6: model_name += "squeezenet"
			elif model[i] == 7: model_name += "yolov2"
			elif model[i] == 8: model_name += "frcnn"
	
			req_str += model_name
			req_str += "_" + str(j) + ":" + str(req_queue[i][j])

			sys.stdout.write(req_str + "\n")

	fw.close()
	sys.stdout = oldstdout


def ExportResults(input_full_path, scenario_num, intensity, model, req_queue, model_tot_req):
	input_file_name = GenerateInputFileName(input_full_path, scenario_num, intensity, model, model_tot_req)
	
	ExportFiles(input_file_name, model, req_queue)
	
	
def GenerateRequestMain(configFile, inputDirectoryPath):
	interval, model, base_lambda, intensity, max_arrtime, scenarios = ReadInputConfigs(configFile)	

	printConfigInfo(interval, model, base_lambda, intensity, max_arrtime)

	for i in range(0, len(interval)):
		for j in range(0, len(base_lambda)):
			for k in range(0, len(intensity)):
				for l in range(0, len(max_arrtime)):
					for m in range(0, len(scenarios)):
						req_queue, model_tot_req = InitSetting(interval[i], model, base_lambda[j], intensity[k], max_arrtime[l], scenarios[m])
						GeneratePoissonMain(req_queue, model_tot_req, max_arrtime[l])
						PrintInfo(interval[i], model, base_lambda[j], intensity[k], max_arrtime[l], scenarios[m], req_queue)
						ExportResults(inputDirectoryPath, m, intensity[k], model, req_queue, model_tot_req)

# Push Input Request to Device &
# Pull Results from Device 
def makeDirectory(localPath, remotePath):
	# For local
	os.system("mkdir " + localPath + "I/")
	os.system("mkdir " + localPath + "AFF/")
	os.system("mkdir " + localPath + "MAEL/")
	os.system("mkdir " + localPath + "SLO-MAEL/")
	os.system("mkdir " + localPath + "PSLO-MAEL/")

	# For remote device
	os.system("adb shell mkdir -p " +  remotePath + "I/") 
	os.system("adb shell mkdir -p " +  remotePath + "AFF/") 
	os.system("adb shell mkdir -p " +  remotePath + "MAEL/") 
	os.system("adb shell mkdir -p " +  remotePath + "SLO-MAEL/") 
	os.system("adb shell mkdir -p " +  remotePath + "PSLO-MAEL/") 

# Push Input Request to Android Device
def pushInputsToDevice(localPath, remotePath):
	onlyfiles = [f for f in os.listdir(localPath) if os.path.isfile(os.path.join(localPath, f))]

	for i in range(0, len(onlyfiles)):
		cmd = "adb push " + localPath + onlyfiles[i] + " " + remotePath + "I/" 
		os.system(cmd)


# Pull Results from Android Device
def pullResultsFromDevice(localPath, remotePath):
	os.system("adb pull " + remotePath + " " + localPath + "AFF/")
	os.system("adb pull " + remotePath + " " + localPath + "MAEL/")
	os.system("adb pull " + remotePath + " " + localPath + "SLO-MAEL/")
	os.system("adb pull " + remotePath + " " + localPath + "PSLO-MAEL/")

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='[Example] Example Usage: python RequestGenerator.py <output_path> <config_file>')
	parser.add_argument('-run', type=str, nargs=2, metavar=('<output_path>', '<config_file>'),
			help='Generate input requests file with config file')
	args = parser.parse_args()

	if args.run: 
		cwd = os.getcwd()
		inputDirectoryName = args.run[0]
		inputDirectoryPath = cwd + "/" + inputDirectoryName
		configFile = args.run[1]

		# already have the directory included in input request files
		# Remove all things 
		if os.path.isdir(inputDirectoryPath) == True:
			os.system("rm -rf " + inputDirectoryPath)		
		os.system("mkdir " + inputDirectoryPath)

		GenerateRequestMain(configFile, inputDirectoryPath)
	else:
		parser.error("[ERROR]: Please run this program with the -h flag to see required arguments")
		
	


