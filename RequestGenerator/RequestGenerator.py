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

# alexnet, vgg, lenet, googlenet, resnet, mobilenet, squeezenet, yolov2, frcnn
Models_min_runtimes = [20, 91, 5, 16, 36.8, 12.2, 12.6, 47.2, 20.4]

def DefineScenarios():
	all_scenarios = []
	ratio = [1, 2, 1, 1, 1, 1] # ratio
	all_scenarios.append(ratio)

	return all_scenarios

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
			
			# second
			#print it all out
			#print(str(i)+': '+str(_inter_arrival_time)+','+str(_arrival_time))
			req_queue_tmp.append(int(_arrival_time * 1000)) # sec -> ms
			#req_queue.append(_arrival_time)

		last_arrival_time = req_queue_tmp[-1]	
	
	for i in range(0, len(req_queue_tmp)):
		req_queue.append(req_queue_tmp[i])
		
	req_queue.sort()
	print(req_queue)
	
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

	print(selected_model_min_runtimes)
	#for i in range(0, len(ideal_req)):
	#	print(ideal_req[i])

	for i in range(0, len(scenario)):
		model_total_req.append( int(intensity * (ideal_req[i] * float(scenario[i])/sum(scenario) )))

	print(model_total_req)
	print(scenario)
		
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

	File = open(configFile, "r")
	while True:
		line = File.readline()
		if not line: break
		
		if line.find("Interval") != -1:
			interval = re.findall("\d+", line)
			intreval = list(np.float_(interval)) 
		elif line.find("Model index") != -1:
			model_list = line.rstrip("\n").split(":")[1].split(",")
			model = GetModelIndex(model_list)
		elif line.find("Base Lambda") != -1:
			base_lambda = re.findall("\d+", line)
			base_lambda = list(np.float_(base_lambda)) 
		elif line.find("Intensity") != -1:
			intensity = re.findall("\d+\.\d+", line)
			intensity = list(np.float_(intensity)) 
		elif line.find("Max arrival time") != -1:
			max_arrtime = re.findall("\d+", line)
			max_arrtime = list(np.float_(max_arrtime)) 
		
	return interval, model, base_lambda, intensity, max_arrtime 


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
	print(input_file_name)

	return input_file_name

def ExportResults(input_full_path, scenario_num, intensity, model, req_queue, model_tot_req):
	GenerateInputFileName(input_full_path, scenario_num, intensity, model, model_tot_req)
	
def GenerateRequestMain(configFile, inputDirectoryPath):
	interval, model, base_lambda, intensity, max_arrtime = ReadInputConfigs(configFile)	
	scenario = DefineScenarios() 

	printConfigInfo(interval, model, base_lambda, intensity, max_arrtime)

	for i in range(0, len(interval)):
		for j in range(0, len(base_lambda)):
			for k in range(0, len(intensity)):
				for l in range(0, len(max_arrtime)):
					for m in range(0, len(scenario)):
						req_queue, model_tot_req = InitSetting(interval[i], model, base_lambda[j], intensity[k], max_arrtime[l], scenario[m])
						GeneratePoissonMain(req_queue, model_tot_req, max_arrtime[l])
						ExportResults(inputDirectoryPath, m, intensity[k], model, req_queue, model_tot_req)


if __name__=="__main__":
	print("Usage: python RequestGenerator.py InputDirectoryName ConfigFile")

	if len(sys.argv) != 3:
		print("[ERROR] Example Usage: python <RequestGenerator.py> <poisson_grms> <config>")
		exit(1)

	cwd = os.getcwd()
	inputDirectoryName = sys.argv[1]
	inputDirectoryPath = cwd + "/" + inputDirectoryName
	configFile = sys.argv[2]

	# already have the directory included in input request files
	# Remove all things 
	if os.path.isdir(inputDirectoryPath) == True:
		os.system("rm -rf " + inputDirectoryPath)		
	os.system("mkdir " + inputDirectoryPath)

	GenerateRequestMain(configFile, inputDirectoryPath)

