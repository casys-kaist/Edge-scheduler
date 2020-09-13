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
	
def GetTotalRequest(selected_model_min_runtimes, max_arrtime, scenario, intensity):
	model_total_requests = []
	ideal_req = []

	for i in range(0, len(selected_model_min_runtimes)):
		ideal_req.append(max_arrtime / selected_model_min_runtimes[i])

	print(selected_model_min_runtimes)
	#for i in range(0, len(ideal_req)):
	#	print(ideal_req[i])

	for i in range(0, len(scenario)):
		model_total_requests.append( int(intensity * (ideal_req[i] * float(scenario[i])/sum(scenario) )))

	print(model_total_requests)
	print(scenario)
		
	return model_total_requests
	
# For each iteration
def InitSetting(interval, model, base_lambda, intensity, max_arrtime, scenario):
	selected_model_min_runtimes = []
	interval = float(interval)	
	base_lambda = float(base_lambda)	
	intensity = float(intensity)	
	max_arrtime = float(max_arrtime)	
	num_of_model = -1
	model_req = []

	num_of_model = len(model)	
	for i in range(0, len(model)):
		selected_model_min_runtimes.append(Models_min_runtimes[model[i]])

	model_req = GetTotalRequest(selected_model_min_runtimes, max_arrtime, scenario, intensity)

	return num_of_model, selected_model_min_runtimes
	

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

	print(model)

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
		elif line.find("Model index") != -1:
			model_list = line.rstrip("\n").split(":")[1].split(",")
			model = GetModelIndex(model_list)
		elif line.find("Base Lambda") != -1:
			base_lambda = re.findall("\d+", line)
		elif line.find("Intensity") != -1:
			intensity = re.findall("\d+\.\d+", line)
		elif line.find("Max arrival time") != -1:
			max_arrtime = re.findall("\d+", line)
		
	return interval, model, base_lambda, intensity, max_arrtime 
	
def GenerateRequestMain(configFile):
	interval, model, base_lambda, intensity, max_arrtime = ReadInputConfigs(configFile)	
	scenario = DefineScenarios() 

	printConfigInfo(interval, model, base_lambda, intensity, max_arrtime)

	for i in range(0, len(interval)):
		for j in range(0, len(base_lambda)):
			for k in range(0, len(intensity)):
				for l in range(0, len(max_arrtime)):
					for m in range(0, len(scenario)):
						InitSetting(interval[i], model, base_lambda[j], intensity[k], max_arrtime[l], scenario[m])


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

	GenerateRequestMain(configFile)

