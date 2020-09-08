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


def printConfigInfo(interval, model, base_lambda, intensity, max_arrtime, base_runtime):
	print("Interval: ", interval)
	print("Model index: ", model)
	print("Base Lambda: ", base_lambda) 
	print("Intensity: ", intensity)
	print("Max arrival time: ", max_arrtime)
	print("Base runtime: ", base_runtime) 

def GetModelIndex(model_list):
	model = []
	if model_list.find("alexnet") != -1: model.append(0)
	if model_list.find("vgg") != -1: model.append(1)
	if model_list.find("lenet") != -1: model.append(2)
	if model_list.find("googlenet") != -1: model.append(3)
	if model_list.find("resnet") != -1: model.append(4)
	if model_list.find("mobilenet") != -1: model.append(5)
	if model_list.find("squeezenet") != -1: model.append(6)
	if model_list.find("yolov2") != -1: model.append(7)
	if model_list.find("fasterrcnn") != -1: model.append(8)

	return model

def ReadInputConfigs(configFile):
	interval = [10]
	model = [] 
	base_lambda = [10]
	intensity = []
	max_arrtime = [2000] # 2 sec
	base_runtime = 2000.0

	File = open(configFile, "r")
	while True:
		line = File.readline()
		if not line: break
		print(line)
		
		if line.find("Interval") != -1:
			interval = re.findall("\d+", line)
		elif line.find("Model index") != -1:
			model = GetModelIndex(line)
		elif line.find("Base Lambda") != -1:
			base_lambda = re.findall("\d+", line)
		elif line.find("Intensity") != -1:
			intensity = re.findall("\d+\.\d+", line)
		elif line.find("Max arrival time") != -1:
			max_arrtime = re.findall("\d+", line)
		elif line.find("Base runtime") != -1:
			base_runtime = re.findall("\d+", line)
		

	return interval, model, base_lambda, intensity, max_arrtime, base_runtime
	
def GenerateRequestMain(configFile):
	interval, model, base_lambda, intensity, max_arrtime, base_runtime = ReadInputConfigs(configFile)		
	printConfigInfo(interval, model, base_lambda, intensity, max_arrtime, base_runtime)


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
