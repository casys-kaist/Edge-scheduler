import sys
import operator
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import math
import random

# Push Input Request to Device &
# Pull Results from Device 


# Push Input Request to Android Device
def pushInputsToDevice(localPath, remotePath):
	os.system("adb shell mkdir -p " +  remotePath + "I/") # for input
	os.system("adb shell mkdir -p " +  remotePath + "AFF/") # for AFF
	os.system("adb shell mkdir -p " +  remotePath + "MAEL/") # for MAEL
	os.system("adb shell mkdir -p " +  remotePath + "SLO-MAEL/") # for SLO-MAEL
	os.system("adb shell mkdir -p " +  remotePath + "PSLO-MAEL/") # for PSLO-MAEL

	onlyfiles = [f for f in os.listdir(localPath) if os.path.isfile(os.path.join(localPath, f))]

	for i in range(0, len(onlyfiles)):
		cmd = "adb push " + localPath + onlyfiles[i] + " " + remotePath + "I/" 
		os.system(cmd)


# Pull Results from Device 

def pull_subset(Req_name, Result_path, algo):
	if os.path.isdir(Result_path) == False:
		os.system("mkdir " + Result_path)

	algo_path = Result_path + "/" + Req_name + algo + "_O"

	os.system("mkdir " + algo_path)
	os.system("adb pull /data/local/tmp/request_file/"+Req_name+algo+"_O/ "+algo_path)

def pull(Req_name, Result_path):
	
	if os.path.isdir(Result_path) == False:
		os.system("mkdir " + Result_path)
		
	gpu = Result_path + "/" + Req_name + "gpu_O"
	dsp = Result_path + "/" + Req_name + "dsp_O"
	lb = Result_path + "/" + Req_name + "lb_O"
	st = Result_path + "/" + Req_name + "st_O"
	my = Result_path + "/" + Req_name + "my_O"
	slo = Result_path + "/" + Req_name + "slo_O"
	slo_div = Result_path + "/" + Req_name + "slo_div_O"
		
	os.system("mkdir " + gpu)
	os.system("mkdir " + dsp)
	os.system("mkdir " + lb)
	os.system("mkdir " + st)
	os.system("mkdir " + my)
	os.system("mkdir " + slo)
	os.system("mkdir " + slo_div)
	
	
	os.system("adb pull /data/local/tmp/request_file/"+Req_name+"gpu_O/ " +gpu)
	os.system("adb pull /data/local/tmp/request_file/"+Req_name+"dsp_O/ " +dsp)
	os.system("adb pull /data/local/tmp/request_file/"+Req_name+"lb_O/ " +lb)
	os.system("adb pull /data/local/tmp/request_file/"+Req_name+"st_O/ " +st)
	os.system("adb pull /data/local/tmp/request_file/"+Req_name+"my_O/ " +my)
	os.system("adb pull /data/local/tmp/request_file/"+Req_name+"slo_O/ " +slo)
	os.system("adb pull /data/local/tmp/request_file/"+Req_name+"slo_div_O/ " +slo_div)




if __name__== "__main__":
	print("============================================================================")
	print("[Usage]: python push.py <cmd> <local> <remote>")
	print("<cmd>: I, push requests to device") 
	print("<cmd>: O, pull results from device") 
	print("[Example]: python IODevicectrl.py I Inputfiles/predefined /data/local/tmp/request_file/")
	print("==========================================================================")

	cmd = sys.argv[1]
	localPath = sys.argv[2] 
	remotePath = sys.argv[3] 
	
	if( len(sys.argv) != 4):
		print("[Error]: requires 4 arguments")
		sys.exit(1)

	
	if(cmd == "I"):
		pushInputsToDevice(localPath, remotePath)
	elif(cmd == "O"):
		pass
	else:
		print("[Error]: requires \'I\' or \'O\' cmd")
		sys.exit(1)
	
