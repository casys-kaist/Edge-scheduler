import sys
import operator
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import math
import random

# Push Input Request to Device &
# Pull Result from Device 

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


if __name__== "__main__":
	print("============================================================================")
	print("[Usage]: python push.py <cmd> <local> <remote>")
	print("<cmd>: \'I\', push requests to device") 
	print("<cmd>: \'O\', pull results from device") 
	print("[Example]: python IODevicectrl.py I Inputfiles/predefined /data/local/tmp/request_file/")
	print("==========================================================================")

	cmd = sys.argv[1]
	localPath = sys.argv[2] 
	remotePath = sys.argv[3] 
	
	if( len(sys.argv) != 4):
		print("[Error]: requires 4 arguments")
		sys.exit(1)

	
	if(cmd == "I"):
		pass
		#pushInputsToDevice(localPath, remotePath):
	
