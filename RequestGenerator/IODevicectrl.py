import sys
import operator
import os
import re

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

if __name__== "__main__":
	print("============================================================================")
	print("[Usage]: python push.py <cmd> <local> <remote>")
	print("<cmd>: I, push requests to device") 
	print("<cmd>: O, pull results from device") 
	print("[Example]: python IODevicectrl.py I Inputfiles/predefined /data/local/tmp/request_file/")
	print("==========================================================================")

	if( len(sys.argv) != 4):
		print("[Error]: requires 4 arguments")
		sys.exit(1)
	else:
		cmd = sys.argv[1]
		localPath = sys.argv[2] 
		remotePath = sys.argv[3] 
		makeDirectory(localPath, remotePath)
	
	if(cmd == "I"):
		pushInputsToDevice(localPath, remotePath)
	elif(cmd == "O"):
		pullResultsFromDevice(localPath, remotePath)
	else:
		print("[Error]: requires \'I\' or \'O\' cmd")
		sys.exit(1)
	
