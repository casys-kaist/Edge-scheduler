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
	parser = argparse.ArgumentParser(description='[Example]: python IODevicectrl.py <cmd> <local_path> <remote_path>')
	parser.add_argument('-push', type=str, nargs=2, metavar=('<local>', '<remote>'),
			help='copy file/dir to device')
	parser.add_argument('-pull', type=str, nargs=2, metavar=('<local>', '<remote>'),
			help='copy file/dir from device')
	args = parser.parse_args()

	if args.push: 
		localPath = args.push[0]
		remotePath = args.push[1]
		makeDirectory(localPath, remotePath)
		pushInputsToDevice(localPath, remotePath)
	elif args.pull: 
		localPath = args.pull[0]
		remotePath = args.pull[1]
		makeDirectory(localPath, remotePath)
		pullResultsFromDevice(localPath, remotePath)
	else:
		parser.error("ERROR: Please run this program with the -h flag to see required arguments")

		
	
