import sys
import operator
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import math
import random


if __name__== "__main__":
	print("Usage: python push.py <local> <remote>")
	print("Example: python push.py Inputfiles/predefined predefined")

	local = sys.argv[1] 
	remote = sys.argv[2] 
	
	# create directory 
	os.system("adb shell mkdir -p /data/local/tmp/request_file/" + remote + "I/")
	os.system("adb shell mkdir -p /data/local/tmp/request_file/" + remote + "gpu_O/")
	os.system("adb shell mkdir -p /data/local/tmp/request_file/" + remote + "dsp_O/")
	os.system("adb shell mkdir -p /data/local/tmp/request_file/" + remote + "lb_O/")
	os.system("adb shell mkdir -p /data/local/tmp/request_file/" + remote + "st_O/")
	os.system("adb shell mkdir -p /data/local/tmp/request_file/" + remote + "my_O/")
	os.system("adb shell mkdir -p /data/local/tmp/request_file/" + remote + "slo_O/")
	os.system("adb shell mkdir -p /data/local/tmp/request_file/" + remote + "slo_div_O/")

	onlyfiles = [f for f in os.listdir(local) if os.path.isfile(os.path.join(local, f))]
	
	for i in range(0, len(onlyfiles)):
		cmd = "adb push " + sys.argv[1] + onlyfiles[i] + " /data/local/tmp/request_file/" + remote + "I/" 
		os.system(cmd)

