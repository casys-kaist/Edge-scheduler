import copy 
import itertools
import sys
import random
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import Counter


def GenerateRequestMain():
	pass

if __name__=="__main__":
	print("Usage: python RequestGenerator.py InputDirectoryName")

	if len(sys.argv) != 2:
		print("[ERROR] Example Usage: python <RequestGenerator.py> <poisson_grms>")
		exit(1)

	cwd = os.getcwd()
	inputDirectoryName = sys.argv[1]
	inputDirectoryPath = cwd + "/" + inputDirectoryName

	# already have the directory included in input request files
	# Remove all things 
	if os.path.isdir(inputDirectoryPath) == True:
		os.system("rm -rf " + inputDirectoryPath)		
	os.system("mkdir " + inputDirectoryPath)

	GenerateRequestMain()
