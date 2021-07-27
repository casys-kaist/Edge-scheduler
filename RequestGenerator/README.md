# Description (func)
1. Generate input requests based on Poisson distribution. 
2. Copy input requests to device. 
3. Copy output from device. 

# Requirement
Config file

# Example of Setting Config File 
Interval: 10

Model index: googlenet, resnet, lenet, squeezenet, yolov2, fasterrcnn

Scenario: 1, 2, 1, 1, 1, 1

Scenario: 2, 1, 1, 1, 1, 1

Base Lambda: 10

Intensity: 1.4

Max arrival time: 2000

# Usage
python RequestGenerator.py -cmd '<'arg1'>' '<'arg2'>' 

-Please run this program with the -h flag to see required arguments
