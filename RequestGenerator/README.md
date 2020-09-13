# Description
Generate input requests based on Poisson distribution.

# Requirement
Config file

# Example of Setting Config File 
Interval: 10
Model index: googlenet, resnet, lenet, squeezenet, yolov2, fasterrcnn
Base Lambda: 10
Intensity: 1.4
Max arrival time: 2000

# Usage
python RequestGenerator.py InputDirectoryName sampleConfig
