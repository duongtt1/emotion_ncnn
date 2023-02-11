#!/bin/bash

python export.py 

python -m onnxsim checkpoints/mobilenet_v2.onnx checkpoints/mobilenet_v2-sim.onnx