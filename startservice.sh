#!/bin/bash

docker pull tensorflow/serving

docker stop collision-detection-ai
docker rm collision-detection-ai

docker run -p 8501:8501 -d --name=collision-detection-ai --mount type=bind,source="your_path_to_model\tfmodel",target=/models/tfmodel/1 -e MODEL_NAME=tfmodel -t tensorflow/serving
