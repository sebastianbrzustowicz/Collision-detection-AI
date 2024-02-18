#!/bin/bash

if [ ! -f "data_factory.py" ]; then
  echo "File 'data_factory.py' does not exist. Aborting the script."
  exit 1
fi

python data_factory.py train

if [ $? -eq 0 ]; then
  echo "Training data created succesufully."
else
  echo "The training data creation failed ."
fi

python data_factory.py test

if [ $? -eq 0 ]; then
  echo "Testing data created succesufully."
else
  echo "The testing data creation failed ."
fi