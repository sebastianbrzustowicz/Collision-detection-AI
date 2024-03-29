#!/bin/bash

if [ ! -f "data_factory.py" ]; then
  echo "File 'data_factory.py' does not exist. Aborting the script."
  exit 1
fi

echo "Creating dataset for model training and evaluation..."

python data_factory.py train

if [ $? -eq 0 ]; then
  echo "Training data created succesufully."
else
  echo "The training data creation failed."
  exit 1
fi

python data_factory.py test

if [ $? -eq 0 ]; then
  echo "Testing data created succesufully."
else
  echo "The testing data creation failed."
  exit 1
fi

read -p "Do you want to continue and do preprocessing? (y/n): " choice

if [[ $choice =~ ^[Yy]$ ]]; then
    echo "Doing preprocessing..."
else
    echo "Script execution aborted."
    exit 0
fi

if [ ! -f "preprocessing.py" ]; then
  echo "File 'preprocessing.py' does not exist. Aborting the script."
  exit 1
fi

python preprocessing.py X_train.csv
if [ $? -eq 0 ]; then
  echo "Preprocessing succesful for X_train.csv."
else
  echo "Preprocessing failed for X_train.csv."
  exit 1
fi

python preprocessing.py X_test.csv
if [ $? -eq 0 ]; then
  echo "Preprocessing succesful for X_test.csv."
else
  echo "Preprocessing failed for X_test.csv."
  exit 1
fi

read -p "Do you want to continue and train model? (y/n): " choice

if [[ $choice =~ ^[Yy]$ ]]; then
    echo "Training model..."
else
    echo "Script execution aborted."
    exit 0
fi

if [ ! -f "main.py" ]; then
  echo "File 'main.py' does not exist. Aborting the script."
  exit 1
fi
TF_ENABLE_ONEDNN_OPTS=0 python main.py

if [ $? -eq 0 ]; then
  echo "Model trained succesfully (tfmodel folder)."
else
  echo "Model training failed."
  exit 1
fi

read -p "Do you want to look at model evaluation more precisely? (y/n): " choice

if [[ $choice =~ ^[Yy]$ ]]; then
    echo "Checking model evaluation more precisely..."
    if [ ! -f "evaluate_model.py" ]; then
      echo "File 'evaluate_model.py' does not exist. Aborting the script."
      exit 1
    fi
    python evaluate_model.py
fi

read -p "Do you want to keep data files after training? (y/n): " choice

if [[ $choice =~ ^[Yy]$ ]]; then
    echo "Data files remain in the project folder."
else
    files_to_remove=("X_train.csv" "X_test.csv" "y_train.csv" "y_test.csv" "X_train_preprocessed.csv" "X_test_preprocessed.csv" "y_hat.csv")
    for file in "${files_to_remove[@]}"; do
    # Check if file exist
    if [ -e "$file" ]; then
        rm "$file"
        echo "File $file deleted."
    else
        echo "File $file does not exist."
    fi
    done
fi

read -p "Do you want to start TensorFlow model hosting service in Docker? (y/n): " choice

if [[ $choice =~ ^[Yy]$ ]]; then
    echo "Starting service in Docker..."
    if [ ! -f "startservice.sh" ]; then
      echo "File 'startservice.py' does not exist. Aborting the script."
      exit 1
    fi
    bash ./startservice.sh

    if [ ! -f "X_test_preprocessed.csv" ] || [ ! -f "y_test.csv" ]; then
      echo "Full example of sending a request to a model in a request.py file."
      echo "Script completed."
      read -n 1 -s -r -p "Press any key to exit..."
      exit 0
    else
      read -p "Do you want to send example request? (y/n): " choice
      if [[ $choice =~ ^[Yy]$ ]]; then
      echo "Sending request to hosted model..."
        if [ ! -f "request.py" ]; then
          echo "File 'request.py' does not exist. Aborting the script."
          exit 1
        fi
        python request.py
        echo "Full example of sending a request to a model in a request.py file."
      fi
    fi
fi

echo "Script completed."
read -n 1 -s -r -p "Press any key to exit..."
exit 0