# Collision detection AI

Train your machine learning model for collision detection with an accelerometer sensor data and TensorFlow.    
The current model is kinda overfitted due to lack of diverse datasets.    
So it would be better for you to provide your own collected data from sensor.   
The data should reflect the real conditions where your sensor is used.    
For example mobile robots, manipulator robots or flying vehicles datasets may vary.    
The point of this training script is to generate relevant model.    

## Commands

Prepare your environment:

```console
pip install -r requirements.txt
```

All you have to do is execute this bash script:

```console
./makemodel.sh
```
This script guides you step by step through the training process.    

<p align="center">
    <img src="https://github.com/sebastianbrzustowicz/Collision-detection-AI/assets/66909222/790ef8e3-bd0d-4513-a2c9-0ab57fac82af" alt="Opis obrazka" style="width:400px;height:200px;">
</p>
<p align="center">
    <img src="https://github.com/sebastianbrzustowicz/Collision-detection-AI/assets/66909222/275f0571-cd34-454c-80f5-8423cbf2c8f4" alt="Opis obrazka" style="width:400px;height:300px;">
</p>

## Technical information
  
The data is designed to mimic the possible orientation of a multi-rotor vehicle during flight, changes in flight direction and collision with an obstacle.
Example of test data generated looks as below.    
<p align="center">
    <img src="https://github.com/sebastianbrzustowicz/Collision-detection-AI/assets/66909222/beeaf908-1eca-4d9a-8025-bc3f04c955b7" alt="Opis obrazka" style="width:500px;height:400px;">
</p>

The data provided is preprocessed (normalised), so it is not necessary to provide it in a specific unit format.    
For current neural network settings you need to provide 500 samples for each dataset.   
It is determined for one second model execution time (in future usage) and 2 ms sampled dataset.    
Data should be provided in csv file (with "," delimiter). One row equals one dataset. One column equals one sample.    
For output data (y_train, y_test) just int value every line, where collision equals 1, otherwise it is 0.    

In evaluation process, script shows you 5 worst learning cases for this dataset with their errors and indexes.    

## License

Collision detection AI is released under the CC BY-NC-ND 4.0 license.

## Author

Sebastian Brzustowicz &lt;Se.Brzustowicz@gmail.com&gt;
