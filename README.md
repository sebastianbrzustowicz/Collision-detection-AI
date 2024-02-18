# Collision detection AI

Repository for training a machine learning model for collision detection via an accelerometer sensor with the help of TensorFlow.    
The current model is kinda overfitted due to lack of diverse datasets.    
So it would be better for you to train model on your own collected data from sensor.   
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

## License

Collision detection AI is released under the CC BY-NC-ND 4.0 license.

## Author

Sebastian Brzustowicz &lt;Se.Brzustowicz@gmail.com&gt;
