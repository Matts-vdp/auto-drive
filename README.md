# auto-drive
A machine learning AI trained to play my car game.

The main purpose of this project is to try and predict what keys to press based on a image.

The AI is trained on screenshots of a human playing the game.
Because of this it will try to mimick human like behavior and won't follow the optimal path.
The version you can see in the preview is trained on 30000 images.

The current network is trained on a custom version of my game. The main differences are:
* The speed is limited. This enables you to hold forward without going to fast.
* A pink rectangle is drawn at the front of the car to help the network with getting the cars current direction

The AI is only trained to use the left and right keys so you have to hold forward constantly while using the AI.

## Performance
While the network isnt perfect it is able to drive around the track without hitting the walls most of the time as seen in the preview.

### Preview
https://user-images.githubusercontent.com/70054706/128602700-11c9b576-ba36-4585-a832-ff10e760b68b.mp4

## Parts

### Gathering training data
To create the training data createData.py takes screenshots of the game while you play.
These images are then resized and saved in the images folder under the directory that corresponds with the pressed key at the moment of the screenshot.


### Training the network
After the data is gathered train.py is launched to train the network.
When it's done it will show a graph that represents the loss and accuracy during the different epochs.
The resulting network will be stored in the savedmodel folder.


### Using the network
predict.py reads the saved model and uses it to predict what key to press.





