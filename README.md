# Vessel Trajectory Prediction
This is a demo of the vessel trajectory prediction and the unofficial and simple implementation of the papers listed below.
+ [Deep Learning Methods for Vessel Trajectory Prediction based on Recurrent Neural Networks](https://arxiv.org/pdf/2101.02486.pdf)
+ [Prediction oof Vessel Trajectories From AIS Data Via Sequence-To-Sequence Recurrent Neural Networks](https://ieeexplore.ieee.org/document/9054421)

### The overview
We use human-made AIS data (only 2-dimension data including longitude and latitude) to train the model. The trajectory is illustrated in the following picture. It has 400 points and the first 300 points are regarded as training data, the next 60 points are regarded as validation data. The remaining 40 points are known as testing data.

![avatar](https://github.com/whubaichuan/Vessel_Trajectory_Prediction/blob/main/data/trajectory.png)

You also can build your custom AIS data with speed, orientation and so on here to predict your vessel trajectory.

### The network

As usual, we use the sequence-to-sequence model. The first step is to send the previous **L steps** of observation into LSTM to encoder the feature. Then in the decoder step, the feature through LSTM can predict the next **h steps** of vessel trajectory.
> Here we set L to 20, h to 1. You could change the value in the run.py easily. We use the adam optimizer and set the epochs to 100.

![avatar](https://github.com/whubaichuan/Vessel_Trajectory_Prediction/blob/main/data/model.png)

### The experiment
+ The training process of our model. We see that the loss of train and valid is decreasing.

![avatar](https://github.com/whubaichuan/Vessel_Trajectory_Prediction/blob/main/data/train.png)

+ The vessel trajectory prediction of train, valid and test corresponds to the true trajectory qualitatively.

![avatar](https://github.com/whubaichuan/Vessel_Trajectory_Prediction/blob/main/data/prediction.png)

If we have lots of AIS data to train the sequence-to-sequence model, we will get a better prediction.