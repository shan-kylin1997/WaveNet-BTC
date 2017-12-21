# WaveNet-BTC
This repository contains an implementation of the WaveNet model used for time series forecasting, particularly on intra-day Bitcoin exchange data.

The general architecture used for this implementation was taken from: 

(1) [Conditional Time Series Forecasting with Convolutional Neural Networks](https://arxiv.org/abs/1703.04691)

The original WaveNet paper:

(2) [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)

Also some code was used from the Fast WaveNet model:

(3) [Fast Wavenet Generation Algorithm](https://arxiv.org/abs/1611.09482)

The model was run on only a single day's at a time at 1-minute resolution. The model was fit on data up to 6pm, then generated a dynamic forecast forward for the rest of the day. Sample foecast on the most recent day in the data set:

![Coinbase Price on 2017-10-19 by minute](https://github.com/kykosic/WaveNet-BTC/blob/master/images/sample_output.png "Sample forecast of most recent date in data set")

## Setup and Requirements
Python 2.7 (The only portion of the code incomparable with Python 3 is the use of dict.iteritems() instead of dict.items())
Packages:
* Numpy
* Pandas
* TensorFlow
* Matplotlib
* Mpld3 (optional, used to make the plots interactive)

Download the Bitcoin Historical Data and unzip it into the ./data folder. The data can be found here:
[Bitcoin Historical Data - Kaggle](https://www.kaggle.com/mczielinski/bitcoin-historical-data)

## Model
The network is very close to what is described in (1). It uses a causal, conditional input layer with parameterized skip connections, followed by a stack of dilated convolutions each with a residual connection similar to ResNet, and finally the output is generated by a 1x1 convolution. For this example, we use 7 convolution layers (resulting in a receptive field of 128 time steps or ~2 hours). The rest of the parameters can be found in the "Run Model.ipynb" notebook.

![TensorBoard Render](https://github.com/kykosic/WaveNet-BTC/blob/master/images/single_wavenet.png "Single WaveNet")

The key to this implementation that there are actually 3 parallel WaveNet networks being used, one for the exchange rate from each of 3 different exchanges: Coinbase, Coincheck, and Bitstamp. Each of these networks is trained on predicting the price for a given exchange, based on the past prices of that exchange and conditioned on the past prices of the other two exchanges. These three networks can be trained asynchronously since all historical data is known. 

![TensorBoard Render](https://github.com/kykosic/WaveNet-BTC/blob/master/images/entire_graph.png "Graph Overview")

At generation time, the networks are coupled together such that each model will forecast a single time step t+1, then these three predictions will be fed back into each of the 3 models to make dynamic forecasts. By training a separate network for each feature, we can make arbitrary out-of-sample forecasts into the future, and maintain the multivariate time series. 

![TensorBoard Render](https://github.com/kykosic/WaveNet-BTC/blob/master/images/input_layer.png "Input layer of a single network")
