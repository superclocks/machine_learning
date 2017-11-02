'''
An example of additive backfitting using Gaussian processes.

Author: Wesley Tansey
Date: 2/24/2014
'''
from datetime import date
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import scoreatpercentile
from functools import partial
import csv
from sklearn.gaussian_process import GaussianProcess, squared_exponential
import random


def load_air():
    '''
    Loads the air quality data from the CSV file.
    Returns a tuple (x,y) with x = [Solar.R, Wind, Temp, x-index, date].
    '''
    with open('air.csv', 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        x = []
        y = []
        
        prev_date = None
        cur_x = 0
        for line in reader:
            cur_date = date(2014, int(line[-2]), int(line[-1]))
            if prev_date is None:
                prev_date = cur_date
            cur_x += (cur_date - prev_date).days
            prev_date = cur_date
            x.append([float(line[1]), float(line[2]), float(line[3]), cur_x, cur_date])
            y.append(float(line[0]))
        return (np.array(x).T, np.array(y))

if __name__ == '__main__':
    ''' Perform Gaussian process backfitting on the data in air.csv. '''
    # Parameters of the experiment
    KERNEL = 'sq. exp.'
    COLORS = ['red', 'blue', 'green', 'gold']

    # Hyperparameters of the squared exponential function
    BANDWIDTH = 10.
    BANDWIDTHS = [50., 15., 5.]
    #TAU1 = 15.
    TAU1 = '0.5*range'
    TAU2 = 0.

    # Variance of the observation noise
    NOISE = '1 stdev'

    # Stopping criteria is defined as when the change in residuals is less than some delta
    CONVERGENCE_THRESHOLD = 0.00001

    # Load the air quality data from the 'air.csv' file
    X,observations = load_air()

    # Separate the values we're regressing on from the dates
    indexes = np.array(X[3], dtype=np.int32)
    dates = X[4]
    X = np.array(X[0:3], dtype=np.float64)

    # Make the data be zero-mean
    y_mean = observations.mean()
    y = observations - y_mean

    # Create a partial function so we can try multiple data points easily
    sqexp = partial(squared_exponential, BANDWIDTH, y.std(), TAU2)

    # Create our Gaussian process with zero mean and squared exponential covariance
    gp = GaussianProcess(lambda x: np.zeros(x.shape[0]), sqexp)

    # Calculate the initial features considering only features with lower indices
    error = np.array(y)
    features = []
    for i,x in enumerate(X):
        f_i = gp.predict(x, x, error, error.std(), percentile=None)[0]
        features.append(f_i)
        error -= f_i
    features = np.array(features)

    # Track the squared error of the estimates
    mse = (error * error).mean()
    mse_delta = mse

    print 'Initial mean squared error: {0}'.format(mse)
    
    # Iterate until convergence is reached
    iteration = 1
    while mse_delta >= CONVERGENCE_THRESHOLD:
        # Calculate the features using the previous step's values where necessary
        for i,x in enumerate(X):
            # Get the partial residual
            error = y - np.sum(features[0:i], axis=0) - np.sum(features[i+1:], axis=0)

            # Update our kernel parameters for the partial derivative
            sqexp = partial(squared_exponential, BANDWIDTHS[i], 0.5 * (error.max() - error.min()), TAU2)
            gp.cov = sqexp

            # Get the new feature
            f_i = gp.predict(x, x, error, error.std(), percentile=None)[0]
            
            # Update the feature list
            features[i] = f_i

        # Calculate the total error
        error = y - np.sum(features, axis=0)

        # Track the squared error of the estimates
        prev_mse = mse
        mse = (error * error).mean()
        mse_delta = np.abs(prev_mse - mse)

        print '#{0}: {1}'.format(iteration, np.sqrt(mse))
        iteration += 1

    # Get the backfit predictions
    predicted = y_mean + np.sum(features, axis=0)

    # Plot the observed data points
    figure = plt.figure()
    ax = plt.axes([.1,.1,.8,.7])
    plt.scatter(dates, observations, label='Observed data', color=COLORS[0])
    plt.scatter(dates, predicted, label='Backfitted', color=COLORS[1])
    
    # Pretty up the plot
    plt.xlim(dates[0],dates[-1])
    plt.ylabel('Avg. Ozone Concentration')
    plt.figtext(.40,.9, 'Gaussian Process Backfitting on Ozone Data', fontsize=18, ha='center')
    plt.figtext(.40,.85, '{0} points, {1} kernel, bandwidth={2} , tau1={3}, tau2={4}, noise={5}'.format(len(x), KERNEL, BANDWIDTHS, TAU1, TAU2, NOISE), fontsize=10, ha='center')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    figure.autofmt_xdate()
    plt.savefig('ozone_backfitting.pdf', bbox_inches='tight')
    plt.clf()

    # Plot the observed data points vs. the predicted (should be a straight line)
    figure = plt.figure()
    ax = plt.axes([.1,.1,.8,.7])
    plt.scatter(observations, predicted, label='Observed vs. Predicted', color=COLORS[0])
    
    # Pretty up the plot
    plt.xlabel('Observed Data')
    plt.ylabel('Predicted Data')
    plt.figtext(.40,.9, 'Observed vs. Predicted Gaussian Process Backfitting on Ozone Data', fontsize=18, ha='center')
    plt.figtext(.40,.85, '{0} points, {1} kernel, bandwidth={2} , tau1={3}, tau2={4}, noise={5}'.format(len(x), KERNEL, BANDWIDTHS, TAU1, TAU2, NOISE), fontsize=10, ha='center')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    figure.autofmt_xdate()
    plt.savefig('ozone_observed_vs_predicted.pdf', bbox_inches='tight')
    plt.clf()

    # Plot the individual features to see how they influence the prediction
    f, axarr = plt.subplots(2, 2)
    plt.subplots_adjust(top=0.8)
    axarr[0,0].scatter(dates, observations)
    axarr[0,0].set_title('Observations')
    axarr[0,1].scatter(X[0], features[0])
    axarr[0,1].set_title('Solar Radiation')
    axarr[1,0].scatter(X[1], features[1])
    axarr[1,0].set_title('Wind Speed')
    axarr[1,1].scatter(X[2], features[2])
    axarr[1,1].set_title('Temperature')
    plt.setp([a.get_xticklabels() for a in f.axes], visible=False)
    plt.figtext(.50,.9, 'Backfitting Feature Values for Ozone Data', fontsize=18, ha='center')
    plt.figtext(.50,.87, '{0} points, {1} kernel, bandwidth={2} , tau1={3}, tau2={4}, noise={5}'.format(len(x), KERNEL, BANDWIDTHS, TAU1, TAU2, NOISE), fontsize=10, ha='center')
    plt.savefig('ozone_features.pdf', bbox_inches='tight')
    plt.clf()