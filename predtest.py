import sys
import json
import build_model
import data_helper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from training_config import *

def train_predict():
	"""Train and predict time series data"""

	# Load command line arguments 
	train_file = sys.argv[1]
	parameter_file = sys.argv[2]
	
	print(train_file)
	print("Complete")
	
	print(parameter_file)
	print("Complete")

	# Load training parameters
	params = json.loads(open(parameter_file).read())

	# Load time series dataset, and split it into train and test
	x_train, y_train, x_test, y_test, x_test_raw, y_test_raw,\
		last_window_raw, last_window = data_helper.load_timeseries(train_file, params)
	# print(type(y_test_raw))	# "y_test_raw" is the predict value parameter that need to pass for the trend check file 'E:\EDU\Python_Projects\trend_analysis\trent_test.py'
	# print(len(y_test_raw))
	
	pred = y_test_raw.tolist()
	dist_points = len(pred)
	# print(dist_points)
	# print(len(pred))
	# print(pred)
	# print ('COMPLETE.....!')
	
	# Build RNN (LSTM) model
	lstm_layer = [1, params['window_size'], params['hidden_unit'], 1]
	model = build_model.rnn_lstm(lstm_layer, params)

	# Train RNN (LSTM) model with train set
	model.fit(
		x_train,
		y_train,
		batch_size=params['batch_size'],
		epochs=params['epochs'],
		validation_split=params['validation_split'])

	# Check the model against test set
	predicted = build_model.predict_next_timestamp(model, x_test)        
	predicted_raw = []
	print (predicted_raw)
	print("Next predicted data values......................,")
	for i in range(len(x_test_raw)):
		predicted_raw.append((predicted[i] + 1) * x_test_raw[i][0])

	# Predict next time stamp 
	#next_timestamp = build_model.predict_next_timestamp(model, last_window)
	#next_timestamp_raw = (next_timestamp[0] + 1) * last_window_raw[0][0]
	
	next_timestamp_data_points_plot = []
	for row in last_window_raw:
		for x1 in row:
			next_timestamp = build_model.predict_next_timestamp(model, last_window)
			next_timestamp_raw = (next_timestamp[0] + 1) * x1
			#next_timestamp_data_points_plot.append(x)
			print('The next time stamp forecasting is: {}'.format(next_timestamp_raw))
			next_timestamp_data_points_plot.append(format(next_timestamp_raw))
	#print(next_timestamp_data_points_plot)
	print('Done predection....!')
	
	# for row in next_timestamp:
		#print(row)
		
	# last_window rss values
	# for row in last_window_raw:
		# print(row)
		
	# print("Done")
	
	#print('The next time stamp forecasting is: {}'.format(next_timestamp_raw))
	
	#next_timestamp_raw = (next_timestamp[1] + 1) * last_window_raw[1][0]
	#print('The next time stamp forecasting is: {}'.format(next_timestamp_raw))
	
	print('\n''Enter window size value according to the json file.....')
	w = int(input('window_size = ')) #window_size
	print ('Entered window value is ',w)
	window_size_list = list(range(1, w+1))
	# for x in range(1, w+1):
		# # print(x)
		# window_size_list = [x for i in range(w)]
	print(window_size_list, '\n''Set value for number of predicted points Completed...!')
	
	# trend test analysis for predicted values
	# difference_trend =[]
	# down_trend = []
	# up_trend = []
	# distance_prediction_points = list(range(1, dist_points+1))
	# # print(distance_prediction_points)
	# #for j in range(1, len(pred) - 1): #for 1 BS Data set
	# for j in range(1, len(pred) + 1): #for 2BS Data set
	# 	current_value = pred[j]
	# 	previous_value = pred[j - 1]
	# 	difference_trend = current_value - previous_value
	# 	if difference_trend < 0:
	# 		if j == 1:
	# 			down_trend.append(previous_value)
	# 			down_trend.append(current_value)
	# 		else:
	# 			down_trend.append(current_value)
	# 	else:
	# 		down_trend.append(None)
	# 		up_trend.append(current_value)
	
	
	# down_trend.append(None)

	# print('\n''DONE Trend categorizing...!''\n')
	# print('up trend number of points:', len(up_trend)+1)
	# print('Down trend number of points :', (len(down_trend)-(len(up_trend)+1)))
	# print('Total number of points :', len(down_trend))
	
	# print(type(down_trend),len(down_trend))
	# print(type(pred),len(pred))
	
	# print(pred)
	# print('division')
	# print(down_trend)
	
	
	# Plot graph: predicted VS actual
	plt.figure(1)
	#plt.subplot(211)
	plt.plot(predicted_raw, label='Actual')
	plt.plot(y_test_raw, label='Predicted')
	plt.title('Predicted and Actual RSS')
	plt.legend()
	# Plot graph: down trend filtered from predicted figure
	#plt.figure(2)
	#plt.plot(distance_prediction_points, down_trend,'C1', label='Predicted Down Trend')
	#plt.xlabel('Distanace(m)')
	#plt.ylabel('Predicted RSS(dBm)')
	#plt.title('Predicted RSS Trend')
	#plt.legend()
	# Plot graph: up trend filtered from predicted figure
	plt.figure(2)
	plt.scatter(window_size_list,next_timestamp_data_points_plot, label='predicted points according to window', color='k', s=25, marker="o")# linestyle='--'
	plt.xlabel('Distanace(m)')
	plt.ylabel('Predicted RSS(dBm)')
	plt.title('Predicted RSS Values')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	# python predtest.py ./data/rss_test.csv ./training_config.json
	# python predtest.py ./data/rss_test_2BS.csv ./training_config.json
	# python predtest.py ./data/rss_real.csv ./training_config.json
	# cd Testpy\time series git examples\forecasting-rnn-tensorflow-10-13
	train_predict()
