#metrics

def metric_calculator(model, data, verbose = False):

	x_predicted = mode.predict(data)
	x_pred = x_predicted.reshape((x_predicted.shape[0], x_predicted.shape[1]))

	original = []
	for item in x_train:
  		#print(item[0])
  		original.append(item[0])
		predicted = []
	for item in x_train_pred:
  		#print(item[0])
  		predicted.append(item[0])
	df_original = pd.DataFrame(original, columns= ['temp'])
	original_mean = df_original.mean(axis = 0)
	df_pred = pd.DataFrame(predicted, columns= ['temp'])
	mse_temp = mean_squared_error(df_pred['temp'], df_original['temp'])
	mse = np.array([np.sqrt(mse_temp)])
	recon_error = pd.DataFrame(mse/original_mean.values)
	
	if verbose == True:
		print(recon_error.T.values)
		
	return recon_error.T.values
