To Run the prediction model first we need to create our data set from the fashion mnist by running the data creator.py file
step_1 run the data_creator.py
       this is to create a dataset csv file with the name fashion_mnist.csv
now we have to train our model and save the model in pkl files to do that we need to run the train model.py
step_2 run the train_model.py
       this will train our ml model by using the data set(fashion_mnist.csv) and save the model in pkl files namely fashion_model.pkl,scaler.pkl,feature_names.pkl
now we are ready to run our prediction model and find out the top 3 treending fashion trends and also test the accuracy and classification report of the model to do this we need to run our trend_predictor.py
step_3 run the trend_predictor.py
       this will print the top 3 trends according to the ml model prodiction and also give us the accuaracy and the f1 scores of the model 
       
