import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import seaborn as sns

import pickle

def getHumidity():
    # Data Analyst dataset
    dataset = pd.read_json('https://sunflower-tree-system-default-rtdb.firebaseio.com/.json')
    data = dataset['DataAnalysis'].tolist()
    df_dataAnalyst = pd.json_normalize(data)
    df_dataAnalyst['Humidity (%)'].replace('', np.nan, inplace=True)
    df_dataAnalyst['Weather'] = df_dataAnalyst['Weather'].str.lower()
    df_dataAnalyst['Humidity (%)'] = pd.to_numeric(df_dataAnalyst['Humidity (%)'], downcast='float')
    df_dataAnalyst['Light Intensity (%)'] = pd.to_numeric(df_dataAnalyst['Light Intensity (%)'], downcast='float')
    df_dataAnalyst['Soil Moisture (%)'] = pd.to_numeric(df_dataAnalyst['Soil Moisture (%)'], downcast='float')
    df_dataAnalyst['Temperature (°C)'] = pd.to_numeric(df_dataAnalyst['Temperature (°C)'], downcast='float')
    df_dataAnalyst['Temperature (°F)'] = pd.to_numeric(df_dataAnalyst['Temperature (°F)'], downcast='float')
    df_dataAnalyst = df_dataAnalyst.drop(["Battery Percentage (%)", "Voltage (V)", "Voltage Sensor"], axis='columns')
    df_dataAnalyst.dropna(subset=['Humidity (%)', 'Light Intensity (%)'], inplace=True)

    df_dataAnalyst = pd.get_dummies(df_dataAnalyst, columns=['Weather'], drop_first=True)
    df_dataAnalyst = df_dataAnalyst.rename(columns = {'Weather_sunny': 'Weather'})

    # Sensor Dataset
    dataset2 = pd.read_json('https://sunflower-tree-system-default-rtdb.firebaseio.com/.json', orient='none')
    dataset2 = dataset2['Sensor'].tolist()
    data_sensor = pd.DataFrame(dataset2)
    data_sensor = data_sensor.dropna()
    data_sensor = data_sensor.T
    # data_sensor.set_axis(['Humidity (%)', 'Light Intensity (%)', 'Photo', 'Soil Moisture (%)', 'Temperature (°C)', 'Temperature (°F)', 'Time', 'Weather'], axis='columns', inplace=True)
    data_sensor.set_axis(['Battery Percentage (%)', 'Humidity (%)', 'Light Intensity (%)', 'Photo', 'Soil Moisture (%)', 'Temperature (°C)', 'Temperature (°F)', 'Time', 'Voltage (V)', 'Voltage Sensor','Weather'], axis='columns', inplace=True)
    data_sensor = pd.get_dummies(data_sensor, columns=['Weather'])
    column_headers = list(data_sensor.columns)

    if 'Weather_rainy' in column_headers:
        data_sensor = data_sensor.rename(columns = {'Weather_rainy': 'Weather'})
    else:
        data_sensor = data_sensor.rename(columns = {'Weather_sunny': 'Weather'})

    sensorTemp = float(data_sensor['Temperature (°F)'].values[0])
    sensorWeather = float(data_sensor['Weather'].values[0])

    # Linear Regression
    df_dataAnalyst.head()
    df_dataAnalyst.corr()

    ax = sns.regplot(x="Temperature (°F)", y="Humidity (%)", data=df_dataAnalyst)
    ax = sns.regplot(x="Weather", y="Humidity (%)", data=df_dataAnalyst)

    #Define input output data
    # Model 1, y = a + b1x1 + b2x2+ ....
    x = df_dataAnalyst[['Temperature (°F)', 'Weather']].values
    y = df_dataAnalyst['Humidity (%)'].values
    
    #Split the model into train and test data and assign as 'X_train,X_test,y_train and y_test'. The test size should be '0.3' and random state should be '0'
    # Split test and train
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
    # x_train = x_train.reshape(-1, 1)
    # x_test = x_test.reshape(-1, 1)

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x_train, y_train)

    #coef for 'Temperature','Light Intensity', 'Weather'
    model.coef_, model.intercept_
    y_pred = model.predict(x_test)
    pd.DataFrame({'Actual Y': y_test, 
                'Predicted Y': y_pred})

    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(y_test, y_pred)
    r2_value = r2_score(y_test, y_pred)
    # print("MSE:", mse)
    # print("R Squared:", r2_value)
    
    # Predict based on current live weather. If current weather is sunny, it will predict rainy weather humidity.
    if sensorWeather == 1:
        linear_temp = model.predict([[sensorTemp, 0]])
        result = "According to my years of experience. When the weather is Raining, the humidity in the greenhouse is around " + str('%.2f' % linear_temp) + "%"
    else:
        linear_temp = model.predict([[sensorTemp, 1]])
        result = "According to my years of experience. When the weather is Sunny, the humidity in the greenhouse is around " + str('%.2f' % linear_temp) + "%"
    return result