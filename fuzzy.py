import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
import pandas as pd
from pandas.io.json import json_normalize
import random

import seaborn as sns

#NLP
import re
import nltk

import pickle

#Loads live data set
def loadSensorDataset():
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
        data_sensor.loc[0, ['Weather']] = [0]
    else:
        data_sensor = data_sensor.rename(columns = {'Weather_sunny': 'Weather'})

    sensorHumidity= float(data_sensor['Humidity (%)'].values[0])
    sensorTemp = float(data_sensor['Temperature (°C)'].values[0])
    sensorTemp2 = float(data_sensor['Temperature (°F)'].values[0])
    sensorSoil = float(data_sensor['Soil Moisture (%)'].values[0])
    sensorLight = float(data_sensor['Light Intensity (%)'].values[0])
    sensorWeather = data_sensor['Weather'].values[0]

    # Fix invalid values gotten from sensor because sensor faulty
    if sensorSoil > 100:
        sensorSoil = 100
    if sensorLight > 100:
        sensorLight = 100
    if sensorSoil < 0:
        sensorSoil = 0
    if sensorLight < 0:
        sensorLight = 0
    
    return sensorHumidity, sensorTemp, sensorTemp2, sensorSoil, sensorLight, sensorWeather


# Fuzzy logic on Sunflower Overall Condition
def sunflowerCondition():
    sensorData = loadSensorDataset()
    sensorHumidity = sensorData[0]
    sensorTemp = sensorData[1]
    sensorTemp2 = sensorData[2]
    sensorSoil = sensorData[3]
    sensorLight = sensorData[4]
    sensorWeather = sensorData[5]

    # Fuzzy Logic
    # Generate universe variables
    # temperature has a range of [60, 130]
    # soil has a range of [0, 60]  
    # lightIntensity has [0, 100]
    # plantCondition has [0, 100]
    temp  = ctrl.Antecedent(np.arange(60, 131, 1),'Temperature (°F)')
    lightIntensity = ctrl.Antecedent(np.arange(0, 101, 1),'Light Intensity (%)')
    soil = ctrl.Antecedent(np.arange(0, 61, 1),'Soil Moisture (%)')
    # weather = ctrl.Antecedent(np.arange(0, 1.1, 0.1),'Weather')
    plantCondition = ctrl.Consequent(np.arange(0, 101, 1),'Condition')

    # Generate fuzzy membership functions
    temp['low'] = fuzz.trimf(temp.universe, [60, 60, 95]) 
    temp['medium'] = fuzz.trimf(temp.universe, [60, 95, 130])
    temp['high'] = fuzz.trimf(temp.universe, [95, 130, 130])

    soil['low'] = fuzz.trimf(soil.universe, [0, 0, 30])
    soil['medium'] = fuzz.trimf(soil.universe, [0, 30, 60])
    soil['high'] = fuzz.trimf(soil.universe, [30, 60, 60])

    lightIntensity['low'] = fuzz.trimf(lightIntensity.universe, [0, 0, 50])
    lightIntensity['medium'] = fuzz.trimf(lightIntensity.universe, [0, 50, 100])
    lightIntensity['high'] = fuzz.trimf(lightIntensity.universe, [50, 100, 100])

    plantCondition['low'] = fuzz.trimf(plantCondition.universe, [0, 0, 50])
    plantCondition['medium'] = fuzz.trimf(plantCondition.universe, [0, 50, 100])
    plantCondition['high'] = fuzz.trimf(plantCondition.universe, [50, 100, 100])

    # weather['low'] = fuzz.trimf(weather.universe, [0, 0, 0.5])
    # weather['medium'] = fuzz.trimf(weather.universe, [0, 0.5, 1])
    # weather['high'] = fuzz.trimf(weather.universe, [0.5, 1, 1])

    rules = list()

    # # Rule Base
    rules.append(ctrl.Rule(temp['low'] | soil['high'] | lightIntensity['low'], plantCondition['low']))
    rules.append(ctrl.Rule(temp['high'] | lightIntensity['medium'] | soil['low'], plantCondition['medium']))
    rules.append(ctrl.Rule(temp['medium'] | soil['medium'] | lightIntensity['high'], plantCondition['high']))
    # rules[0].view()

    plantCondition_ctrl = ctrl.ControlSystem(rules)
    conditioning = ctrl.ControlSystemSimulation(plantCondition_ctrl)

    conditioning.input['Temperature (°F)'] = sensorTemp2
    conditioning.input['Soil Moisture (%)'] = sensorSoil
    conditioning.input['Light Intensity (%)'] = sensorLight

    # Crunch the numbers
    conditioning.compute()
    calculated = conditioning.output['Condition']

    checkConditions = []
    advice = ""

    # Put conditions that is not achieved into an array which will be used to check and output advices.
    if sensorTemp2 <= 70 or sensorTemp2 >= 95:
        checkConditions.append('temperature')
    if sensorSoil <= 20 or sensorSoil >= 30:
        checkConditions.append('soilmoisture')
    if sensorLight <= 50 or sensorLight >= 100:
        checkConditions.append('light')

    # If checkConditions has conditions not achieved, will trigger this.
    if checkConditions:
        advice = "\n\nWhat you can do to improve the situation: "
    # Give advice based on conditions that are not achieved.
    for condition in checkConditions:
        if condition == 'temperature':
            advice += "\n- Sunflower grows best between 70 - 95 (°F)"
        if condition == 'soilmoisture':
            advice += "\n- The soil moisture is recommended between 20 to 30 (%)"
        if condition == 'light':
            advice += "\n- Light intensity for the sunflower is best between 50 - 100 (%)"

    if advice != "":
        advice += "\n\nGood luck! :grin:"

    # Define weather one-hot coding to readable words
    if sensorWeather == 1:
        sensorWeather = 'Sunny'
    else:
        sensorWeather = 'Rainy'

    greenhouse = ("\n__Here are all the information on the greenhouse:__" 
    + "```\nHumidity (%): " + str('%.2f' % sensorHumidity)
    + "\nTemperature (°C): " + str('%.2f' % sensorTemp)
    + "\nTemperature (°F): " + str('%.2f' % sensorTemp2)
    + "\nSoil Moisture (%): " + str('%.2f' % sensorSoil)
    + "\nLight Intensity (%): " + str('%.2f' % sensorLight) 
    + "\nWeather: " + sensorWeather + "```")

    print(calculated)
    
    # Return info based on fuzzy logic percentage
    if calculated <= 20:
        answer = "\nWell, your sunflower is certainly going to be dead soon if this goes on. You have lots of work to do to fix this!"
        answer = greenhouse + answer + advice
    elif calculated > 20 and calculated <= 40:
        answer = "\nYour sunflower looks... very bad. There's still possibility of saving it, act quick!"
        answer = greenhouse + answer + advice
    elif calculated > 40 and calculated <= 50:
        answer = "\nYour sunflower seems to be in a bad spot. Do something before things gets worse"
        answer = greenhouse + answer + advice
    elif calculated > 50 and calculated <= 55:
        answer = "\nYour sunflower looks decently fine, while looking unhealthy."
        answer = greenhouse + answer + advice
    elif calculated > 55 and calculated <= 80:
        answer = "\nSunflower is in a good condition."
        answer = greenhouse + answer + advice
    else:
        answer = "\nThe sunflower is living a very healthy life!"
        answer = greenhouse + answer
    return answer


# def soilmoistureCondition():

#     return answer