import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import seaborn as sns

# Function to get available plant advice
def get_plantAdvice(user_Input):
    dataset = pd.read_csv('Plant_Data.csv') #Loads webscrapped data
    description = ""

    for idx in dataset.index:
        plantName = dataset['Common Name'][idx].lower()
        
        #If user input's requested plant exist in the webscrapped data
        if user_Input in plantName:
            description = (f"\n\nHere are the advice in planting {plantName}")
            description += (f"```- Keep the temperature between {dataset['Temperature'][idx]}")
            description += (f"\n- The soil moisture should be {dataset['Soil Type'][idx]}")
            description += (f"\n- Keep the soil ph between {dataset['Soil pH2'][idx]}")
            description += (f"\n- Humidity of the area - {dataset['Humidity'][idx]} (%)```")
            description += (f"\n\n**Other Information: **")
            description += (f"\n{dataset['Advice'][idx]}")
            description += "\n\nThat's all!"
            break
    return description


# Soil Condition checker
def get_soilCondition(sensorSoil):
    if sensorSoil <= 10:
        answer = (f"I am very dry... and thirsty.... Help me :(")
    elif sensorSoil >= 11 and sensorSoil <= 20:
        answer = (f"I feel hydrated but I might be thirsty very soon!")
    elif sensorSoil >= 21 and sensorSoil <= 30:
        answer = (f"My hydration levels are good!")
    elif sensorSoil >= 31 and sensorSoil <= 40:
        answer = (f"I think that's enough of water for now...")
    else:
        answer = (f"I think I'm over hydrated... feel like vomitting...")
    return answer


# Temperature Condition checker
def get_temperatureCondition(sensorTemperature):
    if sensorTemperature <= 60:
        return (f"I feel... cold... Help :(")
    elif sensorTemperature >= 61 and sensorTemperature <= 90:
        return (f"Good Temperature!")
    elif sensorTemperature >= 91 and sensorTemperature <= 130:
        return (f"Just the right temperature!")
    else:
        return (f"Umm.. help?")


# Weather Checker
def get_weatherCondition(sensorWeather, user_Input):
    # If sensor is sunny, and user input is sunny. It will reply yes. If sensor is rainy, user Input is sunny. It will reply no.
    # Reply based on user input
    if 'sunny' in user_Input.lower() or 'rainy' in user_Input.lower() or 'raining' in user_Input.lower():
        if sensorWeather == 1 and 'sunny' in user_Input.lower():
            return (f"Yes, it is Sunny right now!")
        elif sensorWeather == 0 and 'sunny' in user_Input.lower():
            return (f"No, the weather is Rainy right now!")
        elif sensorWeather == 0 and 'rainy' in user_Input.lower():
            return (f"Yes, it is raining right now!")
        elif sensorWeather == 0 and 'raining' in user_Input.lower():
            return (f"Yes, it is raining right now!")
        elif sensorWeather == 1 and 'rainy' in user_Input.lower():
            return (f"No, the weather is Sunny right now!")
        elif sensorWeather == 1 and 'raining' in user_Input.lower():
            return (f"No, the weather is Sunny right now!")
        else:
            return (f"I don't know")
    else:
        if sensorWeather == 1:
            return (f"The weather is Sunny")
        else:
            return (f"The weather is Rainy")