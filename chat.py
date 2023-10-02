import json
import random

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

from linear_humidity_weather import getHumidity
from fuzzy import sunflowerCondition, loadSensorDataset
from functions import get_plantAdvice, get_soilCondition, get_temperatureCondition, get_weatherCondition

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device) # Neural Network model
model.load_state_dict(model_state)
model.eval()

bot_name = "Lucy"

# Array for checking if that specific plant exist
list_of_plants = [
    'cabbage', 'eggplant', 'hot pepper', 'cantaloupe', 'cherry tomato', 'pea', 'pitcher plant', 'lowland roses', 'zinnias', 
    'radish', 'bok choy', 'tatsoi', 'ginger', 'garlic', 'cilantro', 'sweet basil', 'mint', 'scallion', 'hydrangea', 
    'orchid', 'morning glory', 'anthurium', 'chrysanthemum', 'succulents', 'croton', 'crown of thorns', 'spider plant', 'aloe vera'
    ]

def respondCondition(user_Input):
        print(user_Input)
        sentence = tokenize(user_Input) # tokenize user input
        X = bag_of_words(sentence, all_words) # putting tokenized words into Bag of Words
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)
        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:               
                    if tag == "predict_humidity":
                        response = random.choice(intent['responses'])
                        predict_humidity = getHumidity()
                        return(f"{response}\n{predict_humidity}")
                    elif tag == "plant_condition":
                        response = random.choice(intent['responses'])
                        fuzzification = sunflowerCondition()
                        return (f"{response}\n{fuzzification}")
                    elif tag == "check_soilmoisture":
                        sensorData = loadSensorDataset()
                        sensorSoil = sensorData[3]
                        response = get_soilCondition(sensorSoil)
                        return (f"{response}\n")
                    elif tag == "check_temperature":
                        sensorData = loadSensorDataset()
                        sensorTemp = sensorData[2]
                        response = get_temperatureCondition(sensorTemp)
                        return (f"{response}\n")
                    elif tag == "check_weather":
                        sensorData = loadSensorDataset()
                        sensorWeather = sensorData[5]
                        u_Input = user_Input
                        response = get_weatherCondition(sensorWeather, u_Input)
                        return (f"{response}\n")
                    elif tag in list_of_plants:
                        advice = get_plantAdvice(tag)
                        response = random.choice(intent['responses'])
                        response += advice
                        return(f"{response}")
                    else:
                        response = random.choice(intent['responses'])
                        return(f"{response}")
        else:
            return(f"Sorry, I don't understand what you're trying to tell me... :(")
