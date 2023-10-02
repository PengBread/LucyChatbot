import random
import pickle
from chat import respondCondition


def handle_response(message) -> str:
    p_message = message.lower()
    if p_message.startswith("!lucy"):
        output = respondCondition(message)
        # print(output[0])
        return output

    if p_message == '!help':
        respond_help = """You can try to ask me:
        ```1) What can be planted in the Malaysia/Greenhouse.
        \n2) Predict humidity in the greenhouse based on weather
        \n3) What is the condition of the sunflower in the greenhouse?
        \n4) Check weather - "Is is rainy?, Is it sunny?, What is the weather?"
        \n5) Check hydration - "Are you thirsty?, Is your soil moist enough?"
        \n6) Check temperature - "Is it too cold?, Is it too hot?, Is the temperature alright?"
        ```"""
        return respond_help

