import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt import App
from dotenv import find_dotenv, load_dotenv
from functions import draft_email
import requests
import json
import time

from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request
from flask import jsonify

import torch
from diffusers import StableDiffusionPipeline

import base64
from io import BytesIO

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set Slack API credentials
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
SLACK_BOT_USER_ID = os.environ["SLACK_BOT_USER_ID"]

# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN)

# Initialize the Flask app
# Flask is a web application framework written in Python
flask_app = Flask(__name__)
run_with_ngrok(flask_app)
handler = SlackRequestHandler(app)

# # Define a function to keep Colab active
# def keep_colab_active():
#     while True:
#         print("Colab is still active")
#         time.sleep(300)  # Sleep for 5 minutes

# Start the function in a separate thread to keep running in the background
# import threading
# threading.Thread(target=keep_colab_active).start()

def get_bot_user_id():
    """
    Get the bot user ID using the Slack API.
    Returns:
        str: The bot user ID.
    """
    try:
        # Initialize the Slack client with your bot token
        slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
        response = slack_client.auth_test()
        return response["user_id"]
    except SlackApiError as e:
        print(f"Error: {e}")


def my_function(text):
    """Custom function to process the text and return a response."""
    response = text  # No modification to the text
    return response

@flask_app.route("/get_response", methods=["GET"])
def get_response():
    """Endpoint for fetching response based on the draft_email function."""
    text = request.args.get("text")

    # Extract the email from the text
    email, response = draft_email(text)

    return jsonify({"email": email, "response": response})

@app.event("app_mention")
def handle_mentions(body, say):
    """Event listener for mentions in Slack."""
    text = body["event"]["text"]

    mention = f"<@{SLACK_BOT_USER_ID}>"
    text = text.replace(mention, "").strip()

    # Process the text using your custom function
    email, response = draft_email(text)

    say(f"Sure, I'll get right on that! @{email}: {response}")


# @app.event("app_mention")
# def handle_mentions(body, say):
#     """
#     Event listener for mentions in Slack.
#     When the bot is mentioned, this function processes the text and sends a response.

#     Args:
#         body (dict): The event data received from Slack.
#         say (callable): A function for sending a response to the channel.
#     """
#     text = body["event"]["text"]

#     mention = f"<@{SLACK_BOT_USER_ID}>"
#     text = text.replace(mention, "").strip()

#     say("Sure, I'll get right on that!")
#     # response = my_function(text)
    
#     # Extract the email from the text
#     email, response = draft_email(text)
    
#     # Make the POST request
#     url = "https://hook.us1.make.com/ohyonocw701n4ynie637qcm3roe3yrhn"
#     headers = {"Content-Type": "application/json"}
#     payload = {"email": email, "response": response}
#     data = json.dumps(payload)
#     # data = {"response": response}
    
#     # post_response = requests.post(url, headers=headers, json=data)
    
#     post_response = requests.post(url, headers=headers, data=data)

#     # Check the response status code
#     if post_response.status_code == 200:
#         say("POST request successful")
#     else:
#         say("POST request failed")


# @app.route("/slack/events", methods=["POST"])
# def slack_events():
#     """
#     Route for handling Slack events.
#     This function passes the incoming HTTP request to the SlackRequestHandler for processing.

#     Returns:
#         Response: The result of handling the request.
#     """
#     return handler.handle(request)


@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    """Route for handling Slack events."""
    return handler.handle(request)


# Run the Flask app
if __name__ == "__main__":
    flask_app.run()