# server.py
import io
from flask import Flask, jsonify, request
from google.cloud import texttospeech
from flask_cors import CORS
import sys
import os
import base64
from configparser import ConfigParser
import app as chatApp

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "PATH_TO_GOOGLE_APPLICATION_CREDENTIALS"

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

config = ConfigParser()
config.read('credentials.ini')
api_key = config['gemini_ai']['API_KEY']

chatbot = chatApp.ChatBot(api_key=api_key)

# Dummy messages
messages = [{'text': 'Hi, this is XXX Customer support. How may I help you today?', 'sender': 'Bot'}]

def text_to_speech(text, target_language='en-US'):
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=target_language,
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    return response.audio_content

@app.route('/text-to-speech', methods=['POST'])
def convert_text_to_speech():
    data = request.json
    text = data['text']
    audio_content = text_to_speech(text)
    audio_base64 = base64.b64encode(audio_content).decode('utf-8')
    return jsonify({'audio_base64': audio_base64})

@app.route('/api/data', methods=['GET', 'POST'])
def handle_messages():
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'CORS preflight request successful'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
        return response
    elif request.method == 'GET':
        return jsonify({'messages': messages})
    elif request.method == 'POST':
        data = request.json
        message = data.get('message')
        messages.append(message)
        try:
            response = chatbot.send_prompt(message['text'])
            messages.append({ 'text': response, 'sender': 'Bot'})
        except Exception as e:
            print(f"Error: {e}")
        return jsonify({'message': message}), 201

if __name__ == '__main__':
    app.run(debug=True)
