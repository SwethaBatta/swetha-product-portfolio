from werkzeug.utils import url_quote
from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
from configparser import ConfigParser
import query_knowledgebase as chatApp

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

config = ConfigParser()
config.read('credentials.ini')
api_key = config['open_ai']['API_KEY']
chatbot = chatApp.ChatBot(api_key=api_key)

# Dummy messages
messages = [{'text': 'Hi, this is xxx Customer support. How can I help you today?', 'sender': 'Bot'}]

@app.route('/api/data', methods=['GET', 'POST'])
def handle_messages():
    if request.method == 'GET':
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