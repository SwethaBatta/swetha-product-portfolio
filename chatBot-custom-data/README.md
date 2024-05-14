## Building a chatBot using OpenAI trained on custom data
Developed a chatbot using OpenAI with RAG (Retrieval-Augmented Generation) LLM (Large Language Model) to query specific non-sensitive user data and frequently asked questions, thereby enhancing users' self-service abilities and reducing the volume 
of requests processed by the Customer Service team

**Technologies used:** OpenAI GPT model, React js, Python Flask
 

## Demo
Here is a demo of how the chatbot works

https://github.com/SwethaBatta/genai/assets/33192347/f28ac9e4-b093-420f-aa21-254d1f6af214

## Product Features
<pre>
* <b>Conversational AI:</b> The chatbot utilizes the RAG Language Model architecture to generate contextually relevant responses to user queries in text format.
* <b>Custom knowledge base:</b> The responses are trained using a custom knowledge base which helps users with easy access to frequently requested information.
* <b>Personalization:</b> The bot also helps with personalization as it provides information related to the user's account and assists the user in completing tasks that would otherwise require the user to navigate to the website or app. 
</pre>


## Backend (Python):
```
* Set up Flask: Created a Flask application to serve as the backend.
* Integrated RAG LLM: Used the OpenAI with RAG(Retrieval-Augmented Generation) LLM(Large Language Model - OpenAI's GPT model) to train the model on custom knowledge base
* Created API endpoints and handled requests: Defined API endpoints for handling text input from the frontend and processing the response with custom knowledge base and implemented logic to handle the requests received from the frontend.
```

## Frontend (React):
```
* Set up React app: Created a React application to serve as the frontend.
* Designed chat interface: Developed a chat interface where users can provide input through text and view responses.
* Sending requests to backend: Implemented functionality to send user input to the backend and receive responses.
```
