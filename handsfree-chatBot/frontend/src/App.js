import React, { useState, useEffect, useRef } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import './App.css';
import './ChatApp.css';
import axios from 'axios';
import Logo from './Chatbot_Icon.png';

function App() {
  const [audioUrl, setAudioUrl] = useState('');
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const chatboxRef = useRef(null);

  const [pauseTimer, setPauseTimer] = useState(null);
  const pauseThreshold = 5000; // Adjust this threshold as needed (in milliseconds)
  const { transcript, resetTranscript, listening } = useSpeechRecognition();
  const audioRef = useRef(null);
  const [botSpeaking, setBotSpeaking] = useState(false);

  useEffect(() => {
    // Automatically start listening when the component mounts
    SpeechRecognition.startListening({
      continuous: true
    });
    if (!SpeechRecognition.browserSupportsSpeechRecognition()) {
      return null
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, []);

  useEffect(() => {
    if (!botSpeaking && transcript !== '') {
      // Clear the pause timer whenever new speech input is received
      clearTimeout(pauseTimer);
      // Start the pause timer
      setPauseTimer(setTimeout(handlePause, pauseThreshold));
    }

    if (!botSpeaking) {
      SpeechRecognition.startListening({continuous: true});
    }
  }, [transcript, botSpeaking]);

  useEffect(() => {
    //3️⃣ bring the last item into view        
    chatboxRef.current?.lastElementChild?.scrollIntoView();
    if (messages[messages.length-1] && messages[messages.length-1]['sender'] === 'Bot') {
      handleTextToSpeech(messages[messages.length-1]['text']);
    }
  }, [messages]);

  const playAudio = () => {
    if (audioRef.current) {
      audioRef.current.play()
        .catch(error => {
          console.error('Error playing audio:', error);
        });
    }
  };

  const handlePause = () => {
    // When a pause is detected, stop listening and set the recognized text as the message
    SpeechRecognition.stopListening();
    setInput(transcript);
    sendMessage(transcript);
    resetTranscript();
  };

  const fetchData = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/data');
      setMessages(response.data.messages);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  const sendMessage = async (input) => {
    if (input.trim() !== '') {
      resetTranscript();
      try {
        await axios.post('http://localhost:5000/api/data', {message: { text: input, sender: 'Swetha' }});
        fetchData(); // Fetch updated messages after sending a new one
      } catch (error) {
        console.error('Error sending message:', error);
      }
    }
    setInput('');
  };

  const formatDate = (date: Date) => {
    return new Date().toLocaleString('en-US', { hour: 'numeric', minute: 'numeric', hour12: true });
  }

  const handleTextToSpeech = async (messageText) => {
    const response = await fetch('http://localhost:5000/text-to-speech', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ text: messageText })
    });
    const data = await response.json();
    setAudioUrl(`data:audio/wav;base64,${data.audio_base64}`);
    // Set botSpeaking to true when bot starts speaking
    setBotSpeaking(true);
    // Stop listening while the bot is speaking
    SpeechRecognition.abortListening();
    playAudio();
    // Set a timeout to resume listening after bot stops speaking (simulate bot's speech duration)
    setTimeout(() => setBotSpeaking(false), 5000); // Adjust duration as needed
  };

  return (
    <div className="App">
      <div className="chat-container">
        <header className="message-header">
          <img src={Logo} alt="Logo" width={60} height={60}/>
          {/* <a href="https://www.flaticon.com/free-icons/robot" title="robot icons">Robot icons created by Freepik - Flaticon</a> */}
            <div className="message-header-text">
              <div>Chat support</div>
              <div className="subtext">Online</div>
            </div>
        </header>
        <div className="chat-app">
            <div ref={chatboxRef} className="chat-messages">
                {messages.map((message, index) => (
                    <div key={index} className={`message ${message.sender}`}>
                        <div className="msg-info">
                        <div className="msg-info-name">{message.sender}</div>
                        <div className="msg-info-time">{formatDate(new Date())}</div>
                        </div>
                        <div>{message.text}</div>
                        
                </div>
                ))}
            </div>
            <div className="chat-input">
                <input
                type="text"
                placeholder="Type your message..."
                value={transcript ? transcript : input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={(e) => {
                    if (e.key === "Enter")
                        sendMessage(e.target.value);
                    }}
                />
                <button type="submit" onClick={() => sendMessage(input)}>Send</button>
            </div>
        </div>
    </div>
    {audioUrl && <audio ref={audioRef} src={audioUrl} controls autoPlay />}
    </div>
  );
}

export default App;