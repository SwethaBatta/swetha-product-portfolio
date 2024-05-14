// ChatApp.js
import React, { useState, useEffect, useRef } from 'react';
import './ChatApp.css';
import axios from 'axios';
import ReactLogo from './chat-bot-icon.png';

function ChatApp() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const chatboxRef = useRef(null);

  useEffect(() => {
    const interval = setInterval(fetchData, 2000); // Poll server every 2 seconds for new messages
    return () => clearInterval(interval); // Cleanup interval on component unmount
  }, []);

  useEffect(() => {
    //3️⃣ bring the last item into view        
    chatboxRef.current?.lastElementChild?.scrollIntoView();
  }, [messages]);

  const fetchData = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/data');
      setMessages(response.data.messages);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  const sendMessage = async () => {
    if (input.trim() !== '') {
      setInput('');
      try {
        await axios.post('http://localhost:5000/api/data', {message: { text: input, sender: 'Swetha' }});
        fetchData(); // Fetch updated messages after sending a new one
      } catch (error) {
        console.error('Error sending message:', error);
      }
    }
  };

  const formatDate = (date: Date) => {
    return new Date().toLocaleString('en-US', { hour: 'numeric', minute: 'numeric', hour12: true });
  }

  return (
    <div className="chat-container">
        <header className="message-header">
          <img src={ReactLogo} alt="React Logo" width={60} height={60}/>
          {/* <a href="https://www.freepik.com/icons/chatbot/2#uuid=05eec70c-ada4-476c-9fac-8062b6cfad3a">Icon by Kalashnyk</a> */}
            <div className="message-header-text">
              <div>Chat support 24/7</div>
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
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={(e) => {
                    if (e.key === "Enter")
                        sendMessage();
                    }}
                />
                <button type="submit" onClick={() => sendMessage()}>Send</button>
            </div>
        </div>
    </div>
  );
}

export default ChatApp;
