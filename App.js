import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './index.css'; // Ensure this path is correct

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [chatMessages, setChatMessages] = useState([]);
  const [isListening, setIsListening] = useState(false);
  const [chatInput, setChatInput] = useState('');
  const [availableVoices, setAvailableVoices] = useState([]);

  useEffect(() => {
    const loadVoices = () => {
      const voices = window.speechSynthesis.getVoices();
      setAvailableVoices(voices);
      console.log('Available voices:', voices);
    };

    window.speechSynthesis.onvoiceschanged = loadVoices;
    loadVoices();
  }, []);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setSummary('');
    setChatMessages([]);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
      setSummary('');
      setChatMessages([]);
      e.dataTransfer.clearData();
    }
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file to upload.");
      return;
    }
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/upload', formData);
      setSummary(response.data.summary);
      setChatMessages([
        {
          user: 'Bot',
          message:
            "The document has been uploaded and I've processed it for Q&A. Ask me anything directly from its content!",
        },
      ]);
    } catch (error) {
      console.error('Upload failed:', error);
      setSummary('Something went wrong during summarization. Please try again.');
      setChatMessages([]);
    } finally {
      setLoading(false);
    }
  };

  const handleSpeak = () => {
    if (summary) {
      if (window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel(); // Cancel ongoing speech if any
      }

      const utterance = new SpeechSynthesisUtterance(summary);
      utterance.lang = 'en-US';

      const microsoftVoice = availableVoices.find(
        (voice) => voice.name.includes('Microsoft') && voice.lang === 'en-US'
      );

      if (microsoftVoice) {
        utterance.voice = microsoftVoice;
      } else {
        console.warn('No Microsoft voice found. Using default voice.');
      }

      window.speechSynthesis.speak(utterance);
    }
  };

  const handleDownload = () => {
    if (summary) {
      const element = document.createElement('a');
      const fileBlob = new Blob([summary], { type: 'text/plain' });
      element.href = URL.createObjectURL(fileBlob);
      element.download = 'summary.txt';
      document.body.appendChild(element);
      element.click();
      document.body.removeChild(element);
    }
  };

  const handleChatInputSend = async (input) => {
    if (!input.trim()) return;

    const newUserMessage = { user: 'You', message: input };
    setChatMessages((prev) => [...prev, newUserMessage]);
    setChatInput('');

    try {
      const response = await axios.post('http://localhost:5000/chatbot', { message: input });
      const botResponse = response.data.response;
      setChatMessages((prev) => [...prev, { user: 'Bot', message: botResponse }]);
    } catch (error) {
      console.error('Chatbot interaction failed:', error);
      setChatMessages((prev) => [
        ...prev,
        {
          user: 'Bot',
          message: 'I am sorry, I could not connect to the chatbot service. Please try again later.',
        },
      ]);
    }
  };

  const startListening = () => {
    if (!('webkitSpeechRecognition' in window)) {
      alert('Speech recognition is not supported in your browser.');
      return;
    }

    setIsListening(true);
    const recognition = new window.webkitSpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.start();

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setIsListening(false);
      handleChatInputSend(transcript);
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      setIsListening(false);
      let errorMessage = 'Speech recognition error.';
      if (event.error === 'no-speech') {
        errorMessage = 'No speech detected. Please try again.';
      } else if (event.error === 'not-allowed') {
        errorMessage = 'Microphone access denied. Please allow access.';
      }
      setChatMessages((prev) => [...prev, { user: 'Bot', message: errorMessage }]);
    };
  };

  return (
    <div className="background">
      <div className="header">
        <div className="logo">
          <span className="logo-text">DocuNexus</span>
        </div>
      </div>

      <div className="title-container">
        <h1 className="title">AI Summarizer</h1>
        <p className="subtitle">Summarize Any PDF or Document Instantly</p>
      </div>

      <div className="upload-card fade-in">
        <div className="upload-box" onDragOver={handleDragOver} onDrop={handleDrop}>
          <label htmlFor="file-upload" className="drop-area">
            <div className="plus-icon">+</div>
            <div className="file-name">
              {file
                ? file.name.length > 40
                  ? `${file.name.slice(0, 40)}...`
                  : file.name
                : 'Drop your file here or click to upload'}
            </div>
          </label>
          <input
            id="file-upload"
            type="file"
            accept=".pdf,.docx"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
        </div>

        <div className="button-container">
          <button onClick={handleUpload} disabled={loading || !file} className="upload-button">
            {loading ? 'Processing Document...' : 'Upload & Process'}
          </button>
        </div>

        {summary && (
          <div className="summary-box fade-in">
            <h2>📋 Document Summary</h2>
            <p className="summary-text">{summary}</p>
            <div className="summary-actions">
              <button onClick={handleSpeak} className="summary-btn speak">
                🔊 Listen
              </button>
              <button onClick={handleDownload} className="summary-btn download">
                ⬇ Download
              </button>
            </div>
          </div>
        )}

        {summary && (
          <div className="chatbot-container fade-in">
            <div className="chatbox">
              <div className="chat-messages">
                {chatMessages.map((msg, index) => (
                  <div
                    key={index}
                    className={`chat-message ${msg.user === 'You' ? 'user-message' : 'bot-message'}`}
                  >
                    <strong>{msg.user}: </strong>
                    {msg.message}
                  </div>
                ))}
              </div>
              <div className="chat-input">
                <input
                  type="text"
                  placeholder="Ask something about the document..."
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      handleChatInputSend(chatInput);
                    }
                  }}
                />
                <button className="microphone-btn" onClick={startListening} disabled={isListening}>
                  {isListening ? 'Listening...' : '🎙'}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileUpload;
