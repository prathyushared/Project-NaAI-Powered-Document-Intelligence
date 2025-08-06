import React, { useState } from 'react';
import { OPENAI_API_KEY } from './config'; // Import your API key

const DocumentChat = ({ documentText }) => {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState('');

  const handleAsk = async () => {
    if (!question.trim()) return;

    try {
      const result = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${OPENAI_API_KEY}`,  // Using your API key here
        },
        body: JSON.stringify({
          model: 'gpt-4',
          messages: [
            { role: 'system', content: `You are an assistant summarizing and answering questions based on this document: ${documentText}` },
            { role: 'user', content: question },
          ],
          temperature: 0.5,
        }),
      });

      const data = await result.json();
      if (data?.choices?.[0]?.message?.content) {
        setResponse(data.choices[0].message.content);
      } else {
        setResponse('❌ Failed to get a valid response.');
      }
    } catch (error) {
      console.error(error);
      setResponse('❌ Error fetching response.');
    }
  };

  return (
    <div className="document-chat">
      <h2>Chat with Your Summary</h2>
      <input
        type="text"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Ask something about the document..."
        className="chat-input"
      />
      <button onClick={handleAsk} className="ask-button">Ask</button>

      {response && (
        <div className="response-box">
          <strong>Answer:</strong>
          <p>{response}</p>
        </div>
      )}
    </div>
  );
};

export default DocumentChat;
