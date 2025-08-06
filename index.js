// src/index.js
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css'; // this includes Tailwind CSS
import App from './App'; // this is where your UI lives

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
