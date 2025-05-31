# 🧠 Empathy Analysis Dashboard

This project is a Flask-based web application designed to visualize and analyze empathetic responses from conversational interactions in real time. It integrates data input (external or manual), dynamic web visualization, emotional graphing, word cloud generation, and PDF export capabilities.

---

## ✨ Features

- 🗣 Real-time dialogue entry and empathy scoring  
- 🧾 External input API for Park, Siren&Lights, and Behavior data  
- 📊 Interactive dashboard with message display boxes  
- 📈 Line graph tracking empathy progression  
- ☁️ Word cloud visualization of dialogue content  
- 📄 Downloadable PDF report generation  
- 📡 Server-Sent Events (SSE) stream for live UI updates  

---

## 🛠 Setup Instructions

### 1. Clone the Repository

### 2. Create a Python Virtual Environment (Recommended)
- python -m venv venv
- source venv/bin/activate      # macOS/Linux
- venv\Scripts\activate         # Windows

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Run the Application
python run.py

Open your browser and go to:
http://localhost:8000/

## 📬 API Endpoint

POST /external_input
Send external JSON input using Postman or curl.

### Payload Format:
{
  "Park": "Message about parking",
  "Siren&Lights": "Lights-related input",
  "Behaviour": "Observed behavior input"
}

## 🖥 Project Structure

Empathy/
- src/
  - app/
    - __init__.py           # Flask app setup & factory pattern
    - analysis.py           # Empathy scoring logic (e.g., using OpenAI)
    - config.py             # App configuration
    - data_store.py         # Stores in-memory session data
    - graphing.py           # Score progression graph generation
    - pdf_generator.py      # PDF report formatting and generation
    - routes.py             # All Flask route endpoints (UI & API)
    - word_utils.py         # Word cloud and keyword analysis
    - templates/            # Jinja2 HTML templates
      - welcome.html        # Main live dashboard UI
      - index.html          # Graph visualization UI
      - results.html        # Summary + word cloud view

- static/
  - node_modules/
    - dotenv/              # Local JS/Node dependencies (if applicable)

- .env                     # Need to create (e.g., OpenAI key)
- .gitignore               # Files/folders Git should ignore
- package-lock.json        # Optional Node.js lock file
- package.json             # Optional Node.js config
- README.md                # Project documentation (you are reading this)
- requirements.txt         # Python dependencies
- run.py                   # Main entry point to start the Flask app


## 📤 PDF Export
- You can download a full empathy report in PDF format.

### Endpoint:
- GET /generate_pdf
- Or use the button in the web UI.

Includes:
- Dialogue and empathy scores table
- Score progression graph
- Word cloud summary


## 🧪 Developer Notes
- Restart the app to reset stored messages
- Scores and messages are stored in memory (non-persistent)
- Use browser console and terminal logs for debugging
- Customize logic in analysis.py or pdf_generator.py

