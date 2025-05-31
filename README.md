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
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

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

### Example with curl (Linux/macOS):
curl -X POST http://localhost:8000/external_input \
     -H "Content-Type: application/json" \
     -d '{"Park": "Arrived", "Siren&Lights": "Flashing", "Behaviour": "Calm"}'


## 🖥 Project Structure

Empathy/
├── run.py # Main entry point to start the Flask app
├── requirements.txt # List of Python dependencies
├── README.md # Project documentation (this file)
├── .gitignore # Files and folders Git should ignore
├── .env # Environment variables (e.g., OpenAI API key)
├── package.json # Node-related package config (if needed)
├── package-lock.json # npm lock file
├── venv/ # Python virtual environment (not pushed to Git)
├── static/
│ └── node_modules/
│ └── dotenv/ # Example: dotenv config under node_modules
├── src/
│ └── app/
│ ├── init.py # Flask app factory and configuration setup
│ ├── analysis.py # Core logic for empathy scoring using OpenAI
│ ├── config.py # App configuration settings
│ ├── data_store.py # Global in-memory data storage
│ ├── graphing.py # Empathy progression graph logic
│ ├── pdf_generator.py # PDF report generation logic
│ ├── routes.py # All backend route endpoints
│ ├── word_utils.py # Word cloud generation and empathy word parsing
│ └── templates/
│ ├── index.html # Graph page
│ ├── results.html # Results and word cloud report
│ └── welcome.html # Main dashboard with live input feed

## 📤 PDF Export
You can download a full empathy report in PDF format.

### Endpoint:
GET /generate_pdf
Or use the button in the web UI.

Includes:
Dialogue and empathy scores table
Score progression graph
Word cloud summary


## 🧪 Developer Notes
Restart the app to reset stored messages
Scores and messages are stored in memory (non-persistent)
Use browser console and terminal logs for debugging
Customize logic in analysis.py or pdf_generator.py

