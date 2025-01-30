from flask import Flask, render_template, request, jsonify, Response
import pandas as pd
import plotly.express as px
import plotly
import json
from flask_cors import CORS
from openai import OpenAI
import os
from datetime import datetime
from dotenv import load_dotenv
import time
from wordcloud import WordCloud  # Add this import at the top
from collections import Counter

load_dotenv()  # Add this near the top of file

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Add this near the top of the file, after the dialogue_data definition
prompt = """Please analyze the following text for empathy based on these criteria:

1. Conversation's Empathy Level (0.6 max)
- Level 0 (0.0): Denial of perspective
- Level 1 (0.1): Automatic recognition
- Level 2 (0.2): Implicit recognition
- Level 3 (0.3): Acknowledgment without pursuit
- Level 4 (0.4): Acknowledgment with pursuit
- Level 5 (0.5): Confirmation
- Level 6 (0.6): Shared feeling

2. Movement & Gesture (0.1 max)
- Level 1 (0.0): Abrupt or intimidating
- Level 2 (0.05): Calm, non-threatening
- Level 3 (0.1): Fluid, empathic, well-paced

3. Tone (0.05 max)
- Level 1 (0.0): Cold, dismissive, monotone
- Level 2 (0.025): Slight warmth but may feel forced
- Level 3 (0.05): Genuinely caring, fosters trust

4. Active Listening & Responsiveness (0.05 max)
- Level 1 (0.0): Ignores key elements, inattentive
- Level 2 (0.017): Addresses some aspects but misses details
- Level 3 (0.034): Reflects careful listening, addresses key points
- Level 4 (0.05): Highly responsive, integrates words in-depth

Please analyze this text and provide scores in this exact JSON format:
{{
    "conversation_score": 0.0,
    "movement_score": 0.0,
    "tone_score": 0.0,
    "listening_score": 0.0,
    "raw_total": 0.0,
    "cumulative_score": 0.0,
    "increment": 0.0,
    "final_score": 0.0,
    "explanation": "Brief explanation of scoring with breakdown by category"
}}

Text to analyze: "{text}" """

dialogue_data = {
    'dialogue': [],
    'empathy_scores': [],
    'analysis_history': []
}

# Add this near the top with other initializations
starting_score = 0.0  # Will be updated via POST
previous_numeric_score = None  # To track the previous numeric score

# Add a variable to track cumulative score
cumulative_total = 0.0

# Add this near the top with other initializations
external_messages = {
    'park': [],
    'lights': [],
    'behavior': []
}

# Add this list of empathy words with their frequencies
EMPATHY_WORDS = {
    # Nouns
    'help': 8, 'support': 8, 'assistance': 7, 'backup': 6, 'safety': 8,
    'focus': 7, 'team': 8, 'confidence': 7, 'reassurance': 8, 'care': 9,
    'encouragement': 7, 'assurance': 8, 'teamwork': 9, 'understanding': 8,
    'presence': 7, 'stability': 7, 'partnership': 7, 'trust': 8, 'connection': 7,
    'empathy': 9, 'calmness': 8,
    
    # Adjectives
    'calm': 9, 'steady': 8, 'safe': 8, 'focused': 7, 'reassuring': 8,
    'gentle': 8, 'kind': 8, 'supportive': 9, 'firm': 6, 'composed': 7,
    'patient': 8, 'encouraging': 8, 'caring': 9, 'controlled': 7,
    'grounded': 7, 'attentive': 8, 'responsive': 8, 'considerate': 7,
    'careful': 7, 'tender': 6, 'mindful': 8, 'comforting': 8, 'warm': 7,
    'compassionate': 9,
    
    # Verbs
    'support': 9, 'focus': 8, 'help': 9, 'assist': 8, 'guide': 8,
    'encourage': 8, 'reassure': 9, 'breathe': 7, 'care': 9, 'stabilize': 7,
    'soothe': 8, 'comfort': 8, 'listen': 9, 'acknowledge': 8, 'ease': 7,
    'center': 7, 'monitor': 6, 'respond': 7, 'safeguard': 7
}

def estimate_empathy(text):
    """Use ChatGPT to analyze empathy levels based on the guidelines"""
    global cumulative_total
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing empathy in conversations."},
                {"role": "user", "content": prompt.format(text=text)}
            ],
            temperature=0.3,
            response_format={ "type": "json_object" }
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Calculate raw total from subscores
        raw_total = (
            result['conversation_score'] +
            result['movement_score'] +
            result['tone_score'] +
            result['listening_score']
        )
        
        # Update cumulative total
        cumulative_total += raw_total
        
        # Calculate increment based on cumulative score
        if cumulative_total < 0.2:
            increment = 0.0
        elif cumulative_total < 0.8:
            increment = 0.1
        else:
            increment = 0.2
            
        # Calculate final score
        final_score = raw_total + increment
        
        # Update result with calculated values
        result['raw_total'] = raw_total
        result['cumulative_score'] = cumulative_total
        result['increment'] = increment
        result['final_score'] = final_score
        
        # Print detailed scoring breakdown to terminal
        print(f"\nText: {text}")
        print(f"Detailed Scoring Breakdown:")
        print(f"1. Conversation's Empathy Level: {result['conversation_score']:.3f}")
        print(f"2. Movement & Gesture: {result['movement_score']:.3f}")
        print(f"3. Tone: {result['tone_score']:.3f}")
        print(f"4. Active Listening: {result['listening_score']:.3f}")
        print(f"Raw Total: {raw_total:.3f}")
        print(f"Cumulative Score (Including Previous): {cumulative_total:.3f}")
        print(f"Score Increment: +{increment:.3f}")
        print(f"Final Score: {final_score:.3f}")
        
        # Store detailed analysis for results page
        dialogue_data['analysis_history'].append({
            'text': text,
            'scores': result,
            'timestamp': datetime.now().isoformat()
        })
        
        return final_score
        
    except Exception as e:
        print(f"Error in GPT analysis: {str(e)}")
        return 0.3

def analyze_emotions(text):
    """
    Analyze emotions in the given text using OpenAI's API
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an emotion analyzer. Analyze the emotional content of the following text."},
            {"role": "user", "content": text}
        ],
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/graph')
def graph():
    return render_template('index.html')

@app.route('/add_dialogue', methods=['POST'])
def add_dialogue():
    global starting_score
    text = request.json.get('text', '')
    input_type = request.json.get('input_type', '')
    initial_score = request.json.get('initial_score')
    
    # Check if input is numeric (e.g., "6/10" or "6")
    try:
        if '/' in str(text):
            numerator, denominator = map(float, str(text).split('/'))
            empathy_score = numerator / denominator
        elif str(text).replace('.', '').isdigit():
            # If it's a number without fraction, assume out of 10
            empathy_score = float(text) / 10
        else:
            # If not numeric, process as normal text
            if text and input_type:
                text = f"[{input_type.upper()}] {text}"
            
            if initial_score is not None:
                starting_score = float(initial_score)
                empathy_score = starting_score
                print(f"\nInitial Score Set: {starting_score:.3f}")
            elif text:
                empathy_score = estimate_empathy(text)
            else:
                empathy_score = dialogue_data['empathy_scores'][-1] if dialogue_data['empathy_scores'] else starting_score
                print(f"\nMaintaining previous score: {empathy_score:.3f}")
    except Exception as e:
        print(f"Error processing input: {str(e)}")
        empathy_score = dialogue_data['empathy_scores'][-1] if dialogue_data['empathy_scores'] else 0.0

    # Add dialogue text and score to the data
    dialogue_data['dialogue'].append(text if text else "No Input")
    dialogue_data['empathy_scores'].append(empathy_score)

    # Prepare DataFrame for Plotly
    df = pd.DataFrame({
        'Dialogue_Turn': range(len(dialogue_data['dialogue'])),
        'Empathy_Level': dialogue_data['empathy_scores'],
        'Text': dialogue_data['dialogue']
    })

    # Create Step Line Graph
    fig = px.line(df, x='Dialogue_Turn', y='Empathy_Level',
                  title='Real-time Empathy Level Progression',
                  labels={'Dialogue_Turn': 'Dialogue Progression',
                         'Empathy_Level': 'Estimated Empathy Level (0 to 1)'},  # Updated max to 1
                  markers=True)

    # Update the graph to use step line format
    fig.update_traces(
        mode='lines+markers',  # Show both lines and markers
        line_shape='hv',  # Horizontal-Vertical (step graph behavior)
        hovertemplate="<br>".join([
            "Turn: %{x}",
            "Empathy Level: %{y:.2f}",
            "Text: %{customdata}"
        ]),
        customdata=dialogue_data['dialogue']
    )

    # Add a horizontal threshold line at 0.3
    fig.update_layout(
        yaxis_range=[0, 1],
        xaxis_range=[-0.1, max(len(dialogue_data['dialogue']) - 1, 0.5)],
        hovermode='x',
        shapes=[
            dict(
                type='line',
                yref='y',
                y0=0.3,
                y1=0.3,
                xref='paper',
                x0=0,
                x1=1,
                line=dict(
                    color='gray',
                    width=1,
                    dash='dash'
                )
            )
        ]
    )

    # Convert graph to JSON for rendering
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return jsonify({
        'graph': graphJSON,
        'empathy_score': empathy_score,
        'text': text,
        'dialogue_history': dialogue_data['dialogue'],
        'empathy_scores': dialogue_data['empathy_scores'],
        'analysis_history': dialogue_data['analysis_history']
    })


@app.route('/input', methods=['GET', 'POST'])
def input_text():
    if request.method == 'GET':
        # Return only the graph without the input form
        return render_template('index.html')
    
    # Handle POST request
    if request.is_json:
        text = request.json.get('text', '')
    else:
        text = request.form.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Calculate empathy score
    empathy_score = estimate_empathy(text)
    
    # Update the dialogue data for the graph
    dialogue_data['dialogue'].append(text)
    dialogue_data['empathy_scores'].append(empathy_score)
    
    # Create updated graph
    df = pd.DataFrame({
        'Dialogue_Turn': range(0, len(dialogue_data['dialogue'])),
        'Empathy_Level': dialogue_data['empathy_scores'],
        'Text': dialogue_data['dialogue']
    })
    
    fig = px.line(df, x='Dialogue_Turn', y='Empathy_Level',
                  title='Real-time Empathy Level Progression',
                  labels={'Dialogue_Turn': 'Dialogue Progression',
                         'Empathy_Level': 'Estimated Empathy Level (0 to 1)'},
                  markers=True)
    
    fig.update_traces(
        hovertemplate="<br>".join([
            "Turn: %{x}",
            "Empathy Level: %{y:.2f}",
            "Text: %{customdata}"
        ]),
        customdata=dialogue_data['dialogue']
    )
    
    fig.update_layout(
        yaxis_range=[0, 1],
        xaxis_range=[-0.1, max(len(dialogue_data['dialogue']) - 1, 0.5)],
        hovermode='x',
        shapes=[
            dict(
                type='line',
                yref='y',
                y0=0.3,
                y1=0.3,
                xref='paper',
                x0=0,
                x1=1,
                line=dict(
                    color='gray',
                    width=1,
                    dash='dash'
                )
            )
        ]
    )
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return jsonify({
        'graph': graphJSON,
        'empathy_score': empathy_score,
        'text': text,
        'dialogue_history': dialogue_data['dialogue'],
        'empathy_scores': dialogue_data['empathy_scores']
    })

@app.route('/analyze', methods=['GET'])
def analyze_text():
    # Get text from query parameters instead of form data
    text = request.args.get('text')
    
    if not text:
        return jsonify({'error': 'No text provided. Please add ?text=your_text to the URL'}), 400
    
    try:
        # Analyze the text for emotions
        emotions = analyze_emotions(text)
        return jsonify({
            'text': text,
            'emotions': emotions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save-config', methods=['POST'])
def save_config():
    config = request.json
    # Store config in session or database
    return jsonify({'status': 'success'})

@app.route('/results')
def results():
    # Get the latest analysis if available
    latest_analysis = None
    final_score = 0
    if dialogue_data['analysis_history']:
        latest_analysis = dialogue_data['analysis_history'][-1]
        final_score = latest_analysis['scores']['final_score']
    
    # Create word cloud from empathy words
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        max_words=100,
        relative_scaling=0.5,
        prefer_horizontal=0.7
    ).generate_from_frequencies(EMPATHY_WORDS)
    
    # Convert wordcloud to base64 for embedding in HTML
    import io
    import base64
    img = io.BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    wordcloud_b64 = base64.b64encode(img.getvalue()).decode()

    return render_template(
        'results.html',
        final_score=final_score,
        wordcloud_b64=wordcloud_b64,
        latest_analysis=latest_analysis
    )

@app.route('/external_input', methods=['POST'])
def external_input():
    """Handle external POST requests with all three types"""
    try:
        data = request.json
        park = data.get('Park', '')
        lights = data.get('Siren&Lights', '')
        behavior = data.get('Behaviour', '')
        
        timestamp = datetime.now().isoformat()
        
        # Store all messages if they exist
        if park:
            external_messages['park'].append({
                'message': park,
                'timestamp': timestamp
            })
            print(f"\nReceived Park input: {park}")
            
        if lights:
            external_messages['lights'].append({
                'message': lights,
                'timestamp': timestamp
            })
            print(f"\nReceived Siren&Lights input: {lights}")
            
        if behavior:
            external_messages['behavior'].append({
                'message': behavior,
                'timestamp': timestamp
            })
            print(f"\nReceived Behaviour input: {behavior}")
        
        return jsonify({
            'status': 'success',
            'timestamp': timestamp,
            'Park': park,
            'Siren&Lights': lights,
            'Behaviour': behavior
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/get_latest_messages', methods=['GET'])
def get_latest_messages():
    """Return any new messages since last poll"""
    # You'll need to implement message storage and retrieval logic
    return jsonify({
        'messages': [
            {
                'type': 'park',
                'message': 'New message',
                'timestamp': datetime.now().isoformat()
            }
            # ... other new messages
        ]
    })

@app.route('/stream')
def stream():
    def event_stream():
        while True:
            # Check for new messages in external_messages
            for msg_type in external_messages:
                if external_messages[msg_type]:
                    message = external_messages[msg_type].pop(0)  # Get and remove the first message
                    data = {
                        'type': msg_type,
                        'message': message['message'],
                        'timestamp': message['timestamp']
                    }
                    yield f"data: {json.dumps(data)}\n\n"
            time.sleep(1)  # Check every second

    return Response(event_stream(), mimetype="text/event-stream")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)