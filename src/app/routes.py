from flask import render_template, request, jsonify, Response
from datetime import datetime
import json
import time
import io
import base64
from wordcloud import WordCloud


from src.app.analysis import estimate_empathy, analyze_emotions
from src.app.graphing import create_graph
from src.app.pdf_generator import generate_pdf
from src.app.word_utils import analyze_empathy_words
from src.app.data_store import dialogue_data, external_messages
from src.app.data_store import starting_score, previous_numeric_score, cumulative_total


def register_routes(app):

    @app.route('/')
    def welcome():
        return render_template(
            'welcome.html',
            park_messages=external_messages['park'],
            light_messages=external_messages['lights'],
            behavior_messages=external_messages['behavior']
    )


    @app.route('/graph')
    def graph():
        return render_template('index.html')

    @app.route('/add_dialogue', methods=['POST'])
    def add_dialogue():
        global starting_score, previous_numeric_score

        data = request.json or {}
        text = data.get('text', '').strip()
        input_type = data.get('input_type', '')
        initial_score = data.get('initial_score')

        try:
            if '/' in text:
                numerator, denominator = map(float, text.split('/'))
                empathy_score = numerator / denominator
            elif text.replace('.', '').isdigit():
                empathy_score = float(text) / 10
                previous_numeric_score = empathy_score
            else:
                if not text:
                    empathy_score = previous_numeric_score if previous_numeric_score is not None else starting_score
                else:
                    if input_type:
                        text = f"[{input_type.upper()}] {text}"

                    if initial_score is not None:
                        starting_score = float(initial_score)
                        empathy_score = starting_score
                        previous_numeric_score = starting_score
                    else:
                        empathy_score = estimate_empathy(text)
                        previous_numeric_score = empathy_score

            if text:
                dialogue_data['dialogue'].append(text)
                dialogue_data['empathy_scores'].append(empathy_score)

            return jsonify({
                'graph': create_graph(),
                'empathy_score': empathy_score,
                'text': text,
                'dialogue_history': dialogue_data['dialogue'],
                'empathy_scores': dialogue_data['empathy_scores'],
                'analysis_history': dialogue_data['analysis_history']
            })

        except Exception as e:
            print(f"Error processing input: {str(e)}")
            return jsonify({'error': str(e)}), 400

    @app.route('/input', methods=['GET', 'POST'])
    def input_text():
        if request.method == 'GET':
            return render_template('index.html')

        text = request.json.get('text', '') if request.is_json else request.form.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        empathy_score = estimate_empathy(text)
        dialogue_data['dialogue'].append(text)
        dialogue_data['empathy_scores'].append(empathy_score)

        return jsonify({
            'graph': create_graph(),
            'empathy_score': empathy_score,
            'text': text,
            'dialogue_history': dialogue_data['dialogue'],
            'empathy_scores': dialogue_data['empathy_scores']
        })

    @app.route('/analyze', methods=['GET'])
    def analyze_text():
        text = request.args.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        try:
            emotions = analyze_emotions(text)
            return jsonify({'text': text, 'emotions': emotions})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/results')
    def results():
        latest_analysis = None
        final_score = 0.0
        if dialogue_data['analysis_history']:
            latest_analysis = dialogue_data['analysis_history'][-1]
            final_score = latest_analysis.get('scores', {}).get('final_score', 0.0)

        conversation_history = " ".join(dialogue_data['dialogue'])
        word_frequencies, empathy_word_details = analyze_empathy_words(conversation_history)

        wordcloud_b64 = None
        if word_frequencies:
            wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_frequencies)
            img = io.BytesIO()
            wc.to_image().save(img, format='PNG')
            img.seek(0)
            wordcloud_b64 = base64.b64encode(img.getvalue()).decode()

        return render_template(
            'results.html',
            final_score=final_score,
            wordcloud_b64=wordcloud_b64,
            latest_analysis=latest_analysis,
            has_empathy_words=bool(word_frequencies),
            empathy_word_details=empathy_word_details
        )

    
    @app.route('/external_input', methods=['POST'])
    def external_input():
        """Handle external POST requests with all three types"""
        try:
            data = request.json
            park = data.get('Park', '')
            lights = data.get('Siren&Lights', '')
            behavior = data.get('Behaviour', '')
            
            timestamp = datetime.now().isoformat(timespec='seconds')
            
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
        return jsonify({
            'messages': [
                {
                    'type': 'park',
                    'message': 'New message',
                    'timestamp': datetime.now().isoformat()
                }
            ]
        })

    @app.route('/stream')
    def stream():
        def event_stream():
            while True:
                sent_any = False
                for msg_type in external_messages:
                    if external_messages[msg_type]:
                        message = external_messages[msg_type].pop(0)
                        data = {
                            'type': msg_type,
                            'message': message['message'],
                            'timestamp': message['timestamp'].replace('T', ' ')[:-7]
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                        sent_any = True
                if not sent_any:
                    time.sleep(1)

        return Response(event_stream(), mimetype="text/event-stream")


    @app.route('/generate_pdf')
    def pdf():
        return generate_pdf()

    @app.route('/reset', methods=['POST'])
    def reset_data():
        dialogue_data['dialogue'] = []
        dialogue_data['empathy_scores'] = []
        dialogue_data['analysis_history'] = []
        global cumulative_total
        cumulative_total = 0.0

        # Also reset external messages
        external_messages['park'].clear()
        external_messages['lights'].clear()
        external_messages['behavior'].clear()

        return jsonify({'status': 'success', 'redirect': '/'})
    