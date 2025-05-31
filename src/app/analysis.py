from datetime import datetime
import json
from src.app.config import client
from src.app.data_store import dialogue_data

# Reference to update this in-place
from src.app.data_store import cumulative_total

# Prompt template
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

def estimate_empathy(text):
    global cumulative_total

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing empathy in conversations."},
                {"role": "user", "content": prompt.format(text=text)}
            ],
            temperature=0.3,
            response_format={ "type": "json_object" }
        )

        result = json.loads(response.choices[0].message.content)

        # Scores
        raw_total = (
            result['conversation_score'] +
            result['movement_score'] +
            result['tone_score'] +
            result['listening_score']
        )

        cumulative_total += raw_total

        if cumulative_total < 0.2:
            increment = 0.0
        elif cumulative_total < 0.8:
            increment = 0.1
        else:
            increment = 0.2

        final_score = raw_total + increment

        result['raw_total'] = raw_total
        result['cumulative_score'] = cumulative_total
        result['increment'] = increment
        result['final_score'] = final_score

        dialogue_data['analysis_history'].append({
            'text': text,
            'scores': result,
            'timestamp': datetime.now().isoformat()
        })

        return final_score

    except Exception as e:
        print(f"Error in GPT empathy scoring: {e}")
        return 0.3


def analyze_emotions(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an emotion analyzer. Analyze the emotional content of the following text."},
                {"role": "user", "content": text}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in emotion analysis: {e}")
        return "Could not analyze emotions."
