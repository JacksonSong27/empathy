# src/app/word_utils.py

def analyze_empathy_words(conversation_history):
    EMPATHY_WORD_LIST = {
        'Help': 'Noun', 'Support': 'Noun', 'Care': 'Noun',
        'Calm': 'Adjective', 'Safe': 'Adjective', 'Kind': 'Adjective',
        'Supportive': 'Adjective', 'Gentle': 'Adjective',
        'Support': 'Verb', 'Assist': 'Verb', 'Listen': 'Verb',
        "You're not alone": 'Phrase', "Take a breath": 'Phrase'
    }

    try:
        conv_lower = conversation_history.lower()
        word_frequencies = {}
        word_details = {}

        for word, word_type in EMPATHY_WORD_LIST.items():
            word_lower = word.lower()
            count = conv_lower.count(word_lower)

            if count > 0:
                contexts = []
                start = 0
                for _ in range(count):
                    pos = conv_lower.find(word_lower, start)
                    if pos != -1:
                        context_start = max(0, pos - 50)
                        context_end = min(len(conversation_history), pos + len(word) + 50)
                        context = conversation_history[context_start:context_end].strip()
                        contexts.append(context)
                        start = pos + 1

                word_frequencies[word] = count
                word_details[word] = {
                    "frequency": count,
                    "type": word_type,
                    "context": contexts
                }

        return word_frequencies, word_details

    except Exception as e:
        print(f"Error in empathy word analysis: {e}")
        return {}, {}
