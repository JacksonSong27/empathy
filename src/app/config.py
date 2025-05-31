import os
from openai import OpenAI

# Initialize OpenAI client using environment variable
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
