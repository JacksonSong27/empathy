# run.py
import sys
import os

# Add the 'src' directory to the system path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from app import create_app  # Now this points to src/app/__init__.py

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
