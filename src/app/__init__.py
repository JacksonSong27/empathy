import os
from flask import Flask
from src.app.routes import register_routes

def create_app():
    # Get the path to the folder containing this file
    base_dir = os.path.abspath(os.path.dirname(__file__))

    # Create the Flask app with correct template and static paths
    app = Flask(
        __name__,
        template_folder=os.path.join(base_dir, 'templates'),
        static_folder=os.path.abspath(os.path.join(base_dir, '..', '..', 'static'))
    )

    # Register all routes
    register_routes(app)

    return app
