from flask import Flask
from flask_cors import CORS
from app.config import ensure_dirs
from app.api.routes import bp as api_bp

def create_app():
    ensure_dirs()
    app = Flask(__name__)
    CORS(app, resources={r"*": {"origins": "*"}})
    app.register_blueprint(api_bp)
    return app