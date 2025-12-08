from flask import Flask
from app.config import ensure_dirs
from app.api.routes import bp as api_bp

def create_app():
    ensure_dirs()
    app = Flask(__name__)
    app.register_blueprint(api_bp)
    return app