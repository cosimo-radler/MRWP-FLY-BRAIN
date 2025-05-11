from flask import Flask, request
import os
import sys

# Add the parent directory to the path so we can import the dashboard module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.visualization.dashboard import app as dash_app

# Get the Flask server that powers the Dash app
server = dash_app.server

# This is for Vercel serverless function
app = server 