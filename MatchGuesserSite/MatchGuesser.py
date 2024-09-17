from flask import Flask
from flask_cors import CORS
from api import api
from web import web
import os
from MatchGuesserSite.database.MatchFetcher import fetch_random_match

match = fetch_random_match()
print(match)


app = Flask(__name__)
app.secret_key = os.urandom(24)  # Replace with a random secret key for security
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5000"}}, supports_credentials=True)

app.register_blueprint(api, url_prefix='/api')
app.register_blueprint(web)

if __name__ == '__main__':
    app.run(debug=True)
