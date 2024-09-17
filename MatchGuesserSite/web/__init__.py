from flask import Blueprint, render_template

web = Blueprint('web', __name__)

@web.route('/')
def index():
    # Simply render the index page (no user login checks)
    return render_template('matchguesser.html')

@web.route('/register')
def register():
    # Render the register page (no user-related functionality)
    return render_template('register.html')

@web.route('/matchguesser')
def matchguesser():
    # Always render the matchguesser page (no user login checks)
    return render_template('matchguesser.html')

@web.route('/login')
def login():
    return render_template('login.html')

@web.route('/leaderboard')
def leaderboard():
    return render_template('leaderboard.html')
