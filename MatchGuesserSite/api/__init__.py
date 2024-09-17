from flask import Blueprint, jsonify, request, session
from MatchGuesserSite.database.MatchFetcher import fetch_random_match
from MatchGuesserSite.database.Register import register_user
from MatchGuesserSite.database.Login import login_user
from MatchGuesserSite.database.UserStats import increment_correct_guess, increment_incorrect_guess, get_user_counters
from MatchGuesserSite.database.Leaderboard import get_top_users
api = Blueprint('api', __name__)


@api.route('/rand_match', methods=['GET'])
def get_rand_match():
    match = fetch_random_match()

    if not match:
        return jsonify({"error": "No match found"}), 404

    match_matchId = match[0]  # The first element is the match ID
    player_data_list = match[1]  # The second element is the list of players

    # Convert player data to a list of dictionaries
    match_data = [
        {
            'match_matchId': match_matchId,  # Use the match ID for all players
            'player_teamId': row[2],  # Adjusted index for team ID
            'player_teamPosition': row[3],
            'player_lane': row[4],
            'player_champName': row[5],
            'player_banPickTurn': row[6],
            'player_champName_ban': row[7],
            'player_win': row[8],
        }
        for row in player_data_list
    ]

    return jsonify(match_data), 200


@api.route('/register', methods=['POST'])
def register():
    """
    API route to register a new user.
    It extracts the user details from the request and calls the register_user function.
    """
    try:
        # Extract user details from the POST request
        data = request.json  # Ensure this is JSON
        name = data.get('name')
        password = data.get('pass')

        # Check if name or password is missing
        if not name or not password:
            return jsonify({"error": "Username and password are required"}), 400

        # Call the registration function
        registration_successful = register_user(name, password)

        if registration_successful:
            return jsonify({"message": "User registered successfully"}), 201
        else:
            return jsonify({"error": "Failed to register user"}), 500

    except Exception as e:
        # Log the error for debugging
        print(f"Error in registration API: {e}")
        return jsonify({"error": "An internal error occurred"}), 500


@api.route('/login', methods=['POST'])
def login():
    """
    API route for user login.
    It checks the username and password and starts a session for the user.
    """
    data = request.json
    name = data.get('name')
    password = data.get('pass')

    if not name or not password:
        return jsonify({"error": "Username and password are required"}), 400

    # Call the login_user function
    success, message = login_user(name, password)

    if success:
        session['user'] = name  # Track the logged-in user in the session
        return jsonify({"message": message}), 200
    else:
        return jsonify({"error": message}), 401


@api.route('/logout', methods=['POST'])
def logout():
    session.pop('user', None)  # Remove the user from the session
    return jsonify({"message": "Logged out successfully"}), 200


@api.route('/increment_correct', methods=['POST'])
def increment_correct():
    if 'user' not in session:
        return jsonify({'error': 'User not logged in'}), 401

    username = session['user']

    success = increment_correct_guess(username)
    if success:
        return jsonify({'message': 'Correct guesses incremented'}), 200
    else:
        return jsonify({'error': 'An internal error occurred'}), 500

@api.route('/increment_incorrect', methods=['POST'])
def increment_incorrect():
    if 'user' not in session:
        return jsonify({'error': 'User not logged in'}), 401

    username = session['user']

    success = increment_incorrect_guess(username)
    if success:
        return jsonify({'message': 'Incorrect guesses incremented'}), 200
    else:
        return jsonify({'error': 'An internal error occurred'}), 500

@api.route('/get_user_counters', methods=['GET'])
def get_user_counters_route():
    if 'user' not in session:
        return jsonify({'error': 'User not logged in'}), 401

    username = session['user']

    counters = get_user_counters(username)
    if counters is not None:
        return jsonify(counters), 200
    else:
        return jsonify({'error': 'An internal error occurred'}), 500


@api.route('/leaderboard', methods=['GET'])
def leaderboard():
    try:
        leaderboard_data = get_top_users(limit=10)
        if leaderboard_data is not None:
            return jsonify(leaderboard_data), 200
        else:
            return jsonify({'error': 'Failed to fetch leaderboard data'}), 500
    except Exception as e:
        print(f'Error in leaderboard API route: {e}')
        return jsonify({'error': 'An internal error occurred'}), 500

