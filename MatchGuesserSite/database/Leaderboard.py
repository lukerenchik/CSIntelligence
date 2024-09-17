from MatchGuesserSite.database.db import get_db_connection

def get_top_users(limit=10):
    """
    Retrieves the top users with the best ratio of correct guesses to incorrect guesses.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Calculate the accuracy percentage, handle division by zero
        query = '''
        SELECT name, correct_guesses, incorrect_guesses,
        CASE
            WHEN (correct_guesses + incorrect_guesses) = 0 THEN 0
            ELSE (correct_guesses::float / (correct_guesses + incorrect_guesses)) * 100
        END as accuracy_percentage
        FROM user_info
        ORDER BY accuracy_percentage DESC, correct_guesses DESC
        LIMIT %s;
        '''

        cur.execute(query, (limit,))
        results = cur.fetchall()
        cur.close()
        conn.close()

        # Format the results into a list of dictionaries
        leaderboard = []
        for row in results:
            user = {
                'name': row[0],
                'correct_guesses': row[1],
                'incorrect_guesses': row[2],
                'accuracy_percentage': round(row[3], 2)
            }
            leaderboard.append(user)

        return leaderboard

    except Exception as e:
        print(f'Error fetching leaderboard data: {e}')
        return None