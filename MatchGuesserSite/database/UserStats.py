from MatchGuesserSite.database.db import get_db_connection

def increment_correct_guess(username):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            UPDATE user_info
            SET correct_guesses = correct_guesses + 1
            WHERE name = %s
        ''', (username,))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f'Error incrementing correct guesses: {e}')
        return False

def increment_incorrect_guess(username):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            UPDATE user_info
            SET incorrect_guesses = incorrect_guesses + 1
            WHERE name = %s
        ''', (username,))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f'Error incrementing incorrect guesses: {e}')
        return False

def get_user_counters(username):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            SELECT correct_guesses, incorrect_guesses
            FROM user_info
            WHERE name = %s
        ''', (username,))
        result = cur.fetchone()
        cur.close()
        conn.close()
        if result:
            correct_guesses, incorrect_guesses = result
            return {
                'correct_guesses': correct_guesses,
                'incorrect_guesses': incorrect_guesses
            }
        else:
            return None
    except Exception as e:
        print(f'Error fetching user counters: {e}')
        return None
