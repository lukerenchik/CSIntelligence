import random
from flask import session
import psycopg2
from MatchGuesserSite.database.db import get_db_connection


def fetch_random_unseen_match():
    """
    Fetch a random match that the logged-in user has not yet evaluated using psycopg2.

    This function checks the matches that the user has already evaluated and fetches a random one
    they haven't seen from the 'matches' table.
    """
    # Step 1: Get the current logged-in user's ID from session (using Flask session)
    user_id = session.get('user_id')
    if not user_id:
        return None  # Return None or raise an error indicating that no user is logged in

    # Step 2: Establish database connection
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Step 3: Fetch matches the user has already seen
        cursor.execute("""
            SELECT match_id
            FROM user_matches
            WHERE user_id = %s
        """, (user_id,))

        seen_matches = cursor.fetchall()

        # If the user has seen matches, convert to a list of IDs
        seen_match_ids = [row[0] for row in seen_matches]

        # Step 4: Fetch a random unseen match
        if seen_match_ids:
            # Exclude seen matches and get a random one
            query = """
                SELECT match_matchId
                FROM matches
                WHERE match_matchId NOT IN %s
                ORDER BY RANDOM()
                LIMIT 1;
            """
            cursor.execute(query, (tuple(seen_match_ids),))
        else:
            # If the user hasn't seen any matches, just get any random match
            cursor.execute("""
                SELECT match_matchId
                FROM matches
                ORDER BY RANDOM()
                LIMIT 1;
            """)

        unseen_match = cursor.fetchone()

        # Step 5: Return the result
        if unseen_match:
            return unseen_match[0]  # Return the match ID
        else:
            return None  # No matches left unseen

    except Exception as e:
        print(f"Error fetching unseen match: {e}")
        return None
    finally:
        # Close cursor and connection
        cursor.close()
        conn.close()


def fetch_random_match():
    """
    Fetch a random match from the 'matches' table and return all player entries
    from the 'players' table that correspond to that match.

    This function fetches a random match ID from the 'matches' table and retrieves
    all players from the 'players' table with the same match ID. It ensures that
    each player is only included once by filtering out duplicates.
    """
    # Step 1: Establish database connection
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Step 2: Fetch a random match ID from the matches table
        cursor.execute("""
            SELECT match_matchId
            FROM matches
            ORDER BY RANDOM()
            LIMIT 1;
        """)
        random_match = cursor.fetchone()

        if not random_match:
            return None  # No match found

        random_match_id = random_match[0]

        # Step 3: Fetch unique player entries with the matching match_matchId
        # Ensure that duplicates are removed by using DISTINCT on relevant player fields.
        cursor.execute("""
            SELECT DISTINCT ON (player_id) player_id, match_matchId, player_teamId, player_teamPosition, 
                   player_lane, player_champName, player_banPickTurn, player_champName_ban, player_win
            FROM players
            WHERE match_matchId = %s;
        """, (random_match_id,))

        players_in_match = cursor.fetchall()

        # Return the random match ID and the list of players
        return random_match_id, players_in_match

    except Exception as e:
        print(f"Database error: {e}")
        return None
    finally:
        # Close cursor and connection
        cursor.close()
        conn.close()
