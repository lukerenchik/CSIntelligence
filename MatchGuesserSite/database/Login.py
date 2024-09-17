from werkzeug.security import check_password_hash
from MatchGuesserSite.database.db import get_db_connection

def login_user(name, password):
    """
    Authenticates a user by checking the username and password against the database.
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Fetch the user's hashed password from the database
        cursor.execute("SELECT pass FROM user_info WHERE name = %s", (name,))
        user = cursor.fetchone()

        if user is None:
            return False, "User does not exist"

        hashed_password = user[0]

        # Verify the password
        if check_password_hash(hashed_password, password):
            return True, "Login successful"
        else:
            return False, "Invalid password"

    except Exception as e:
        print(f"Database error: {e}")
        return False, str(e)

    finally:
        cursor.close()
        conn.close()
