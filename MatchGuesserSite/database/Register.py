from werkzeug.security import generate_password_hash
from MatchGuesserSite.database.db import get_db_connection

def register_user(name, password):
    """
    Registers a new user with a hashed password into the database.
    """
    # Hash the password before storing it
    hashed_password = generate_password_hash(password)
    print(hashed_password)
    try:
        # Step 1: Establish database connection using get_db_connection()
        conn = get_db_connection()
        cursor = conn.cursor()

        # Step 2: Insert the new user into the user_info table
        cursor.execute("""
            INSERT INTO user_info (name, pass)
            VALUES (%s, %s);
        """, (name, hashed_password))

        # Step 3: Commit the transaction
        conn.commit()

        # Check if the user was inserted successfully
        print(f"User {name} registered successfully.")
        return True

    except Exception as e:
        print(f"Database error: {e}")
        return False

    finally:
        # Close cursor and connection
        cursor.close()
        conn.close()

