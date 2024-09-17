# db.py
import psycopg2
import os


def get_db_connection():

    """
    Establishes a connection to the PostgreSQL database and returns the connection object.
    """
    db_config = {
        'dbname': 'matchguesser',
        'user': 'luke_dev',
        'password': os.getenv('PGPASSWORD'),
        'host': 'localhost',
        'port': '5432'
    }

    try:
        conn = psycopg2.connect(
            dbname=db_config['dbname'],
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        )
        return conn
    except psycopg2.DatabaseError as e:
        print(f"Error connecting to database: {e}")
        return None
