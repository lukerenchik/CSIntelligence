import pandas as pd
import psycopg2
from db import get_db_connection  # Import the connection function from db.py

import pandas as pd
from db import get_db_connection  # Assuming db.py contains the logic for getting the DB connection


class MatchLoader:
    def __init__(self, data_filepath, features):
        """
        Initialize the MatchLoader with a dataset and features to extract.

        :param data_filepath: CSV filepath containing the match data.
        :param features: List of column names to extract as features.
        """
        self.data_filepath = data_filepath
        self.features = features
        self.matches_df = self._csv_to_dataframe()

    def _csv_to_dataframe(self):
        """
        Convert the CSV match data to a Pandas DataFrame if needed.
        :return: A Pandas DataFrame containing the feature data.
        """
        try:
            df = pd.read_csv(self.data_filepath)
            return df[self.features]
        except FileNotFoundError:
            print(f"File {self.data_filepath} not found.")
            return pd.DataFrame()

    def load_matches_to_db(self):
        """
        Add the matches to the 'matches' and 'players' tables within the MatchGuesser database.

        :return: A status message indicating whether the matches were successfully added to the database.
        """
        if self.matches_df.empty:
            return "No match data to load."

        # Get the database connection
        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            # Step 1: Insert data into the matches table (ensure no duplicate matches)
            match_ids = set()
            for index, row in self.matches_df.iterrows():
                match_id = row['match_matchId']  # Assuming match_matchId is now a VARCHAR(50)
                match_ids.add(match_id)

            for match_id in match_ids:
                cursor.execute("""
                    INSERT INTO matches (match_matchId)
                    VALUES (%s)
                    ON CONFLICT (match_matchId) DO NOTHING;
                """, (match_id,))

            # Step 2: Insert data into the players table for each match
            for index, row in self.matches_df.iterrows():
                cursor.execute("""
                    INSERT INTO players (match_matchId, player_teamId, player_teamPosition, player_lane, 
                                         player_champName, player_banPickTurn, player_champName_ban, player_win)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING;
                """, (
                    row['match_matchId'], row['player_teamId'], row['player_teamPosition'],
                    row['player_lane'], row['player_champName'], row['player_banPickTurn'],
                    row['player_champName_ban'], row['player_win']
                ))

            # Commit the transaction to save changes
            conn.commit()

            return f"Successfully added matches and players to the database."

        except Exception as e:
            conn.rollback()  # Rollback in case of error
            return f"Error loading matches: {e}"

        finally:
            cursor.close()
            conn.close()


# Load match data and initialize match loader
data_filepath = '/home/lightbringer/Documents/Dev/ChampionSelectIntelligence/Data/lol_match_data.csv'
features = [
    'match_matchId', 'player_teamId', 'player_teamPosition', 'player_lane',
    'player_champName', 'player_banPickTurn', 'player_champName_ban', 'player_win'
]

match_loader = MatchLoader(data_filepath, features)
status_message = match_loader.load_matches_to_db()
print(status_message)
