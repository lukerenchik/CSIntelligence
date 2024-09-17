import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from ChampionUtilities.ChampionTranslator import ChampionTranslator
import re


# Function to load and process the matches
def load_and_process_matches(csv_file):
    # Load the dataset into a pandas DataFrame
    data = pd.read_csv(csv_file)

    # Ensure the data is sorted by match ID and team ID
    data.sort_values(by=['match_matchId', 'player_teamId'], inplace=True)

    # Group by match ID to bundle the players for each match
    grouped_matches = data.groupby('match_matchId')

    # List to hold the results for each match
    matches = []

    # Iterate over each match group
    for match_id, match_data in grouped_matches:
        # Split the data into blue and red teams based on 'player_teamId'
        blue_team_data = match_data[match_data['player_teamId'] == 'blue']
        red_team_data = match_data[match_data['player_teamId'] == 'red']

        # Ensure each team has exactly 5 players
        if len(blue_team_data) == 5 and len(red_team_data) == 5:
            # Extract the champion names for both teams
            blue_team_champs = blue_team_data['player_champName'].tolist()
            red_team_champs = red_team_data['player_champName'].tolist()

            # Extract the win label for the blue team (1 if blue team won, 0 if red team won)
            blue_team_win = blue_team_data['player_win'].iloc[0]
            label = 1 if blue_team_win == 1 else 0

            # Append the result as a tuple (blue team, red team, label)
            matches.append((blue_team_champs, red_team_champs, label))

    return matches


def get_unique_champions(matches):
    champions = set()
    for blue_team, red_team, _ in matches:
        champions.update(blue_team)
        champions.update(red_team)
    return list(champions)


def convert_matches_to_ids(matches, champion_to_id, champion_attributes, translator):
    matches_ids = []
    for blue_team, red_team, label in matches:
        # Normalize champion names when accessing champion_to_id
        blue_team_ids = [champion_to_id[normalize_name(champ)] for champ in blue_team]
        red_team_ids = [champion_to_id[normalize_name(champ)] for champ in red_team]

        # Handle missing champion attributes by using default attributes if not found
        default_attributes = [0] * 8  # Assuming 8 attributes per champion

        def get_champion_attrs(champ):
            champ_id = translator.get_champion_id(champ)
            attrs = champion_attributes.get(champ_id, default_attributes)
            return list(attrs.values()) if isinstance(attrs, dict) else attrs

        # Retrieve champion attributes as a list of values
        blue_team_attrs = [get_champion_attrs(champ) for champ in blue_team]
        red_team_attrs = [get_champion_attrs(champ) for champ in red_team]

        matches_ids.append((blue_team_ids, red_team_ids, blue_team_attrs, red_team_attrs, label))
    return matches_ids


def normalize_name(champ_name):
    # Convert to lowercase
    champ_name = champ_name.lower()
    # Remove apostrophes and other special characters (keep only alphanumeric characters)
    champ_name = re.sub(r"[^a-z0-9]+", "", champ_name)
    return champ_name


class LoLMatchDataset(Dataset):
    def __init__(self, matches):
        self.matches = matches

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        blue_team_ids, red_team_ids, blue_team_attrs, red_team_attrs, label = self.matches[idx]
        blue_team_tensor = torch.tensor(blue_team_ids, dtype=torch.long)
        red_team_tensor = torch.tensor(red_team_ids, dtype=torch.long)
        blue_team_attrs_tensor = torch.tensor(blue_team_attrs, dtype=torch.float)
        red_team_attrs_tensor = torch.tensor(red_team_attrs, dtype=torch.float)
        label_tensor = torch.tensor(label, dtype=torch.float)
        return blue_team_tensor, red_team_tensor, blue_team_attrs_tensor, red_team_attrs_tensor, label_tensor


class LoLMatchPredictor(nn.Module):
    def __init__(self, num_champions, embedding_dim, attribute_dim):
        super(LoLMatchPredictor, self).__init__()

        # Embedding layer for champions
        self.embedding = nn.Embedding(num_champions, embedding_dim)

        # The input dimension will include the embeddings and the attributes
        input_dim = (embedding_dim + attribute_dim) * 2  # Times 2 because we have two teams

        # Define 5 fully connected layers
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)  # Batch normalization after the first fully connected layer
        self.dropout1 = nn.Dropout(0.5)  # Dropout with probability 0.5

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)  # Batch normalization after the second fully connected layer
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)  # Batch normalization after the third fully connected layer
        self.dropout3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)  # Batch normalization after the fourth fully connected layer
        self.dropout4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(64, 1)  # Output layer

    def forward(self, team_a, team_b, team_a_attrs, team_b_attrs):
        # Get embeddings for both teams
        emb_a = self.embedding(team_a)  # Shape: [batch_size, 5, embedding_dim]
        emb_b = self.embedding(team_b)  # Shape: [batch_size, 5, embedding_dim]

        # Aggregate embeddings by averaging across the champions in each team
        team_a_vec = torch.mean(emb_a, dim=1)  # Shape: [batch_size, embedding_dim]
        team_b_vec = torch.mean(emb_b, dim=1)  # Shape: [batch_size, embedding_dim]

        # Concatenate the embedding with the attributes for each team
        team_a_combined = torch.cat([team_a_vec, team_a_attrs], dim=1)
        team_b_combined = torch.cat([team_b_vec, team_b_attrs], dim=1)

        # Concatenate the two teams' combined embeddings and attributes
        x = torch.cat([team_a_combined, team_b_combined],
                      dim=1)  # Shape: [batch_size, (embedding_dim + attribute_dim) * 2]

        # First fully connected layer with batch normalization and dropout
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)

        # Second fully connected layer with batch normalization and dropout
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)

        # Third fully connected layer with batch normalization and dropout
        x = torch.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout3(x)

        # Fourth fully connected layer with batch normalization and dropout
        x = torch.relu(self.fc4(x))
        x = self.bn4(x)
        x = self.dropout4(x)

        # Final output layer with sigmoid activation for binary classification
        x = torch.sigmoid(self.fc5(x))

        return x.squeeze()


# Load the matches and champion attributes
csv_file = '/home/lightbringer/Documents/Dev/ChampionSelectIntelligence/Data/lol_match_data.csv'
matches = load_and_process_matches(csv_file)

# Load champion attributes (assuming you have a CSV or another source for the attributes)
champion_attributes = pd.read_csv(
    '/home/lightbringer/Documents/Dev/ChampionSelectIntelligence/Data/LoL-Champions.csv').set_index('Id').T.to_dict()

# Create the ChampionTranslator instance
translator = ChampionTranslator()

# Split into training and testing sets (80% train, 20% test)
train_matches, test_matches = train_test_split(matches, test_size=0.2, random_state=42)

unique_champions = get_unique_champions(matches)

# Normalize champion names while creating the champion_to_id mapping
champion_to_id = {normalize_name(champion): idx for idx, champion in enumerate(unique_champions)}

# Inverse mapping from ID to original (non-normalized) champion names
id_to_champion = {idx: champion for idx, champion in enumerate(unique_champions)}

# Count the total number of unique champions
num_champions = len(unique_champions)


# Convert matches to IDs and include attributes
train_matches_ids = convert_matches_to_ids(train_matches, champion_to_id, champion_attributes, translator)
test_matches_ids = convert_matches_to_ids(test_matches, champion_to_id, champion_attributes, translator)

# Create datasets
train_dataset = LoLMatchDataset(train_matches_ids)
test_dataset = LoLMatchDataset(test_matches_ids)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
embedding_dim = 100
attribute_dim = 8  # Number of attributes for each champion (assuming 8 attributes)
model = LoLMatchPredictor(num_champions, embedding_dim, attribute_dim)

# Training Loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for blue_team_batch, red_team_batch, blue_team_attrs_batch, red_team_attrs_batch, labels in train_loader:
        blue_team_batch = blue_team_batch.to(device)
        red_team_batch = red_team_batch.to(device)
        blue_team_attrs_batch = blue_team_attrs_batch.to(device)
        red_team_attrs_batch = red_team_attrs_batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(blue_team_batch, red_team_batch, blue_team_attrs_batch, red_team_attrs_batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    # Validation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for blue_team_batch, red_team_batch, blue_team_attrs_batch, red_team_attrs_batch, labels in test_loader:
            blue_team_batch = blue_team_batch.to(device)
            red_team_batch = red_team_batch.to(device)
            blue_team_attrs_batch = blue_team_attrs_batch.to(device)
            red_team_attrs_batch = red_team_attrs_batch.to(device)
            labels = labels.to(device)

            outputs = model(blue_team_batch, red_team_batch, blue_team_attrs_batch, red_team_attrs_batch)
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')

print("Training complete.")
