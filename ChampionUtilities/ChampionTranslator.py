import re

def normalize_name(champ_name):
    # Convert to lowercase
    champ_name = champ_name.lower()
    # Remove apostrophes and other special characters (keep only alphanumeric characters)
    champ_name = re.sub(r"[^a-z0-9]+", "", champ_name)
    return champ_name


class ChampionTranslator:
    def __init__(self):
        # Provided champion mapping dictionary
        self.champion_mapping = {
            1: "Aatrox", 2: "Ahri", 3: "Akali", 4: "Akshan", 5: "Alistar",
            6: "Amumu", 7: "Anivia", 8: "Annie", 9: "Aphelios", 10: "Ashe",
            11: "Aurelion Sol", 12: "Aurora", 13: "Azir", 14: "Bard", 15: "Bel'Veth",
            16: "Blitzcrank", 17: "Brand", 18: "Braum", 19: "Briar", 20: "Caitlyn",
            21: "Camille", 22: "Cassiopeia", 23: "Cho'Gath", 24: "Corki", 25: "Darius",
            26: "Diana", 27: "Dr. Mundo", 28: "Draven", 29: "Ekko", 30: "Elise",
            31: "Evelynn", 32: "Ezreal", 33: "Fiddlesticks", 34: "Fiora", 35: "Fizz",
            36: "Galio", 37: "Gangplank", 38: "Garen", 39: "Gnar", 40: "Gragas",
            41: "Graves", 42: "Gwen", 43: "Hecarim", 44: "Heimerdinger", 45: "Hwei",
            46: "Illaoi", 47: "Irelia", 48: "Ivern", 49: "Janna", 50: "Jarvan IV",
            51: "Jax", 52: "Jayce", 53: "Jhin", 54: "Jinx", 55: "Kai'Sa",
            56: "Kalista", 57: "Karma", 58: "Karthus", 59: "Kassadin", 60: "Katarina",
            61: "Kayle", 62: "Kayn", 63: "Kennen", 64: "Kha'Zix", 65: "Kindred",
            66: "Kled", 67: "Kog'Maw", 68: "K'Sante", 69: "Leblanc", 70: "Lee Sin",
            71: "Leona", 72: "Lillia", 73: "Lissandra", 74: "Lucian", 75: "Lulu",
            76: "Lux", 77: "Malphite", 78: "Malzahar", 79: "Maokai", 80: "Master Yi",
            81: "Milio", 82: "Miss Fortune", 83: "Mordekaiser", 84: "Morgana",
            85: "Naafiri", 86: "Nami", 87: "Nasus", 88: "Nautilus", 89: "Neeko",
            90: "Nidalee", 91: "Nilah", 92: "Nocturne", 93: "Nunu", 94: "Olaf",
            95: "Orianna", 96: "Ornn", 97: "Pantheon", 98: "Poppy", 99: "Pyke",
            100: "Qiyana", 101: "Quinn", 102: "Rakan", 103: "Rammus", 104: "Rek'Sai",
            105: "Rell", 106: "Renata Glasc", 107: "Renekton", 108: "Rengar", 109: "Riven",
            110: "Rumble", 111: "Ryze", 112: "Samira", 113: "Sejuani", 114: "Senna",
            115: "Seraphine", 116: "Sett", 117: "Shaco", 118: "Shen", 119: "Shyvana",
            120: "Singed", 121: "Sion", 122: "Sivir", 123: "Skarner", 124: "Smolder",
            125: "Sona", 126: "Soraka", 127: "Swain", 128: "Sylas", 129: "Syndra",
            130: "Tahm Kench", 131: "Taliyah", 132: "Talon", 133: "Taric", 134: "Teemo",
            135: "Thresh", 136: "Tristana", 137: "Trundle", 138: "Tryndamere", 139: "Twisted Fate",
            140: "Twitch", 141: "Udyr", 142: "Urgot", 143: "Varus", 144: "Vayne",
            145: "Veigar", 146: "Vel'Koz", 147: "Vex", 148: "Vi", 149: "Viego",
            150: "Viktor", 151: "Vladimir", 152: "Volibear", 153: "Warwick", 154: "Wukong",
            155: "Xayah", 156: "Xerath", 157: "Xin Zhao", 158: "Yasuo", 159: "Yone",
            160: "Yorick", 161: "Yuumi", 162: "Zac", 163: "Zed", 164: "Zeri",
            165: "Ziggs", 166: "Zilean", 167: "Zoe", 168: "Zyra"
        }

        # Normalize the champion names and store in a new dictionary
        self.champion_mapping = {normalize_name(name): champ_id for champ_id, name in self.champion_mapping.items()}

    # Method to get champion name by ID
    def get_champion_id(self, champ_name):
        # Normalize the input champion name before looking it up
        normalized_name = normalize_name(champ_name)
        return self.champion_mapping.get(normalized_name, None)  # Return None if not found

    # Method to get champion name by ID (if needed)
    def get_champion_name(self, champ_id):
        # Reverse lookup: find the key corresponding to the champion ID
        for key, value in self.champion_mapping.items():
            if value == champ_id:
                return key
        return "Unknown"  # Return "Unknown" if the champion ID is not found