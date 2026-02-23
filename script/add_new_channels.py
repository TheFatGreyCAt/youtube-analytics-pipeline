import sys
sys.path.insert(0, '.')

from extract.db_manager import PostgresManager

channels_to_add = [
    ("UCUHW94eEFW7hkUMVaZz4eDg", "StatQuest with Josh Starmer"),
    ("UCh9nVJoWXmFb7sLApWGcLPQ", "Codebasics"),
    ("UCLLw7jmFsvfIVaUFsLs8mlQ", "Luke Barousse"),
    ("UC2UXDak6o7rBm23k3Vv5dww", "Tina Huang"),
    ("UCiT9RITQ9PW6BhXK0y2jaeg", "Ken Jee"),
    ("UCmGSJVG3mCRXVOP4yZrU1Dw", "DataCamp"),
    ("UCnVzApLJE2ljPZSeQylSEyg", "365 Data Science"),
    ("UCLqEr-xV-ceUL7W1ivs0J1A", "Data Professor"),
    ("UC7cs8q-gJRlGwj4A8OmCmXg", "Alex The Analyst"),
    ("UCW5YeuERMmlnqo4oq8vwUpg", "The Net Ninja"),
    ("UCsBjURrPoezykLs9EqgamOA", "Fireship"),
    ("UC8butISFwT-Wl7EV0hUK0BQ", "freeCodeCamp"),
    ("UCW6TXMZ5Pq6yL6_k5NZ2e0Q", "Lex Fridman"),
    ("UC9-y-6csu5WGm29I7JiwpnA", "Computerphile"),
    ("UCkw4JCwteGrDHIsyIIKo4tQ", "Edureka"),
    ("UCEBb1b_L6zDS3xTUrIALZOw", "MIT OpenCourseWare"),
    ("UCfzlCWGWYyIQ0aLC5w48gBQ", "Sentdex"),
    ("UCYO_jab_esuFRV4b17AJtAw", "3Blue1Brown"),
    ("UCxX9wt5FWQUAAz4UrysqK9A", "CS Dojo"),
    ("UCsvqVGtbbyHaMoevxPAq9Fg", "Tech With Tim"),
    ("UC8i9mqTTjg0F0CxMTbI4X7g", "Krish Naik"),
    ("UCCezIgC97PvUuR4_gbFUs5g", "Corey Schafer"),
    ("UC7T2-sdZJD6NlXS1tF1bZWg", "Egor Howell"),
    ("UCkQX1tChV7lrewriPhf2g1Q", "Network Chuck"),
    ("UCJ24N4O0bP7LGLBDvye7oCA", "Matt D'Avella"),
    ("UCO1cgjhGzsSYb1rsB4bFe4Q", "Fun Fun Function"),
    ("UCqrILQNl5Ed9Dz6CGMyvMTQ", "Simply Explained"),
    ("UCvjgXvBlbQiydffxwpqEJQA", "Computerphile"),
    ("UCm_hOyxLP_G7KfXS6IAlI2w", "Khan Academy"),
    ("UCP7WmQ_U4GB3K51Od9QvM0w", "Google Developers"),
    ("UCaLlzGqiPE0QRj6sEGPCvNQ", "Microsoft Developer"),
    ("UCRPMAqdtSgd0Ipeef7iFsKw", "IBM Technology"),
    ("UCXgGY0wkgOzynnHvSEVmE3A", "Two Minute Papers"),
    ("UC-lHJZR3Gqxm24_Vd_AJ5Yw", "PewDiePie"),
    ("UCsT0YIqwnpJCM-mx7-gSA4Q", "TEDEd"),
    ("UC-9-kyTW8ZkZNDHQJ6FgpwQ", "Music"),
    ("UCbfYPyITQ-7l4upoX8nvctg", "2 Minute Papers"),
    ("UC0RhatS1pyxInC00YKjjBqQ", "Numberphile"),
    ("UCJbPGzawDH1njbqV-D5HqKw", "Nick White"),
    ("UCiMhD4jzUqG-IgPzUmmytRQ", "NeetCode"),
    ("UCW0gH2G-cMKAEjEkI4YhnPA", "Python Programmer"),
    ("UCZCFT11CWBi3MHNlGf019nw", "Abdul Bari"),
    ("UClEEsT7DkdVO_fkrBw0OTrA", "Geek's Lesson"),
    ("UCeVMnSShP_Iviwkknt83cww", "Code with Harry"),
    ("UC8n8ftV94ZU_DJLOLtrpORA", "Web Dev Simplified"),
    ("UC29ju8bIPH5as8OGnQzwJyA", "Traversy Media"),
]

db = PostgresManager()
added = 0
failed = 0

print(f"Adding {len(channels_to_add)} channels to database...")

for channel_id, channel_name in channels_to_add:
    try:
        db.add_channel(channel_id, channel_name)
        print(f"Added: {channel_name}")
        added += 1
    except Exception as e:
        print(f"Failed to add {channel_name}: {e}")
        failed += 1

print(f"\nSummary: Added {added}, Failed {failed}")
