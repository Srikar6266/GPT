import sqlite3

DB_PATH = "documents.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS documents
                 (id INTEGER PRIMARY KEY, filename TEXT, text TEXT, summary TEXT)''')
    conn.commit()
    conn.close()