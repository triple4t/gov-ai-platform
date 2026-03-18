import sqlite3
import json
from datetime import datetime
from app.core.config import settings

def init_db():
    """Initializes the SQLite database and creates the necessary tables."""
    conn = sqlite3.connect(settings.OCR_DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            raw_text TEXT,
            name TEXT,
            id_number TEXT,
            date_of_birth TEXT,
            address TEXT,
            phone_number TEXT,
            other_fields TEXT,
            processed_at TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def insert_document(filename: str, raw_text: str, extracted_data: dict) -> int:
    """Inserts a processed document into the database."""
    conn = sqlite3.connect(settings.OCR_DB_FILE)
    cursor = conn.cursor()
    
    other_fields = extracted_data.get("other_fields", {})
    other_fields_json = json.dumps(other_fields) if other_fields else "{}"
    
    processed_at = datetime.utcnow().isoformat()
    
    cursor.execute('''
        INSERT INTO documents (filename, raw_text, name, id_number, date_of_birth, address, phone_number, other_fields, processed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        filename,
        raw_text,
        extracted_data.get("name"),
        extracted_data.get("id_number"),
        extracted_data.get("date_of_birth"),
        extracted_data.get("address"),
        extracted_data.get("phone_number"),
        other_fields_json,
        processed_at
    ))
    
    doc_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return doc_id

def get_all_documents():
    """Retrieves all processed documents."""
    conn = sqlite3.connect(settings.OCR_DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM documents ORDER BY processed_at DESC')
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]
