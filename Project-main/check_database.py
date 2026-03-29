import sqlite3
import pickle
import os
import json

print("\n===== CHECKING DATABASES =====\n")

# Check SQLite Database
print("1. SQLite Database (organization.db):")
if os.path.exists('organization.db'):
    conn = sqlite3.connect('organization.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, name, employee_id, face_encoding FROM persons')
    persons = cursor.fetchall()
    print(f"   Found {len(persons)} registered persons:")
    for p in persons:
        encoding = json.loads(p[3]) if p[3] else []
        print(f"   - ID: {p[0]}, Name: {p[1]}, Employee: {p[2]}, Face Label: {encoding}")
    conn.close()
else:
    print("   Database file NOT FOUND!")

print()

# Check Face Recognition Database
print("2. Face Recognition Database (face_database.pkl):")
if os.path.exists('face_database.pkl'):
    with open('face_database.pkl', 'rb') as f:
        data = pickle.load(f)
        db = data['database']
        print(f"   Found {len(db)} persons in face recognizer:")
        for label, info in db.items():
            print(f"   - Label: {label}, Name: {info['name']}, Employee: {info['employee_id']}")
            print(f"     Features: {len(info['features'])} sets")
else:
    print("   Face database file NOT FOUND!")

print("\n==============================\n")
