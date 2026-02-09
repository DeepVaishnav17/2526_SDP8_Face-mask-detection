import os
import sqlite3

print("Cleaning up old databases...")

# Remove old databases
if os.path.exists('organization.db'):
    os.remove('organization.db')
    print("✓ Removed organization.db")

if os.path.exists('face_database.pkl'):
    os.remove('face_database.pkl')
    print("✓ Removed face_database.pkl")

print("\nDatabases cleared. Restart the server to create fresh databases.")
