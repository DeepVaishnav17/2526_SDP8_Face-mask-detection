import sqlite3
from datetime import datetime
import json
import os

class OrganizationDatabase:
    """Database manager for organization gate entry system with face recognition"""
    
    def __init__(self, db_name="organization.db"):
        self.db_name = db_name
        self.init_database()
    
    def init_database(self):
        """Initialize database with all required tables"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Table 1: Registered Persons (Employees/Visitors)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                employee_id TEXT UNIQUE,
                department TEXT,
                role TEXT,
                phone TEXT,
                email TEXT,
                photo_path TEXT,
                face_encoding TEXT NOT NULL,
                registration_date TEXT NOT NULL,
                active INTEGER DEFAULT 1
            )
        ''')
        
        # Table 2: Daily Entries (Attendance + Mask Compliance)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                entry_date TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                mask_status TEXT NOT NULL,
                confidence REAL NOT NULL,
                temperature REAL,
                snapshot_path TEXT,
                FOREIGN KEY (person_id) REFERENCES persons(id)
            )
        ''')
        
        # Table 3: Compliance Summary (Daily stats per person)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compliance_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                total_entries INTEGER DEFAULT 0,
                mask_compliant INTEGER DEFAULT 0,
                non_compliant INTEGER DEFAULT 0,
                compliance_rate REAL DEFAULT 0.0,
                FOREIGN KEY (person_id) REFERENCES persons(id),
                UNIQUE(person_id, date)
            )
        ''')
        
        # Table 4: System Settings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
        print("[DB] Database initialized successfully")
    
    def register_person(self, name, employee_id, department, role, phone, email, 
                       photo_path, face_encoding):
        """Register a new person in the system"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        try:
            # Convert face encoding to JSON string for storage
            if hasattr(face_encoding, 'tolist'):
                encoding_str = json.dumps(face_encoding.tolist())
            else:
                encoding_str = json.dumps(face_encoding)
            
            cursor.execute('''
                INSERT INTO persons 
                (name, employee_id, department, role, phone, email, photo_path, 
                 face_encoding, registration_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (name, employee_id, department, role, phone, email, photo_path,
                  encoding_str, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            
            conn.commit()
            person_id = cursor.lastrowid
            print(f"[DB] Registered: {name} (ID: {person_id})")
            return person_id
            
        except sqlite3.IntegrityError:
            print(f"[DB] Error: Employee ID {employee_id} already exists")
            return None
        finally:
            conn.close()
    
    def get_all_persons(self, active_only=True):
        """Get all registered persons"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        if active_only:
            cursor.execute('SELECT * FROM persons WHERE active = 1')
        else:
            cursor.execute('SELECT * FROM persons')
        
        persons = cursor.fetchall()
        conn.close()
        return persons
    
    def get_person_by_id(self, person_id):
        """Get person details by ID"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM persons WHERE id = ?', (person_id,))
        person = cursor.fetchone()
        conn.close()
        return person
    
    def get_person_by_face_label(self, face_label):
        """Get person details by face recognizer label"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # face_encoding stores the face recognizer label as JSON
        cursor.execute('SELECT * FROM persons WHERE active = 1')
        persons = cursor.fetchall()
        conn.close()
        
        for person in persons:
            # person[8] is face_encoding column
            encoding = json.loads(person[8])
            if encoding and len(encoding) > 0 and encoding[0] == face_label:
                return person
        return None
    
    def get_all_face_encodings(self):
        """Get all face encodings for matching"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, name, face_encoding FROM persons WHERE active = 1')
        results = cursor.fetchall()
        conn.close()
        
        encodings = []
        for person_id, name, encoding_str in results:
            encoding = json.loads(encoding_str)
            encodings.append({
                'id': person_id,
                'name': name,
                'encoding': encoding
            })
        
        return encodings
    
    def log_entry(self, person_id, mask_status, confidence, snapshot_path=None, temperature=None):
        """Log a person's entry"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        now = datetime.now()
        entry_date = now.strftime("%Y-%m-%d")
        entry_time = now.strftime("%H:%M:%S")
        
        # Insert entry record
        cursor.execute('''
            INSERT INTO daily_entries 
            (person_id, entry_date, entry_time, mask_status, confidence, 
             temperature, snapshot_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (person_id, entry_date, entry_time, mask_status, confidence,
              temperature, snapshot_path))
        
        # Update compliance summary
        mask_compliant = 1 if mask_status == "DETECTED" else 0
        non_compliant = 1 if mask_status == "NOT_DETECTED" else 0
        
        cursor.execute('''
            INSERT INTO compliance_summary 
            (person_id, date, total_entries, mask_compliant, non_compliant)
            VALUES (?, ?, 1, ?, ?)
            ON CONFLICT(person_id, date) DO UPDATE SET
                total_entries = total_entries + 1,
                mask_compliant = mask_compliant + ?,
                non_compliant = non_compliant + ?,
                compliance_rate = (CAST(mask_compliant AS REAL) / total_entries) * 100
        ''', (person_id, entry_date, mask_compliant, non_compliant,
              mask_compliant, non_compliant))
        
        conn.commit()
        conn.close()
        
        print(f"[DB] Entry logged for person_id={person_id}, mask={mask_status}")
    
    def get_today_entries(self):
        """Get all entries for today"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        cursor.execute('''
            SELECT e.id, p.name, p.employee_id, p.department, 
                   e.entry_time, e.mask_status, e.confidence, e.snapshot_path
            FROM daily_entries e
            JOIN persons p ON e.person_id = p.id
            WHERE e.entry_date = ?
            ORDER BY e.entry_time DESC
        ''', (today,))
        
        entries = cursor.fetchall()
        conn.close()
        return entries
    
    def get_person_entries(self, person_id, days=30):
        """Get recent entries for a specific person"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT entry_date, entry_time, mask_status, confidence
            FROM daily_entries
            WHERE person_id = ?
            ORDER BY entry_date DESC, entry_time DESC
            LIMIT ?
        ''', (person_id, days))
        
        entries = cursor.fetchall()
        conn.close()
        return entries
    
    def get_compliance_report(self, date=None):
        """Get compliance report for a specific date"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        cursor.execute('''
            SELECT p.name, p.employee_id, p.department,
                   c.total_entries, c.mask_compliant, c.non_compliant, c.compliance_rate
            FROM compliance_summary c
            JOIN persons p ON c.person_id = p.id
            WHERE c.date = ?
            ORDER BY c.compliance_rate DESC
        ''', (date,))
        
        report = cursor.fetchall()
        conn.close()
        return report
    
    def get_statistics(self):
        """Get overall system statistics"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Total registered persons
        cursor.execute('SELECT COUNT(*) FROM persons WHERE active = 1')
        total_persons = cursor.fetchone()[0]
        
        # Today's entries
        cursor.execute('SELECT COUNT(*) FROM daily_entries WHERE entry_date = ?', (today,))
        today_entries = cursor.fetchone()[0]
        
        # Today's compliance
        cursor.execute('''
            SELECT SUM(mask_compliant), SUM(non_compliant)
            FROM compliance_summary
            WHERE date = ?
        ''', (today,))
        result = cursor.fetchone()
        mask_compliant = result[0] or 0
        non_compliant = result[1] or 0
        
        # Overall compliance rate
        compliance_rate = (mask_compliant / today_entries * 100) if today_entries > 0 else 0
        
        conn.close()
        
        return {
            'total_persons': total_persons,
            'today_entries': today_entries,
            'mask_compliant': mask_compliant,
            'non_compliant': non_compliant,
            'compliance_rate': round(compliance_rate, 2)
        }
    
    def deactivate_person(self, person_id):
        """Deactivate a person (soft delete)"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('UPDATE persons SET active = 0 WHERE id = ?', (person_id,))
        conn.commit()
        conn.close()
        
        print(f"[DB] Deactivated person_id={person_id}")
    
    def search_person(self, query):
        """Search person by name or employee ID"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, employee_id, department, role
            FROM persons
            WHERE (name LIKE ? OR employee_id LIKE ?) AND active = 1
        ''', (f'%{query}%', f'%{query}%'))
        
        results = cursor.fetchall()
        conn.close()
        return results


# Test the database
if __name__ == "__main__":
    db = OrganizationDatabase()
    print("Database created successfully!")
    print("\nTables created:")
    print("1. persons - Store registered employees/visitors")
    print("2. daily_entries - Log all gate entries")
    print("3. compliance_summary - Daily compliance statistics")
    print("4. settings - System configuration")
