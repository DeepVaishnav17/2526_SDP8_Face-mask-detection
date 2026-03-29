import requests
import sys

API_URL = "http://localhost:5000"

def set_user(name):
    try:
        response = requests.post(f"{API_URL}/set_user", json={"name": name})
        if response.status_code == 200:
            print(f"✅ Active user set to: {name}")
        else:
            print(f"❌ Failed to set user: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python set_user.py <name>")
        name = input("Enter person name: ")
    else:
        name = sys.argv[1]
    
    set_user(name)
