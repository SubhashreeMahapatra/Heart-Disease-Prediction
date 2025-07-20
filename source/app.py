import sys
import os

# Get the absolute path of the root directory 
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_DIR)

from source.utils import ConnectDB

def main():
    print("🔄 Fetching data from DB...")
    db = ConnectDB()
    df = db.retrieve_data()
    print("✅ Data fetched!")
    print(df.head())

if __name__ == "__main__":
    main()