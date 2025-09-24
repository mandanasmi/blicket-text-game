#!/usr/bin/env python3
"""
Firebase Setup Helper Script
This script helps you extract Firebase credentials from the downloaded JSON file
and create a properly formatted .env file.
"""

import json
import os
import sys

def setup_firebase_credentials():
    print("🔥 Firebase Setup Helper")
    print("=" * 50)
    
    # Get Firebase project details
    project_id = input("Enter your Firebase Project ID: ").strip()
    database_url = input("Enter your Firebase Database URL (e.g., https://your-project-id-default-rtdb.firebaseio.com/): ").strip()
    
    # Get path to downloaded JSON file
    json_path = input("Enter the path to your downloaded Firebase service account JSON file: ").strip()
    
    if not os.path.exists(json_path):
        print(f"❌ Error: File not found at {json_path}")
        return False
    
    try:
        # Read and parse the JSON file
        with open(json_path, 'r') as f:
            credentials = json.load(f)
        
        # Extract required fields
        private_key_id = credentials.get('private_key_id', '')
        private_key = credentials.get('private_key', '')
        client_email = credentials.get('client_email', '')
        client_id = credentials.get('client_id', '')
        client_x509_cert_url = credentials.get('client_x509_cert_url', '')
        
        # Create .env file content
        env_content = f"""# Firebase Configuration
FIREBASE_PROJECT_ID={project_id}
FIREBASE_PRIVATE_KEY_ID={private_key_id}
FIREBASE_PRIVATE_KEY="{private_key}"
FIREBASE_CLIENT_EMAIL={client_email}
FIREBASE_CLIENT_ID={client_id}
FIREBASE_CLIENT_X509_CERT_URL={client_x509_cert_url}
FIREBASE_DATABASE_URL={database_url}
"""
        
        # Write .env file
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("✅ Successfully created .env file!")
        print("🔒 Your Firebase credentials are now configured.")
        print("⚠️  Keep your .env file secure and never commit it to version control.")
        
        return True
        
    except json.JSONDecodeError:
        print("❌ Error: Invalid JSON file format")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_firebase_connection():
    """Test the Firebase connection"""
    print("\n🧪 Testing Firebase connection...")
    
    try:
        import firebase_admin
        from firebase_admin import credentials, db
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        # Check if Firebase is already initialized
        if firebase_admin._apps:
            print("✅ Firebase already initialized")
            return True
        
        # Create credentials from environment variables
        firebase_credentials = {
            "type": "service_account",
            "project_id": os.getenv("FIREBASE_PROJECT_ID"),
            "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
            "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace("\\n", "\n"),
            "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
            "client_id": os.getenv("FIREBASE_CLIENT_ID"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL"),
            "universe_domain": "googleapis.com"
        }
        
        cred = credentials.Certificate(firebase_credentials)
        firebase_admin.initialize_app(cred, {
            'databaseURL': os.getenv("FIREBASE_DATABASE_URL")
        })
        
        # Test database connection
        db_ref = db.reference()
        test_data = {"test": "connection", "timestamp": "2024-01-01"}
        db_ref.child("test").set(test_data)
        
        print("✅ Firebase connection successful!")
        print("✅ Database write test successful!")
        
        # Clean up test data
        db_ref.child("test").delete()
        print("✅ Test data cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Firebase connection failed: {e}")
        return False

if __name__ == "__main__":
    print("This script will help you set up Firebase credentials for the Blicket Text Game.")
    print()
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test existing configuration
        test_firebase_connection()
    else:
        # Setup new configuration
        if setup_firebase_credentials():
            print("\nWould you like to test the connection now? (y/n): ", end="")
            if input().lower().startswith('y'):
                test_firebase_connection()
