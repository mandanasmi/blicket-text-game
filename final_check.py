#!/usr/bin/env python3
import firebase_admin
from firebase_admin import credentials, db
import os

try:
    import json
    from dotenv import load_dotenv
    load_dotenv()
    
    firebase_config = {
        "type": "service_account",
        "project_id": os.getenv('FIREBASE_PROJECT_ID'),
        "private_key_id": os.getenv('FIREBASE_PRIVATE_KEY_ID'),
        "private_key": os.getenv('FIREBASE_PRIVATE_KEY').replace('\\n', '\n'),
        "client_email": os.getenv('FIREBASE_CLIENT_EMAIL'),
        "client_id": os.getenv('FIREBASE_CLIENT_ID'),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": os.getenv('FIREBASE_CLIENT_X509_CERT_URL')
    }
    
    cred = credentials.Certificate(firebase_config)
    firebase_admin.initialize_app(cred, {
        'databaseURL': os.getenv('FIREBASE_DATABASE_URL')
    })
    
    print("✅ Firebase initialized\n")
    
    ref = db.reference()
    all_data = ref.get()
    
    if not all_data:
        print("No data found")
        exit(0)
    
    found_progress = False
    for participant_id, data in all_data.items():
        if isinstance(data, dict) and 'main_game' in data:
            main_game = data['main_game']
            progress_keys = [k for k in main_game.keys() if k.endswith('_progress')]
            if progress_keys:
                found_progress = True
                print(f"📋 {participant_id}:")
                for key in progress_keys:
                    print(f"   - {key}")
                    ref.child(participant_id).child('main_game').child(key).delete()
                    print(f"     ✅ Deleted")
                print()
    
    if not found_progress:
        print("✅ No round_X_progress entries found! Firebase is clean.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

