import os
from dotenv import load_dotenv

def test_api_key():
    # Load the environment variables
    load_dotenv()
    
    # Get the API key
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key is None:
        print("❌ API key not found. Make sure you have created a .env file with OPENAI_API_KEY=")
        return False
    
    if not api_key.startswith("sk-"):
        print("❌ API key format looks incorrect. Should start with 'sk-'")
        return False
    
    print("✅ API key found and format looks correct")
    # Print first 5 and last 4 characters of the key for verification
    print(f"Key starts with: {api_key[:5]}...")
    print(f"Key ends with: ...{api_key[-4:]}")
    return True

if __name__ == "__main__":
    test_api_key()