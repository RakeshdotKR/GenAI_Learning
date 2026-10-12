import os
from dotenv import load_dotenv,dotenv_values,find_dotenv

# Loads .env from current directory
load_dotenv()

config = dotenv_values(".env")
api_key = config.get("GEMINI_API_KEY")

api_key = os.getenv("GEMINI_API_KEY","")
api_key = os.environ["GEMINI_API_KEY"] 

