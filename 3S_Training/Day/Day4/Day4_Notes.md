### Step 1: Ensure virtual env is activated (terminal of vs code)

## [IF NOT ALREADY DONE] To create virtual environment
python -m venv .venv #win
python3 -m venv .venv #mac

## To Activate virtual environment
.venv\Scripts\activate.bat      #windows
source .venv/bin/activate       #Mac

## To deactivate
deactivate

======================================================================================

## Install packages we need for today
*pip install google-genai chromadb langchain-chroma langchain-text-splitters pypdf python-dotenv*

## Run below in cmd prompt
*ollama pull nomic-embed-text*

Install dependencies exactly inside the virtual environment:
"d:\Your_Repository_Folder\.venv\Scripts\python.exe" -m pip install package-name*

=======================================================================================
