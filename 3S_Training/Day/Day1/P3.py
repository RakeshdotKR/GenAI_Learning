# pip, virtual environment, .env file
# pip: pip installs Packages. Package manager

# virtual environment: A 'venv' is a directory that contains
# 1. A copy of the python interpreter
# 2. Its own 'pip'
# 3. Its own site-packages directory. Default target location
# where python installs third party libraries
# 4. Activation scripts that modify PATH

python --version #windows
python3 --version #mac

# To create virtual environment
python -m venv .venv #win
python3 -m venv .venv #mac

# Activate virtual environment
.venv\Scripts\activate.bat #win
source .venv/bin/activate #Mac

#deactivate
deactivate

# pip installation (install dependencies/libraries)
pip install package_name #syntax

pip install python-dotenv
