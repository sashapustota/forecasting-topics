# Create a virtual environment
python3 -m venv forecasting-topics-venv

# Activate the virtual environment
source ./forecasting-topics-venv/bin/activate

# Install requirements
python3 -m pip install --upgrade pip
python3 -m pip install -r ./requirements.txt

# deactivate
deactivate

# To remove the virtual environment run the following command in the terminal
#rm -rf forecasting-topics-venv