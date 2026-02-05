AgroFrost: AI-Based Frost Early Warning System
AgroFrost is an advanced early warning system designed to protect agricultural producers against frost risks by combining Deep Learning (LSTM) and a Physics Engine. It takes standard meteorological data, processes it with physical calculations, and simulates the actual microclimate of the farmer's field.

Project Purpose
Standard weather applications typically base their data on stations located in city centers. However, agricultural lands are usually located at higher altitudes, and temperature differences can cause crop loss. AgroFrost solves this problem with these three fundamental components:

LSTM Model: Analyzes 25 years of historical data for the Konya region to learn future temperature patterns and generate forecasts.

Physics Engine: Calculates the "Lapse Rate" based on the altitude difference between the meteorological station and the field to find the real temperature at the field.

Safety Shield: Adds a "Risk Tolerance" to the model against sudden Cold Fronts, reducing the margin of error and presenting the worst-case scenario to the farmer.

Key Features
Live Data Integration: Daily and instant meteorological data is automatically fetched via the Meteostat API.

Deep Learning Architecture: Customized LSTM (Long Short-Term Memory) layers based on TensorFlow and Keras are used.

Safety Mode: The user can adjust risk tolerance to make the model's predictions more cautious.

Interactive Dashboard: Thanks to the interface developed with Streamlit, farmers can analyze data and examine charts without knowing how to code.

Frost Type Detection: The system can distinguish between "White Frost" or "Black Frost" risks based on the temperature and humidity balance.

Technologies Used
This project was developed using industry-standard libraries in the fields of data science and artificial intelligence:

Python: The project's fundamental programming language.

TensorFlow & Keras: Training and architecture of the LSTM model.

Streamlit: Web-based user interface and dashboard development.

Pandas & NumPy: Processing of time-series data and matrix operations.

Scikit-Learn: Data preprocessing and normalization (MinMaxScaler).

Meteostat API: Fetching climate data.

Matplotlib: Data visualization and plotting.

Installation and Execution
Follow the steps below to run the project on your local computer:

Clone the Project:

Bash
git clone https://github.com/YusufKayace/AgroFrost.git
cd AgroFrost
Install Requirements:

Bash
pip install -r requirements.txt
Start the Application:

Bash
python -m streamlit run app.py
Project Structure
app.py: Main file of the Streamlit web interface.

src/physics_engine.py: Physics engine performing altitude difference and dew point calculations.

models/: Folder containing trained LSTM model files (.h5).

requirements.txt: List of libraries required for the project.

Developer: YUSUF TALHA KAYA