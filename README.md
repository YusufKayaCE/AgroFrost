# AgroFrost: AI-Based Frost Early Warning System

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

[🇹🇷 Türkçe Versiyonu İçin Tıklayın (Click for Turkish Version)](README_TR.md)

AgroFrost is an advanced early warning system designed to protect agricultural producers against frost risks by combining **Deep Learning (LSTM)** and a **Physics Engine**. It processes standard meteorological data through physical calculations to simulate the actual microclimate of a farmer's specific field.



## Project Objective

Standard weather applications typically rely on data from stations located in city centers. However, agricultural lands are often situated at higher altitudes and experience significantly lower temperatures. AgroFrost solves this problem with three core components:

1.  **LSTM Model:** Analyzes 25 years of historical data (specifically for the pilot region) to learn temperature patterns and generate future forecasts.
2.  **Physics Engine:** Calculates the **Lapse Rate** based on the altitude difference between the meteorology station and the field to determine the real on-site temperature.
3.  **Safety Shield:** Adds a configurable "Risk Tolerance" margin to the model against sudden **Cold Fronts**, minimizing the error rate and presenting the worst-case scenario to the user.

## Key Features

* **Live Data Integration:** Automatic fetching of daily and hourly meteorological data via the Meteostat API.
* **Deep Learning Architecture:** Utilizes customized LSTM (Long Short-Term Memory) layers built with TensorFlow and Keras.
* **Safety Mode:** Allows users to adjust risk tolerance levels, making model predictions more cautious during critical periods.
* **Interactive Dashboard:** A user-friendly web interface developed with Streamlit, enabling farmers to perform analyses and visualize trends without coding knowledge.
* **Frost Type Detection:** Distinguishes between "White Frost" and the more dangerous "Black Frost" risks based on temperature and dew point balance.

## Technologies Used

This project was developed using industry-standard libraries in data science and artificial intelligence:

* **Python:** Core programming language.
* **TensorFlow & Keras:** Architecture and training of the LSTM model.
* **Streamlit:** Web-based interactive user interface and dashboard development.
* **Pandas & NumPy:** Time-series data processing and matrix operations.
* **Scikit-Learn:** Data preprocessing and normalization (MinMaxScaler).
* **Meteostat API:** Retrieval of historical and live climate data.
* **Matplotlib:** Data visualization and graphing.

## Installation & Usage

Follow these steps to run the project on your local machine:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YusufKayace/AgroFrost.git](https://github.com/YusufKayace/AgroFrost.git)
    cd AgroFrost
    ```

2.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application:**
    ```bash
    python -m streamlit run app.py
    ```

## Project Structure

* `app.py`: The main file for the Streamlit web interface.
* `src/physics_engine.py`: The physics engine handling altitude adjustments and dew point calculations.
* `models/`: Directory containing the trained LSTM model files (.h5).
* `requirements.txt`: List of dependencies required for the project.

---
**Developer:** YUSUF TALHA KAYA