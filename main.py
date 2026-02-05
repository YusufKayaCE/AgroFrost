import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.data_loader import fetch_historical_data
from src.ai_engine import create_windowed_dataset, build_lstm_model

# KONYA AYARLARI
LAT = 37.8714
LON = 32.4846
START_YEAR = 2000
END_YEAR = 2025

def run_training_pipeline():
    print("🚀 AgroFrost Başlatılıyor...")
    
    # 1. Klasör Kontrolü
    if not os.path.exists('models'):
        os.makedirs('models')

    # 2. Veri
    df = fetch_historical_data(LAT, LON, START_YEAR, END_YEAR)
    
    # 3. Ön İşleme
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)
    
    X, y = create_windowed_dataset(scaled_data, window_size=7)
    
    # 4. Model
    print(f"🧠 Model Eğitiliyor (Veri Boyutu: {X.shape})...")
    model = build_lstm_model((X.shape[1], X.shape[2]))
    
    model.fit(X, y, epochs=20, batch_size=32, validation_split=0.1)
    
    # 5. Kayıt
    save_path = 'models/konya_lstm_v1.h5'
    model.save(save_path)
    print(f"✅ Model başarıyla kaydedildi: {save_path}")

if __name__ == "__main__":
    run_training_pipeline()
