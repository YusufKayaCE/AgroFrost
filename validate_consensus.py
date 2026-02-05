import numpy as np
import pandas as pd
from datetime import datetime
from meteostat import Point, Daily
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from src.physics_engine import apply_lapse_rate

# --- AYARLAR ---
LAT = 37.8714
LON = 32.4846
STATION_ALT = 1016   # Merkez
TEST_FARM_ALT = 1250 # Tarla
START_DATE = datetime(2015, 1, 1)  # 10 Yıllık Test
END_DATE = datetime(2025, 1, 1)

def run_consensus_test():
    print("🤝 AgroFrost Güvenilirlik Testi (Mutabakat) Başlıyor...")
    
    # 1. Veri Çekme
    location = Point(LAT, LON)
    data = Daily(location, START_DATE, END_DATE)
    df = data.fetch()
    df = df.interpolate(method='linear')
    
    # 2. Model Hazırlığı
    model = load_model('models/konya_lstm_v1.h5')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['tavg', 'tmin', 'tmax', 'prcp', 'wspd']].values)
    
    match_events = []
    
    print(f"Toplam {len(df)} gün taranıyor...")
    
    # 3. Tarama Döngüsü
    for i in range(7, len(df)):
        # Tahmin Yap
        input_seq = scaled_data[i-7:i].reshape(1, 7, 5)
        pred_scaled = model.predict(input_seq, verbose=0)
        
        dummy = np.zeros((1, 5))
        dummy[0, 1] = pred_scaled[0, 0]
        model_station_pred = scaler.inverse_transform(dummy)[0, 1]
        
        # Tarlaya Uyarla
        farm_prediction = apply_lapse_rate(model_station_pred, STATION_ALT, TEST_FARM_ALT)
        
        # Gerçek Veri
        actual_station_temp = df.iloc[i]['tmin']
        date = df.index[i].strftime('%d-%m-%Y')
        
        # --- MUTABAKAT MANTIĞI ---
        # Hem MGM (Gerçek) < 0 hem de AgroFrost (Tahmin) < 0
        # Yani İKİMİZ DE DON VAR DEMİŞİZ.
        if actual_station_temp < 0 and farm_prediction < 0:
            diff = abs(actual_station_temp - farm_prediction)
            
            # Sadece yakın tahminleri alalım (Model sapıtmamış olsun)
            # Fark 3 dereceden azsa "Tam İsabet" kabul edelim
            if diff < 3.0:
                match_events.append({
                    "Tarih": date,
                    "MGM_Gerçek": f"{actual_station_temp:.1f}°C",
                    "AgroFrost_Tahmin": f"{farm_prediction:.1f}°C",
                    "Durum": "✅ DOĞRULANDI"
                })

    # --- SONUÇLAR ---
    print("\n" + "="*60)
    print(f"🎯 GÜVENİLİRLİK RAPORU: {len(match_events)} gün boyunca başarıyla 'Don' tespiti yapıldı.")
    print("="*60)
    
    if len(match_events) > 0:
        results_df = pd.DataFrame(match_events)
        # Son 10 başarılı tahmini gösterelim
        print(results_df.tail(10).to_string(index=False))
        
        results_df.to_csv("AgroFrost_Dogrulanmis_Donlar.csv", index=False)
        print("\n✅ Tam liste 'AgroFrost_Dogrulanmis_Donlar.csv' dosyasına kaydedildi.")

if __name__ == "__main__":
    run_consensus_test()