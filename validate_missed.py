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
STATION_ALT = 1016   
# Burada doğrudan istasyon tahminine bakacağız.
# Çünkü eğer AI istasyonda bile donu kaçırdıysa, tarlada da kaçırmış demektir.
START_DATE = datetime(2015, 1, 1) 
END_DATE = datetime(2025, 1, 1)

def run_missed_frost_test():
    print("🚨 AgroFrost 'Kaçırılan Don' (False Negative) Testi Başlıyor...")
    
    # 1. Veri
    location = Point(LAT, LON)
    data = Daily(location, START_DATE, END_DATE)
    df = data.fetch()
    df = df.interpolate(method='linear')
    
    # 2. Model
    model = load_model('models/konya_lstm_v1.h5')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['tavg', 'tmin', 'tmax', 'prcp', 'wspd']].values)
    
    missed_events = []
    
    print(f"Toplam {len(df)} gün taranıyor...")
    
    # 3. Tarama
    for i in range(7, len(df)):
        input_seq = scaled_data[i-7:i].reshape(1, 7, 5)
        pred_scaled = model.predict(input_seq, verbose=0)
        
        dummy = np.zeros((1, 5))
        dummy[0, 1] = pred_scaled[0, 0]
        model_pred = scaler.inverse_transform(dummy)[0, 1]
        
        actual_temp = df.iloc[i]['tmin']
        date = df.index[i].strftime('%d-%m-%Y')
        
        # --- KRİTİK HATA MANTIĞI ---
        # Gerçekte DON VAR (< 0) ama Model DON YOK (> 0.5) demiş.
        # 0.5 derece marj koydum ki 0.1 gibi kıl payı kaçanları eleyelim,
        # bize "Çiftçiyi yakan" büyük hatalar lazım.
        if actual_temp < 0 and model_pred > 0.5:
            diff = abs(actual_temp - model_pred)
            
            missed_events.append({
                "Tarih": date,
                "Gerçek_MGM": f"{actual_temp:.1f}°C",
                "Hatalı_Tahmin": f"{model_pred:.1f}°C",
                "Hata_Payı": f"{diff:.1f}°C",
                "Durum": "❌ RİSKLİ HATA"
            })

    # --- SONUÇLAR ---
    print("\n" + "="*60)
    print(f"⚠️ DİKKAT: Toplam {len(missed_events)} kritik don olayı tahmin edilemedi.")
    print("="*60)
    
    if len(missed_events) > 0:
        results_df = pd.DataFrame(missed_events)
        # Hataya göre sıralayalım (En büyüğü en üstte)
        results_df['Sort_Key'] = results_df['Hata_Payı'].apply(lambda x: float(x.replace('°C','')))
        results_df = results_df.sort_values(by='Sort_Key', ascending=False).drop(columns=['Sort_Key'])
        
        print(results_df.head(10).to_string(index=False))
        
        results_df.to_csv("AgroFrost_Kacirilan_Donlar.csv", index=False)
        print("\n✅ Hata raporu 'AgroFrost_Kacirilan_Donlar.csv' dosyasına kaydedildi.")
    else:
        print("Mükemmel! Model belirtilen kriterlerde hiçbir don olayını kaçırmadı.")

if __name__ == "__main__":
    run_missed_frost_test()