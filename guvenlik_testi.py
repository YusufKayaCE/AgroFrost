import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from meteostat import Point, Daily
from src.physics_engine import apply_lapse_rate, calculate_dew_point

# --- AYARLAR ---
STATION_LAT = 37.8714
STATION_LON = 32.4846
STATION_ALTITUDE = 1016 # Konya Merkez

def get_live_data():
    # Son verileri çek
    end = datetime.now()
    start = end - timedelta(days=60) 
    location = Point(STATION_LAT, STATION_LON)
    data = Daily(location, start, end)
    df = data.fetch()
    df = df.interpolate(method='linear')
    return df[['tavg', 'tmin', 'tmax', 'prcp', 'wspd']]

def run_safety_test():
    print("\n🛡️ AGROFROST GÜVENLİK SİMÜLASYONU BAŞLATILIYOR...\n")
    
    # 1. Kullanıcıdan Girdileri Al
    user_alt = float(input("1. Tarlanızın Rakımı (metre): "))
    safety_margin = float(input("2. Güvenlik Payı kaç derece olsun? (Örn: 1.5): "))
    
    print("\n📡 Veriler çekiliyor ve analiz yapılıyor...")
    df = get_live_data()
    
    # 2. Yapay Zeka Tahmini (HAM)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)
    
    last_7_days = scaled_data[-7:].reshape(1, 7, 5)
    model = load_model('models/konya_lstm_v1.h5')
    
    pred_scaled = model.predict(last_7_days, verbose=0)
    
    dummy = np.zeros((1, 5))
    dummy[0, 1] = pred_scaled[0, 0]
    
    # İSTASYONDAKİ HAM TAHMİN
    raw_station_pred = scaler.inverse_transform(dummy)[0, 1]
    
    # 3. Güvenlik Payı Uygulanmış Tahmin
    safe_station_pred = raw_station_pred - safety_margin
    
    # 4. Tarlaya Uyarlama (Fizik Motoru)
    # a) Ham Veri Tarlada Kaç Derece?
    farm_raw = apply_lapse_rate(raw_station_pred, STATION_ALTITUDE, user_alt)
    
    # b) Güvenli Veri Tarlada Kaç Derece?
    farm_safe = apply_lapse_rate(safe_station_pred, STATION_ALTITUDE, user_alt)
    
    # Çiğ Noktası Hesabı (Risk Türü İçin)
    dew_point = calculate_dew_point(farm_safe, humidity=45)

    # --- 5. SONUÇ TABLOSU (ŞEFFAFLIK RAPORU) ---
    print("\n" + "="*50)
    print(f"📊 KARŞILAŞTIRMALI ANALİZ RAPORU")
    print("="*50)
    print(f"{'METRİK':<25} | {'YAPAY ZEKA (HAM)':<15} | {'GÜVENLİK MODU 🛡️':<15}")
    print("-" * 60)
    
    print(f"{'İstasyon Tahmini':<25} | {raw_station_pred:>10.2f}°C    | {safe_station_pred:>10.2f}°C")
    print(f"{'SİZİN TARLANIZ (' + str(int(user_alt)) + 'm)':<25} | {farm_raw:>10.2f}°C    | {farm_safe:>10.2f}°C")
    
    print("-" * 60)
    print(f"📉 Uygulanan Güvenlik Kesintisi: -{safety_margin}°C")
    print("="*50)

    # KARAR MEKANİZMASI
    print("\n📢 SİSTEM TAVSİYESİ:")
    
    if farm_safe <= 0:
        print(f"🔴 DİKKAT! Güvenlik modunda DON RİSKİ tespit edildi ({farm_safe:.2f}°C).")
        if farm_raw > 0:
            print("   (Yapay Zeka 'Don Yok' dese de biz tedbiren uyarıyoruz. Önlem alın!)")
        else:
            print("   (Hem Yapay Zeka hem Güvenlik Modu hemfikir: KESİN RİSK!)")
            
        if farm_safe <= dew_point:
            print("   ❄️ Tür: BEYAZ DON (Kırağı)")
        else:
            print("   ☠️ Tür: SİYAH DON (Sinsi Tehlike)")
    else:
        print("✅ GÜVENLİ. Güvenlik payı düşülmesine rağmen risk görünmüyor.")

if __name__ == "__main__":
    run_safety_test()