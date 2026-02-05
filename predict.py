import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from meteostat import Point, Daily
from src.physics_engine import calculate_dew_point, apply_lapse_rate

# --- AYARLAR ---
# Konya İstasyon Bilgileri (Modelin Referans Noktası)
STATION_LAT = 37.8714
STATION_LON = 32.4846
STATION_ALTITUDE = 1016 # Konya Ovası ortalama rakım (metre)

def get_live_data():
    """
    Modelin tahmin yapabilmesi için SON 1 YILLIK veriyi çeker.
    Neden 1 yıl? Çünkü 'Scaler' (Ölçekleyici) kalibrasyonu için 
    geniş bir aralığa ihtiyacımız var. Sadece dünü çekersek matematik bozulur.
    """
    end = datetime.now()
    start = end - timedelta(days=365) 
    
    location = Point(STATION_LAT, STATION_LON)
    data = Daily(location, start, end)
    df = data.fetch()
    
    # Eksikleri doldur
    df = df.interpolate(method='linear')
    
    # Sonuçta bize sadece son 7 gün lazım ama scaler için hepsini kullandık
    return df[['tavg', 'tmin', 'tmax', 'prcp', 'wspd']]

def make_prediction():
    print("📡 Canlı meteoroloji verileri alınıyor...")
    df = get_live_data()
    
    # --- VERİ HAZIRLIĞI ---
    # Veriyi 0 ile 1 arasına sıkıştır (Eğitimdeki mantığın aynısı)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)
    
    # Modelin son 7 güne bakması lazım (Son 7 satırı al)
    last_7_days = scaled_data[-7:] 
    
    # LSTM formatına sok: (1, 7, 5) -> 1 tahmin, 7 gün, 5 özellik
    X_input = last_7_days.reshape(1, 7, 5)
    
    # --- KATMAN 1: YAPAY ZEKA TAHMİNİ ---
    print("🧠 Yapay Zeka (LSTM) çalıştırılıyor...")
    model = load_model('models/konya_lstm_v1.h5')
    prediction_scaled = model.predict(X_input, verbose=0)
    
    # Tahmini normal sıcaklığa geri çevir (Inverse Transform)
    # Scaler 5 sütun bekler, bizde tek çıktı var. Hile yaparak matrisi dolduralım.
    dummy_matrix = np.zeros((1, 5))
    dummy_matrix[0, 1] = prediction_scaled[0, 0] # 1. indeks tmin idi
    prediction_actual = scaler.inverse_transform(dummy_matrix)[0, 1]
    
    print(f"\n--- 🌡️ İSTASYON TAHMİNİ (MERKEZ) ---")
    print(f"Yarın için Öngörülen Min. Sıcaklık: {prediction_actual:.2f}°C")
    
    return prediction_actual, df.iloc[-1] # Tahmin ve son günün verisi

# --- ANA PROGRAM ---
if __name__ == "__main__":
    base_pred, last_day_data = make_prediction()
    
    print("\n🚜 --- KATMAN 2: TARLA ÖZEL ANALİZİ ---")
    user_alt = float(input("Lütfen tarlanızın rakımını (metre) girin: "))
    
    # 1. Lapse Rate Düzeltmesi (Yükseklik Farkı)
    field_temp = apply_lapse_rate(base_pred, STATION_ALTITUDE, user_alt)
    
    # 2. Çiğ Noktası Riski (Siyah Don)
    # Eğer nem verisi yoksa Konya ortalaması %40 al
    humidity = 40 
    # Not: Meteostat ücretsiz sürümde bazen nem vermez, burada varsayım yaptık.
    
    dew_point = calculate_dew_point(field_temp, humidity)
    
    print(f"\n📊 SONUÇ RAPORU:")
    print(f"--------------------------------------")
    print(f"📍 İstasyon Sıcaklığı : {base_pred:.2f}°C")
    print(f"🏔️ Sizin Tarlanız     : {field_temp:.2f}°C (Rakım farkı uygulandı)")
    print(f"💧 Çiğ Noktası        : {dew_point:.2f}°C")
    print(f"--------------------------------------")
    
    # KARAR MEKANİZMASI
    if field_temp <= 0:
        if field_temp <= dew_point:
            print("⚠️ RİSK: KIRAĞI (Beyaz Don). Bitki buzla kaplanacak.")
        else:
            print("☠️ KRİTİK RİSK: SİYAH DON! Havadaki nem donmadan bitki donacak.")
            print("   (Sulama sistemlerini şimdiden hazırlayın!)")
    else:
        print("✅ Güvendesiniz. Don riski düşük.")