import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from meteostat import Point, Daily
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from src.physics_engine import apply_lapse_rate

# --- AYARLAR ---
LAT = 37.8714
LON = 32.4846
STATION_ALT = 1016   # Konya Merkez
TEST_FARM_ALT = 1250 # Test Tarlası (Biraz daha yüksekte)
START_DATE = datetime(2000, 1, 1) # Son 1-2 yılı test edelim
END_DATE = datetime(2024, 12, 31)

def run_validation_test():
    print("🕵️‍♂️ AgroFrost Dedektifi Geçmiş Kayıtları İnceliyor...")
    
    # 1. Gerçek Verileri Çek
    location = Point(LAT, LON)
    data = Daily(location, START_DATE, END_DATE)
    df = data.fetch()
    df = df.interpolate(method='linear')
    
    # 2. Modeli Hazırla
    model = load_model('models/konya_lstm_v1.h5')
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Tüm veriyi ölçeklendir (Scaler'ı eğitmek için)
    scaled_data = scaler.fit_transform(df[['tavg', 'tmin', 'tmax', 'prcp', 'wspd']].values)
    
    caught_events = [] # Yakalanan olayları buraya atacağız
    
    print(f"Toplam {len(df)} gün taranıyor...")
    
    # 3. Gün Gün Gez ve Tahmin Yap
    # İlk 7 günü atlıyoruz çünkü geçmiş veriye ihtiyacımız var
    for i in range(7, len(df)):
        # Girdi: Önceki 7 gün
        input_seq = scaled_data[i-7:i]
        input_reshaped = input_seq.reshape(1, 7, 5)
        
        # Tahmin (Model ne dedi?)
        pred_scaled = model.predict(input_reshaped, verbose=0)
        
        # Ölçeği geri çevir
        dummy = np.zeros((1, 5))
        dummy[0, 1] = pred_scaled[0, 0]
        model_station_pred = scaler.inverse_transform(dummy)[0, 1]
        
        # --- GERÇEKLER VS AGROFROST ---
        actual_station_temp = df.iloc[i]['tmin'] # O gün gerçekten ne oldu?
        date = df.index[i].strftime('%d-%m-%Y')
        
        # Fizik Motorunu Uygula (Tarlaya uyarla)
        farm_prediction = apply_lapse_rate(model_station_pred, STATION_ALT, TEST_FARM_ALT)
        
        # --- DEDEKTİF MANTIĞI (THE CATCH) ---
        # Kriter: İstasyon > 0.5°C (Güvenli) AMA AgroFrost < 0°C (Risk)
        # 0.5 derece marj koydum ki sınır durumları eyleyelim, net hataları bulalım.
        if actual_station_temp > 0.5 and farm_prediction < 0:
            diff = actual_station_temp - farm_prediction
            caught_events.append({
                "Tarih": date,
                "MGM_İstasyon (Gerçek)": f"{actual_station_temp:.1f}°C",
                "AgroFrost_Tarla (Tahmin)": f"{farm_prediction:.1f}°C",
                "Fark": f"{diff:.1f}°C",
                "Durum": "⚠️ GİZLİ DON YAKALANDI"
            })

    # --- RAPORLAMA ---
    print("\n" + "="*60)
    print(f"🎉 TEST SONUCU: {len(caught_events)} adet Kritik 'Gizli Don' olayı yakalandı!")
    print("="*60)
    
    if len(caught_events) > 0:
        results_df = pd.DataFrame(caught_events)
        print(results_df.to_string(index=False))
        
        # CSV olarak da kaydet, yatırımcıya gösteririz
        results_df.to_csv("AgroFrost_Yakalanan_Donlar.csv", index=False)
        print("\n✅ Detaylı liste 'AgroFrost_Yakalanan_Donlar.csv' dosyasına kaydedildi.")
    else:
        print("Taranan aralıkta bu kriterlere uyan keskin bir ayrım bulunamadı.")
        print("Not: Rakım farkını artırarak (TEST_FARM_ALT) tekrar deneyebilirsin.")

if __name__ == "__main__":
    run_validation_test()