import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from meteostat import Point, Daily
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from src.physics_engine import apply_lapse_rate

# --- AYARLAR ---
LAT = 37.8714
LON = 32.4846
STATION_ALT = 1016  # İstasyon Rakımı
FARM_ALT = 1250     # ÖRNEK: Dağ yamacındaki bir kiraz bahçesi (Daha yüksek)

def generate_comparison_report():
    print("🎨 Gelişmiş Grafik Motoru Çalışıyor...")
    
    # 1. Veriyi Çek
    end_date = datetime.now()
    start_date = end_date - timedelta(days=15) # Son 15 gün yeterli
    
    location = Point(LAT, LON)
    data = Daily(location, start_date, end_date)
    df = data.fetch()
    df = df.interpolate(method='linear')
    
    dates = df.index
    station_temps = df['tmin'].values
    
    # 2. Tarlayı Hesapla (Fizik Motoru Devrede)
    # Her günün sıcaklığını Lapse Rate ile tarlaya uyarla
    farm_temps = [apply_lapse_rate(t, STATION_ALT, FARM_ALT) for t in station_temps]
    
    # 3. Gelecek Tahmini (AI)
    model = load_model('models/konya_lstm_v1.h5')
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    input_data = df[['tavg', 'tmin', 'tmax', 'prcp', 'wspd']].tail(7).values
    input_scaled = scaler.fit_transform(input_data)
    X_pred = input_scaled.reshape(1, 7, 5)
    
    pred_scaled = model.predict(X_pred, verbose=0)
    dummy = np.zeros((1, 5))
    dummy[0, 1] = pred_scaled[0, 0]
    pred_station = scaler.inverse_transform(dummy)[0, 1]
    
    # Tahmini Tarlaya Uyarlama
    pred_farm = apply_lapse_rate(pred_station, STATION_ALT, FARM_ALT)
    
    tomorrow = dates[-1] + timedelta(days=1)
    
    # --- GRAFİK (Comparison Mode) ---
    plt.figure(figsize=(12, 6))
    
    # İSTASYON VERİSİ (Mavi - Güvenli Gibi)
    plt.plot(dates, station_temps, label=f'İstasyon ({STATION_ALT}m)', color='#3498db', linestyle='--', alpha=0.6)
    plt.plot(tomorrow, pred_station, marker='o', markersize=10, color='#3498db', alpha=0.6)
    
    # TARLA VERİSİ (Turuncu - Gerçek Risk)
    plt.plot(dates, farm_temps, label=f'SİZİN TARLANIZ ({FARM_ALT}m)', color='#e74c3c', linewidth=3, marker='o')
    plt.plot(tomorrow, pred_farm, marker='*', markersize=25, color='#e74c3c', label=f'Tarla Tahmini: {pred_farm:.1f}°C', zorder=10)
    
    # Kritik Eşik (0 Derece)
    plt.axhline(y=0, color='black', linewidth=1)
    
    # Görsel Süslemeler
    plt.title(f"AgroFrost Farkı: İstasyon vs Tarla ({datetime.now().strftime('%d-%m-%Y')})", fontsize=14)
    plt.ylabel('Sıcaklık (°C)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig('agrofrost_final_karsilastirma.png')
    print("✅ Final raporu oluşturuldu: agrofrost_final_karsilastirma.png")
    plt.show()

if __name__ == "__main__":
    generate_comparison_report()