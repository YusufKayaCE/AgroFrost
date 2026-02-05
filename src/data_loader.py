import pandas as pd
from datetime import datetime
from meteostat import Point, Daily

def fetch_historical_data(lat, lon, start_year, end_year):
    print(f"📡 Veri çekiliyor: {lat}, {lon} ({start_year}-{end_year})...")
    location = Point(lat, lon)
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    
    data = Daily(location, start, end)
    df = data.fetch()
    
    if df.empty:
        raise ValueError("❌ Veri bulunamadı! Koordinatları veya tarihleri kontrol et.")

    df = df.interpolate(method='linear')
    print(f"✅ Veri başarıyla çekildi: {len(df)} gün")
    
    # Gerekli sütunları seç (Meteostat sütun isimleri)
    # tavg: Ortalama, tmin: En düşük, tmax: En yüksek, prcp: Yağış, wspd: Rüzgar
    return df[['tavg', 'tmin', 'tmax', 'prcp', 'wspd']]
