import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from meteostat import Point, Daily
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from src.physics_engine import apply_lapse_rate, calculate_dew_point

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="AgroFrost AI", page_icon="❄️", layout="wide")

st.title("❄️ AgroFrost: Zirai Don Erken Uyarı Sistemi")
st.markdown("**Konya Pilot Bölgesi** | Yapay Zeka Destekli Mikroklima Analizi")

# --- YAN MENÜ ---
st.sidebar.header("🚜 Tarla ve Risk Ayarları")
user_lat = st.sidebar.number_input("Enlem", value=37.8714, format="%.4f")
user_lon = st.sidebar.number_input("Boylam", value=32.4846, format="%.4f")
user_alt = st.sidebar.number_input("Tarla Rakımı (m)", value=1250, step=10)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Güvenlik Kalkanı")
safety_margin = st.sidebar.slider(
    "Güvenlik Payı (°C)", 
    min_value=0.0, max_value=3.0, value=2.0, step=0.5,
    help="Model tahmininden düşülecek güvenlik payı."
)
st.sidebar.info(f"Aktif: -{safety_margin}°C düşülüyor.")

# --- ANALİZ FONKSİYONLARI ---
@st.cache_data
def get_prediction_data(lat, lon):
    end = datetime.now()
    start = end - timedelta(days=45)
    location = Point(lat, lon)
    data = Daily(location, start, end)
    df = data.fetch()
    df = df.interpolate(method='linear')
    return df

def run_analysis():
    with st.spinner('📡 Uydu verileri işleniyor...'):
        df = get_prediction_data(user_lat, user_lon)
        
    if len(df) < 10:
        st.error("Veri alınamadı.")
        return

    # Model Tahmini
    model = load_model('models/konya_lstm_v1.h5')
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    input_data = df[['tavg', 'tmin', 'tmax', 'prcp', 'wspd']].tail(7).values
    input_scaled = scaler.fit_transform(input_data)
    X_pred = input_scaled.reshape(1, 7, 5)
    
    pred_scaled = model.predict(X_pred, verbose=0)
    dummy = np.zeros((1, 5))
    dummy[0, 1] = pred_scaled[0, 0]
    
    # HAM İSTASYON TAHMİNİ
    raw_station_pred = scaler.inverse_transform(dummy)[0, 1]
    
    # GÜVENLİ TAHMİN (İstasyon)
    safe_station_pred = raw_station_pred - safety_margin

    # TARLAYA UYARLAMA (Fizik Motoru)
    # 1. Ham Veri Tarlada Kaç Derece?
    farm_raw = apply_lapse_rate(raw_station_pred, 1016, user_alt)
    # 2. Güvenli Veri Tarlada Kaç Derece? (Kullanıcıya gösterilen ana değer)
    farm_safe = apply_lapse_rate(safe_station_pred, 1016, user_alt)
    
    dew_point = calculate_dew_point(farm_safe, humidity=45)
    
    # --- SONUÇ KPI KARTLARI ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Yapay Zeka (Ham)", f"{raw_station_pred:.1f}°C", "MGM İstasyonu")
    with col2:
        st.metric("Güvenli Tahmin", f"{safe_station_pred:.1f}°C", f"-{safety_margin}°C (Güv. Payı)")
    with col3:
        st.metric("SİZİN TARLANIZ", f"{farm_safe:.1f}°C", f"Rakım: {user_alt}m", delta_color="inverse")
    with col4:
        if farm_safe <= 0:
            if farm_safe <= dew_point:
                 st.error("🚨 RİSK: BEYAZ DON")
            else:
                 st.error("☠️ RİSK: SİYAH DON")
        elif farm_safe <= 2.0:
             st.warning("⚠️ DİKKAT: Sınırda")
        else:
            st.success("✅ GÜVENLİ")

    # --- DETAYLI KARŞILAŞTIRMA TABLOSU (İSTEĞİN ÜZERİNE EKLENDİ) ---
    st.markdown("---")
    st.subheader("📊 Detaylı Analiz Raporu")
    
    comparison_data = {
        "Metrik": ["İstasyon (Merkez)", f"Sizin Tarlanız ({int(user_alt)}m)"],
        "Yapay Zeka (Ham)": [f"{raw_station_pred:.2f}°C", f"{farm_raw:.2f}°C"],
        "Güvenlik Modu 🛡️": [f"{safe_station_pred:.2f}°C", f"{farm_safe:.2f}°C"],
        "Fark": [f"-{safety_margin}°C", f"-{safety_margin}°C"]
    }
    df_comp = pd.DataFrame(comparison_data)
    st.table(df_comp)
    
    if farm_raw > 0 and farm_safe < 0:
        st.info("💡 **Bilgi:** Yapay Zeka 'Don Yok' öngörse de, Güvenlik Kalkanı sizi korumak için 'Risk' uyarısı veriyor.")

    # Grafik
    st.subheader("📈 Sıcaklık Trendi")
    dates = df.index[-30:]
    station_temps = df['tmin'].tail(30).values
    farm_temps = [apply_lapse_rate(t, 1016, user_alt) for t in station_temps]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, station_temps, label="İstasyon", linestyle="--", alpha=0.5)
    ax.plot(dates, farm_temps, label="Sizin Tarlanız (Güvenli Mod)", color="red", linewidth=2)
    ax.axhline(0, color='black', linewidth=1)
    ax.fill_between(dates, farm_temps, 0, where=(np.array(farm_temps) < 0), color='red', alpha=0.1)
    ax.legend()
    st.pyplot(fig)

if st.button("Analizi Başlat"):
    run_analysis()