# AgroFrost: Yapay Zeka Tabanlı Zirai Don Erken Uyarı Sistemi

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

[🇺🇸 Click for English Version](README.md)

AgroFrost, tarımsal üreticileri don riskine karşı koruyan, Derin Öğrenme (LSTM) ve Fizik Motorunu birleştiren gelişmiş bir erken uyarı sistemidir. Standart meteorolojik verileri alır, fiziksel hesaplamalarla işler ve çiftçinin tarlasındaki gerçek mikroklimayı simüle eder.



## Projenin Amacı

Standart hava durumu uygulamaları genellikle şehir merkezlerindeki istasyon verilerini baz alır. Ancak tarım arazileri genellikle daha yüksek rakımlarda bulunur ve sıcaklık farkları ürün kaybına neden olabilir. AgroFrost bu sorunu şu üç temel bileşenle çözer:

1.  **LSTM Modeli:** Konya bölgesine ait 25 yıllık tarihsel veriyi analiz ederek geleceğe yönelik sıcaklık desenlerini öğrenir ve tahmin üretir.
2.  **Fizik Motoru:** Meteoroloji istasyonu ile tarla arasındaki rakım farkını baz alarak "Lapse Rate" hesaplaması yapar ve tarladaki gerçek sıcaklığı bulur.
3.  **Güvenlik Kalkanı:** Ani soğuk hava dalgalarına (Cold Fronts) karşı modele bir "Risk Toleransı" ekleyerek yanılma payını düşürür ve çiftçiye en kötü senaryoyu sunar.

## Temel Özellikler

* **Canlı Veri Entegrasyonu:** Meteostat API üzerinden günlük ve anlık meteorolojik veriler otomatik çekilir.
* **Derin Öğrenme Mimarisi:** TensorFlow ve Keras tabanlı özelleştirilmiş LSTM (Long Short-Term Memory) katmanları kullanılır.
* **Safety Mode (Güvenlik Kalkanı):** Kullanıcı, risk toleransını ayarlayarak modelin tahminlerini daha temkinli hale getirebilir.
* **İnteraktif Dashboard:** Streamlit ile geliştirilmiş arayüz sayesinde çiftçiler kod bilmeden analiz yapabilir, grafikleri inceleyebilir.
* **Don Türü Tespiti:** Sistem, sıcaklık ve nem dengesine göre "Beyaz Don" veya "Siyah Don" riskini ayırt edebilir.

## Kullanılan Teknolojiler

Bu proje, veri bilimi ve yapay zeka alanındaki endüstri standardı kütüphaneler kullanılarak geliştirilmiştir:

* **Python:** Projenin temel programlama dili.
* **TensorFlow & Keras:** LSTM modelinin eğitimi ve mimarisi.
* **Streamlit:** Web tabanlı kullanıcı arayüzü ve dashboard geliştirimi.
* **Pandas & NumPy:** Zaman serisi verilerinin işlenmesi ve matris operasyonları.
* **Scikit-Learn:** Veri ön işleme ve normalizasyon (MinMaxScaler).
* **Meteostat API:** İklim verilerinin çekilmesi.
* **Matplotlib:** Veri görselleştirme ve grafik çizimi.

## Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

1.  **Projeyi Klonlayın:**
    ```bash
    git clone [https://github.com/YusufKayace/AgroFrost.git](https://github.com/YusufKayace/AgroFrost.git)
    cd AgroFrost
    ```

2.  **Gereksinimleri Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Uygulamayı Başlatın:**
    ```bash
    python -m streamlit run app.py
    ```

## Proje Yapısı

* `app.py`: Streamlit web arayüzünün ana dosyası.
* `src/physics_engine.py`: Rakım farkı ve çiğ noktası hesaplamalarını yapan fizik motoru.
* `models/`: Eğitilmiş LSTM model dosyalarının (.h5) bulunduğu klasör.
* `requirements.txt`: Proje için gerekli kütüphane listesi.

---
**Geliştirici:** YUSUF TALHA KAYA