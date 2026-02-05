import numpy as np

def calculate_dew_point(temp, humidity):
    '''Magnus-Tetens Formülü ile Çiğ Noktası Hesabı'''
    A = 17.27
    B = 237.7
    humidity = max(humidity, 1.0) 
    alpha = ((A * temp) / (B + temp)) + np.log(humidity / 100.0)
    dew_point = (B * alpha) / (A - alpha)
    return dew_point

def apply_lapse_rate(base_temp, base_altitude, target_altitude):
    '''Rakım farkına göre sıcaklık düzeltmesi'''
    diff = target_altitude - base_altitude
    correction = (diff / 100.0) * 0.65
    return base_temp - correction
