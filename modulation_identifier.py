import pywt, numpy as np
from scipy.signal import medfilt
from scipy.stats import norm


from modulation import DataTransmitionSimulator as Data

def amplitude_normalization(data):
    return data / np.linalg.norm(data)

def haar_wavelet_transform(data):
    return pywt.dwt(data,"haar")

def median_filter(data):
    return medfilt(data)

def variance(data):
    return np.var(data)

def probability_density(data):
    return norm.pdf(data)

def threshold_calculation(data):
    return
 
def main():
    transmition = Data(16, int(300))
    data = transmition.generate_data()
    qam_modulated, psk_modulated = transmition.modulate_data(data)
    qam_modulated, psk_modulated = transmition.transmit_data(qam_modulated, psk_modulated, 20)
    
    pdf = probability_density(qam_modulated)

    # print(pdf)

    qam_normalized = amplitude_normalization(qam_modulated)
    psk_normalized = amplitude_normalization(psk_modulated)

    qam_transformed_ca, qam_transformed_cd = haar_wavelet_transform(qam_normalized)
    psk_transformed_ca, psk_transformed_cd = haar_wavelet_transform(psk_normalized)

    qam_filtered = median_filter(qam_transformed_ca)
    psk_filtered = median_filter(psk_transformed_ca)

    qam_var_1 = variance(qam_filtered)
    psk_var_1 = variance(psk_filtered)

    qam_transformed_ca, qam_transformed_cd = haar_wavelet_transform(qam_modulated)
    psk_transformed_ca, psk_transformed_cd = haar_wavelet_transform(psk_modulated)

    qam_filtered = median_filter(qam_transformed_ca)
    psk_filtered = median_filter(psk_transformed_ca)

    qam_var_2 = variance(qam_filtered)
    psk_var_2 = variance(psk_filtered)

    print(f"{qam_var_1}, {qam_var_2}")
    print(f"{psk_var_1}, {psk_var_2}")

    if qam_var_1 < 0 and qam_var_2>0:
        return True
    else:
        return False

print(main())