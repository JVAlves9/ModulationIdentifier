import pywt

from modulation import DataTransmitionSimulator as Data

def main():
    transmition = Data(64, int(1e3))
    data = transmition.generate_data()
    qam_modulated, psk_modulated = transmition.modulate_data(data)

    pywt.cwt()

print(pywt.wavelist())