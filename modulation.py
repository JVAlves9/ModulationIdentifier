import numpy as np
import matplotlib.pyplot as plt
from pyphysim.modulators import fundamental
from pyphysim.util.conversion import dB2Linear
from pyphysim.util.misc import pretty_time, randn_c

class Data:
    """Class used to generate all data to transmit, as well as  the modulation objects
    """    
    def __init__(self, num_symbols : int, num_symbols_transmit : int) -> None:
        """

        Args:
            num_symbols (int): Number of symbols supported, 256, for example, would mean a 256-QAM and 256-PSK
            num_symbols_transmit (int): How much data should be created to transmit
        """             
        self.num_symbols = num_symbols
        self.num_symbols_transmit = num_symbols_transmit
        self.data = np.random.randint(0, self.num_symbols, self.num_symbols_transmit)
        self.__psk = fundamental.PSK(self.num_symbols)
        self.__qam = fundamental.QAM(self.num_symbols)
        self.__psk_modulated_data = None
        self.__qam_modulated_data = None
        self.__psk_demodulated_data = None
        self.__qam_demodulated_data = None


    def modulate_data(self) -> None:
        """Generate the modulated data for PSK and qam
        """        
        self.__psk_modulated_data = self.__psk.modulate(self.data)
        self.__qam_modulated_data = self.__qam.modulate(self.data)

    def __awgn_noise(self, noise : int):
        noise_power = 1 / dB2Linear(noise)
        n = randn_c(self.num_symbols_transmit)
        return n * np.sqrt(noise_power)

    def __phase_noise(self):
        return np.exp(1j * np.random.randn(self.num_symbols_transmit))
    
    def print_constellations(self):
        plt.plot(self.__qam.symbols.real, self.__qam.symbols.imag, '.', label="QAM")
        plt.plot(self.__psk.symbols.real, self.__psk.symbols.imag, '.', label="PSK")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          fancybox=True, shadow=True, ncol=5)
        plt.grid(True)
        
        plt.show()
    
    def transmit_data(self, noise : int = 20):
        """Transmit the generated data through a channel with AWGN

        Args:
            noise (int, optional): The noise to be applied to the data in dB. Defaults to 20dB.
        """        
        channel_awg_noise = self.__awgn_noise(noise)
        noisy_qam = self.__qam_modulated_data + channel_awg_noise
        noisy_psk = self.__psk_modulated_data + channel_awg_noise

        self.__qam_demodulated_data = self.__qam.demodulate(noisy_qam)
        self.__psk_demodulated_data = self.__psk.demodulate(noisy_psk)



        # figure, axis = plt.subplots(1, 2)
        
        # axis[0].plot(self.__qam_modulated_data.real, self.__qam_modulated_data.imag, '.')
        # axis[0].grid(True)

        # axis[1].plot(np.real(noisy_qam),np.imag(noisy_qam), '.')
        # axis[1].grid(True)
        # plt.show()

if __name__ == "__main__":
    cla = Data(64,32)
    cla.modulate_data()
    cla.transmit_data()
    # cla.print_constellations()
    
