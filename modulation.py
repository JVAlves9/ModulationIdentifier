import numpy as np
import matplotlib.pyplot as plt
from pyphysim.modulators import fundamental
from pyphysim.util.conversion import dB2Linear
from pyphysim.util.misc import randn_c

class DataTransmitionSimulator():
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
        self.__psk = fundamental.PSK(self.num_symbols)
        self.__qam = fundamental.QAM(self.num_symbols)

    def generate_data(self) -> np.ndarray :
        """Generate data to transmit

        Returns:
            np.ndarray: an array with data to transmit
        """        
        return np.random.randint(0, self.num_symbols, self.num_symbols_transmit)

    def modulate_data(self, data : np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        """Generate the modulated data for PSK and QAM

        Args:
            data (np.ndarray): an array with the data to modulate

        Returns:
            tuple[np.ndarray,np.ndarray]: (array with QAM modulated data, array with PSK modulated data)
        """                    
        psk_modulated_data = self.__psk.modulate(data)
        qam_modulated_data = self.__qam.modulate(data)

        return (qam_modulated_data, psk_modulated_data)

    def __awgn_noise(self, noise : int) -> np.ndarray :
        """Generate White Gaussian Noise

        Args:
            noise (int): noise level in dB

        Returns:
            np.ndarray: array with noisy data to be added with another array
        """    
        noise_power = 1 / dB2Linear(noise)
        n = randn_c(self.num_symbols_transmit)
        return n * np.sqrt(noise_power)

    def __phase_noise(self):
        return np.exp(1j * np.random.randn(self.num_symbols_transmit))
    
    def print_constellations(self):
        """Display the constellations used for both QAM and PSK
        """        
        print(self.__qam.symbols.real)
        plt.plot(self.__qam.symbols.real, self.__qam.symbols.imag, '.', label="QAM")
        plt.plot(self.__psk.symbols.real, self.__psk.symbols.imag, '.', label="PSK")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          fancybox=True, shadow=True, ncol=5)
        plt.grid(True)
        plt.show()
    
    def transmit_data(self, qam_modulated_data : np.ndarray, psk_modulated_data : np.ndarray, noise : int = 20) -> tuple [np.ndarray, np.ndarray]:
        """Transmit the generated data through a channel with AWGN

        Args:
            qam_modulated_data (np.ndarray): Data already modulated with QAM
            psk_modulated_data (np.ndarray): Data already modulated with PSK
            noise (int, optional): The noise to be applied to the data in dB. Defaults to 20dB.

        Returns:
            tuple [np.ndarray, np.ndarray]: (QAM ndarray data with noise, PSK ndarray with noise)
        """            
        channel_awg_noise = self.__awgn_noise(noise)
        noisy_qam = qam_modulated_data + channel_awg_noise
        noisy_psk = psk_modulated_data + channel_awg_noise

        return (noisy_qam, noisy_psk)
    
    def demodulate(self, qam_data : np.ndarray, psk_data : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Demodulated data for PSK and QAM

        Args:
            qam_data (np.ndarray): Data to be demodulated with QAM
            psk_data (np.ndarray): Data to be demodulated with PSK

        Returns:
            tuple[np.ndarray, np.ndarray]: (QAM data demodulated, PSK data demodulated)
        """           
        qam_demodulated_data = self.__qam.demodulate(qam_data)
        psk_demodulated_data = self.__psk.demodulate(psk_data)

        return (qam_demodulated_data, psk_demodulated_data)


    def symbol_error_rate(self, data, qam_demodulated_data, psk_demodulated_data):
        return (1 - sum(qam_demodulated_data == data) / self.num_symbols_transmit, 1 - sum(psk_demodulated_data == data) / self.num_symbols_transmit)
        # qam_error = sum(qam_demodulated_data != data)
        # psk_error = sum(psk_demodulated_data != data)

        # qam_simulate_results = SimulationResults()
        # psk_simulate_results = SimulationResults()

        # qam_simulate_results.add_new_result("symbol_error_rate", Result.RATIOTYPE, values=qam_error, total=self.num_symbols_transmit)
        # psk_simulate_results.add_new_result("symbol_error_rate", Result.RATIOTYPE, values=psk_error, total=self.num_symbols_transmit)
        # return (qam_simulate_results, psk_simulate_results)
    
    def simulate(self, num_rep : int = 5000, noise = 20):
        qam_ser = 0
        psk_ser = 0

        for rep in range(num_rep):
            data = self.generate_data()
            qam_mod_data, psk_mod_data = self.modulate_data(data)
            qam_data, psk_data = self.transmit_data(qam_mod_data, psk_mod_data, noise)
            qam_demo, psk_demo = self.demodulate(qam_data, psk_data)
            qam_result, psk_result = self.symbol_error_rate(data, qam_demo, psk_demo)
            qam_ser += qam_result
            psk_ser += psk_result

        return (qam_ser/num_rep, psk_ser/num_rep)
    
    def simulate_range_noise(self, initial_noise, final_noise, num_rep : int = 5000):
        qam_ser_values = []
        psk_ser_values = []

        for noise in range(initial_noise, final_noise):
            qam_ser, psk_ser = self.simulate(num_rep, noise)
            qam_ser_values.append(qam_ser)
            psk_ser_values.append(psk_ser)
        
        return (qam_ser_values, psk_ser_values)
    

    def ser_teoretical(self, noise : int = 20):
        return (self.__qam.calcTheoreticalSER(noise),self.__psk.calcTheoreticalSER(noise))
    
    def ser_teoretical_noise_range(self, initial_noise, final_noise):
        qam_ser_values = []
        psk_ser_values = []
        for noise in range(initial_noise, final_noise):
            qam_ser, psk_ser = self.ser_teoretical_value(noise)
            qam_ser_values.append(qam_ser)
            psk_ser_values.append(psk_ser)
        
        return (qam_ser_values,psk_ser_values)
    
    def ser_plot(self, qam_ser, qam_ser_teoretical, psk_ser, psk_ser_teoretical, inital_noise, final_noise):
        noise_list = [noise for noise in range(inital_noise, final_noise)]

        figure, axis = plt.subplots(1, 2)
        axis[0].plot(noise_list, qam_ser, label="QAM")
        axis[0].plot(noise_list, qam_ser_teoretical, '.', label="QAM Teoretical")
        axis[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          fancybox=True, shadow=True, ncol=5)
        axis[0].grid(True)

        axis[1].plot(noise_list, psk_ser, label="PSK")
        axis[1].plot(noise_list, psk_ser_teoretical, '.', label="PSK Teoretical")
        axis[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          fancybox=True, shadow=True, ncol=5)
        axis[1].grid(True)
        plt.show()


if __name__ == "__main__":
    cla = DataTransmitionSimulator(64,64)
    print(f"Simulated SER value : {cla.simulate()}\nTeoretical SER valeu : {cla.ser_teoretical_value()}")
    qam_ser, psk_ser = cla.simulate_range_noise(-5,15,2000)
    qam_ser_ter, psk_ser_ter = cla.ser_teoretical_value_noise_range(-5,15)
    cla.ser_plot(
        qam_ser,
        qam_ser_ter,
        psk_ser,
        psk_ser_ter,
        -5,
        15
    )
    # cla.print_constellations()
    
