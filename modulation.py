import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from pyphysim.modulators import fundamental
from pyphysim.util.conversion import dB2Linear
from pyphysim.util.misc import randn_c

class DataTransmissionSimulator():
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

    def generate_data(self) -> ndarray :
        """Generate data to transmit

        Returns:
            ndarray: an array with data to transmit
        """        
        return np.random.randint(0, self.num_symbols, self.num_symbols_transmit)

    def modulate_data(self, data : ndarray) -> tuple[ndarray,ndarray]:
        """Generate the modulated data for PSK and QAM

        Args:
            data (ndarray): an array with the data to modulate

        Returns:
            tuple[ndarray,ndarray]: (array with QAM modulated data, array with PSK modulated data)
        """                    
        psk_modulated_data = self.__psk.modulate(data)
        qam_modulated_data = self.__qam.modulate(data)

        return (qam_modulated_data, psk_modulated_data)

    def __awgn_noise(self, noise : int) -> ndarray :
        """Generate White Gaussian Noise

        Args:
            noise (int): noise level in dB

        Returns:
            ndarray: array with noisy data to be added with another array
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
    
    def transmit_data(self, qam_modulated_data : ndarray, psk_modulated_data : ndarray, noise : int = 20) -> tuple [ndarray, ndarray]:
        """Transmit the generated data through a channel with AWGN

        Args:
            qam_modulated_data (ndarray): Data already modulated with QAM
            psk_modulated_data (ndarray): Data already modulated with PSK
            noise (int, optional): The noise to be applied to the data in dB. Defaults to 20dB.

        Returns:
            tuple [ndarray, ndarray]: (QAM ndarray data with noise, PSK ndarray with noise)
        """            
        channel_awg_noise = self.__awgn_noise(noise)
        noisy_qam = qam_modulated_data + channel_awg_noise
        noisy_psk = psk_modulated_data + channel_awg_noise

        return (noisy_qam, noisy_psk)
    
    def demodulate(self, qam_data : ndarray, psk_data : ndarray) -> tuple[ndarray, ndarray]:
        """Demodulated data for PSK and QAM

        Args:
            qam_data (ndarray): Data to be demodulated with QAM
            psk_data (ndarray): Data to be demodulated with PSK

        Returns:
            tuple[ndarray, ndarray]: (QAM data demodulated, PSK data demodulated)
        """           
        qam_demodulated_data = self.__qam.demodulate(qam_data)
        psk_demodulated_data = self.__psk.demodulate(psk_data)

        return (qam_demodulated_data, psk_demodulated_data)


    def symbol_error_rate(self, data : ndarray, qam_demodulated_data : ndarray, psk_demodulated_data : ndarray) -> tuple[float,float]:
        """Calculate the Symbol Error Rate for QAM and PSK

        Args:
            data (ndarray): The original data
            qam_demodulated_data (ndarray): QAM demodulated data
            psk_demodulated_data (ndarray): PSK demodulated data

        Returns:
            tuple[float, float]: (SER for QAM, SER for PSK)
        """                
        return (1 - sum(qam_demodulated_data == data) / self.num_symbols_transmit, 1 - sum(psk_demodulated_data == data) / self.num_symbols_transmit)
        # qam_error = sum(qam_demodulated_data != data)
        # psk_error = sum(psk_demodulated_data != data)

        # qam_simulate_results = SimulationResults()
        # psk_simulate_results = SimulationResults()

        # qam_simulate_results.add_new_result("symbol_error_rate", Result.RATIOTYPE, values=qam_error, total=self.num_symbols_transmit)
        # psk_simulate_results.add_new_result("symbol_error_rate", Result.RATIOTYPE, values=psk_error, total=self.num_symbols_transmit)
        # return (qam_simulate_results, psk_simulate_results)
    
    def simulate(self, num_rep : int = 5000, noise = 20) -> tuple[float,float]:
        """Simulate multiple transmission

        Args:
            num_rep (int, optional): Number of transmissions to simulate. Defaults to 5000.
            noise (int, optional): Noise in dB. Defaults to 20dB.

        Returns:
            tuple[float, float]: (average SER for QAM, average SER for PSK)
        """          
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
    
    def simulate_range_noise(self, initial_noise : int, final_noise : int, num_rep : int = 5000) -> tuple[list[float],list[float]]:
        """Simulate multiple transmissions with a range of noise values

        Args:
            initial_noise (int): Minimum accepted noise value
            final_noise (int): Maximum accepted noise value
            num_rep (int, optional): Number of repetitions. Defaults to 5000.

        Returns:
            tuple[list[float], list[float]]: (list of SER values for each noise value for QAM, list of SER values for each noise value for PSK)
        """        
        qam_ser_values = []
        psk_ser_values = []

        for noise in range(initial_noise, final_noise):
            qam_ser, psk_ser = self.simulate(num_rep, noise)
            qam_ser_values.append(qam_ser)
            psk_ser_values.append(psk_ser)
        
        return (qam_ser_values, psk_ser_values)
    

    def ser_theoretical(self, noise : int = 20) -> tuple[float, float]:
        """Generate theoretical SER values

        Args:
            noise (int, optional): Noise in dB. Defaults to 20dB.

        Returns:
            tuple[float, float]: (SER value for QAM, SER value for PSK)
        """        
        return (self.__qam.calcTheoreticalSER(noise),self.__psk.calcTheoreticalSER(noise))
    
    def ser_theoretical_noise_range(self, initial_noise : int, final_noise : int) -> tuple[list[float],list[float]]:
        """Generate a list with theoretical SER values for QAM and PSK

        Args:
            initial_noise (int): Minimum accepted noise value
            final_noise (int): Maximum accepted noise value

        Returns:
            tuple[list[float], list[float]]: (list of SER values for each noise value for QAM, list of SER values for each noise value for PSK)
        """        
        qam_ser_values = []
        psk_ser_values = []
        for noise in range(initial_noise, final_noise):
            qam_ser, psk_ser = self.ser_theoretical(noise)
            qam_ser_values.append(qam_ser)
            psk_ser_values.append(psk_ser)
        
        return (qam_ser_values,psk_ser_values)
    
    def ser_plot(self, qam_ser:list[float], qam_ser_theoretical:list[float], psk_ser:list[float], psk_ser_theoretical:list[float], inital_noise:int, final_noise:int):
        """Plot the SER values comparing the theoretical with the simulated one

        Args:
            qam_ser (list[float]): List SER for the QAM simulation
            qam_ser_theoretical (list[float]): List theoretical SER values for QAM
            psk_ser (list[float]): List SER for the PSK simulation
            psk_ser_theoretical (list[float]): List theoretical SER values for PSK
            initial_noise (int): Minimum accepted noise value
            final_noise (int): Maximum accepted noise value
        """        
        
        noise_list = [noise for noise in range(inital_noise, final_noise)]

        figure, axis = plt.subplots(1, 2)
        axis[0].plot(noise_list, qam_ser, label="QAM")
        axis[0].plot(noise_list, qam_ser_theoretical, '.', label="QAM theoretical")
        axis[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          fancybox=True, shadow=True, ncol=5)
        axis[0].grid(True)

        axis[1].plot(noise_list, psk_ser, label="PSK")
        axis[1].plot(noise_list, psk_ser_theoretical, '.', label="PSK theoretical")
        axis[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          fancybox=True, shadow=True, ncol=5)
        axis[1].grid(True)
        plt.show()


if __name__ == "__main__":
    cla = DataTransmissionSimulator(64,64)
    print(f"Simulated SER value : {cla.simulate()}\ntheoretical SER value : {cla.ser_theoretical()}")
    qam_ser, psk_ser = cla.simulate_range_noise(-5,15,2000)
    qam_ser_ter, psk_ser_ter = cla.ser_theoretical_noise_range(-5,15)
    cla.ser_plot(
        qam_ser,
        qam_ser_ter,
        psk_ser,
        psk_ser_ter,
        -5,
        15
    )
    # cla.print_constellations()
    
