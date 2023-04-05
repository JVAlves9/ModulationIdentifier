import numpy as np
import matplotlib.pyplot as plt
from pyphysim.modulators import fundamental

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
        self.__psk = fundamental.PSK(num_symbols)
        self.__qam = fundamental.QAM(num_symbols)
        self.__psk_modulated_data = None
        self.__qam_modulated_data = None


    def modulate_data(self) -> None:
        """Generate the modulated data
        """        
        self.__psk_modulated_data = self.__psk.modulate(self.data)
        self.__qam_modulated_data = self.__qam.modulate(self.data)

    # def add_noise()
    
    
    
    
