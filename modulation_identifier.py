import dill, os, random
import pywt, numpy as np
from scipy.signal import medfilt
from scipy.stats import norm


from modulation import DataTransmitionSimulator as Data

DATA_FOLDER_NAME = "data"
TRAIN_FILE = f"{DATA_FOLDER_NAME}/train_data.pkl"
TEST_FILE = f"{DATA_FOLDER_NAME}/test_data.pkl"
THRESHOLD_FILE = f"{DATA_FOLDER_NAME}/thresholds.pkl"

SYMBOL_NUM = 16
SYMBOL_NUM_TRANSMIT = int(1e3)
DATASET_NUM = 500
THRESHOLD_LIST_SIZE = 300

def amplitude_normalization(data):
    return data / np.linalg.norm(data)

def haar_wavelet_transform(data):
    return pywt.dwt(data,"haar")

def median_filter(data):
    return medfilt(np.real(data),kernel_size=5)

def variance(data):
    return np.var(data)

def probability_density(data):
    return norm.pdf(data)

def threshold_calculation(data):
    return

def branch_with_normalization(data):
    normalized_data = amplitude_normalization(data)
    return branch_without_normalization(normalized_data)

def branch_without_normalization(data):
    data_transformed_ca, data_transformed_cd = haar_wavelet_transform(data)

    qam_filtered = median_filter(data_transformed_ca)

    return variance(qam_filtered)

def identifier(data,threshold,threshold_normalized):
    var_norm = branch_with_normalization(data)
    var = branch_without_normalization(data)

    print(var,"\n\r",var_norm)

    if var < threshold:
        return "PSK"
    elif var_norm > threshold_normalized:
        return "FSK"
    else:
        return "QAM"

def generate_datasets(num_symbols,num_symbols_transmit,size):
    temp = []
    for i in range(size):
        data = Data(num_symbols,int(num_symbols_transmit))
        temp.append((data,data.generate_data()))
    
    return temp

def generate_random_thresholds(size):
    return [random.random() for x in range(size)]


def save_object(object, path):
    save_path = os.path.abspath(path)
    folder_path = "/".join(save_path.split("/")[:-1])
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    with open(path,"wb") as datase_file:
        dill.dump(object,datase_file)

def read_object(path):
    with open(path,"rb") as dataset_file:
        dataset = dill.load(dataset_file)
    return dataset

def main():
    transmition = Data(16, int(300))
    data = transmition.generate_data()
    qam_modulated, psk_modulated = transmition.modulate_data(data)
    qam_modulated, psk_modulated = transmition.transmit_data(qam_modulated, psk_modulated, 15)

    identifier(qam_modulated, 0.5, 0.0002)
    identifier(psk_modulated, 0.5, 0.0002)

    if os.path.exists(os.path.abspath(TRAIN_FILE)):
        train_data = read_object(TRAIN_FILE)
    else:
        train_data = generate_datasets(SYMBOL_NUM,SYMBOL_NUM_TRANSMIT,DATASET_NUM)
        save_object(train_data, TRAIN_FILE)
    
    if os.path.exists(os.path.abspath(TEST_FILE)):
        test_data = read_object(TEST_FILE)
    else:
        test_data = generate_datasets(SYMBOL_NUM,SYMBOL_NUM_TRANSMIT,DATASET_NUM)
        save_object(test_data, TEST_FILE)

    if os.path.exists(os.path.abspath(THRESHOLD_FILE)):
        thresholds = read_object(THRESHOLD_FILE)
    else:
        thresholds = generate_random_thresholds(THRESHOLD_LIST_SIZE)
        save_object(thresholds, THRESHOLD_FILE)

    
    
if __name__ == "__main__":
    main()