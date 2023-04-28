import dill, os, random, threading
from queue import Queue
import pywt, numpy as np
from scipy.signal import medfilt
from scipy.stats import norm

from modulation import DataTransmissionSimulator as Data

DATA_FOLDER_NAME = "data"
TRAIN_FILE = f"{DATA_FOLDER_NAME}/train_data.pkl"
TEST_FILE = f"{DATA_FOLDER_NAME}/test_data.pkl"
THRESHOLD_FILE = f"{DATA_FOLDER_NAME}/thresholds.pkl"

SYMBOL_NUM = 16
SYMBOL_NUM_TRANSMIT = int(1e3)
DATASET_NUM = 500
THRESHOLD_LIST_SIZE = 300

def amplitude_normalization(data):
    return data/np.linalg.norm(data)

def haar_wavelet_transform(data):
    return pywt.dwt(data, "haar")

def median_filter(data):
    return medfilt(np.real(data), kernel_size=5)

def __calculate_threshold(data, attempt, q : Queue):
    threshold_candidates = {}
    ran_num = random.uniform(0.0, 0.05)
    for thr_acc in [1 * ran_num, -1*ran_num]:
        thr_ini = branch_without_normalization(data[attempt][0]) + thr_acc
        threshold_candidates[thr_ini] = 0

        for mod in data:
            var = branch_without_normalization(mod[0])
            
            if var < thr_ini:
                threshold_candidates[thr_ini] += 1 if "PSK" == mod[1] else 0
            else:
                threshold_candidates[thr_ini] += 1 if "QAM" == mod[1] else 0
            
    q.put(threshold_candidates)

def threshold_calculation(num_sym, num_sym_transmit, num_gen_data):
    attempts = num_gen_data
    transmission = Data(num_sym, num_sym_transmit)
    data = []
    threshold_candidates = {}
    threads = []
    q = Queue()
    
    for num_gen in range(num_gen_data):
        data_trans = transmission.generate_data()
        qam_modulated, psk_modulated = transmission.modulate_data(data_trans)
        qam_modulated, psk_modulated = transmission.transmit_data(qam_modulated, psk_modulated, 15)
        data.append((qam_modulated, "QAM"))
        data.append((psk_modulated, "PSK"))
    
    for attempt in range(attempts):
        thread = threading.Thread(target=__calculate_threshold, args=(data.copy(), attempt, q))
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
    
    for i in range(q.qsize()):
        threshold_candidates.update(q.get())

    max_value = max(threshold_candidates.values())
    return list(filter(lambda x: x[1]==max_value, list(threshold_candidates.items())))[0]

def variance(data):
    return np.var(data)

def probability_density(data):
    return norm.pdf(data)

def branch_with_normalization(data):
    normalized_data = amplitude_normalization(data)
    return branch_without_normalization(normalized_data)

def branch_without_normalization(data):
    data_transformed_ca, data_transformed_cd = haar_wavelet_transform(data)
    qam_filtered = median_filter(data_transformed_ca)
    return variance(qam_filtered)

def identifier(data, threshold, threshold_normalized):
    var = branch_without_normalization(data)

    if var < threshold:
        return "PSK"
    else:
        return "QAM"

def generate_datasets(num_symbols, num_symbols_transmit, size):
    temp = []
    for i in range(size):
        data = Data(num_symbols, int(num_symbols_transmit))
        temp.append((data, data.generate_data()))

    return temp

def generate_random_thresholds(size):
    return [random.random() for x in range(size)]

def save_object(object, path):
    save_path = os.path.abspath(path)
    folder_path = "/".join(save_path.split("/")[:-1])
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    with open(path, "wb") as datase_file:
        dill.dump(object, datase_file)

def read_object(path):
    with open(path,"rb") as dataset_file:
        dataset = dill.load(dataset_file)
    return dataset

def main():
    transmission = Data(16, int(300))
    data = transmission.generate_data()
    qam_modulated, psk_modulated = transmission.modulate_data(data)
    qam_modulated, psk_modulated = transmission.transmit_data(qam_modulated, psk_modulated, 15)

    threshold, threshold_normalized = threshold_calculation(psk_modulated,16)
    r1 = identifier(qam_modulated, threshold, threshold_normalized)

    r2 = identifier(psk_modulated, threshold, threshold_normalized)

    print(r1)
    print(r2)
    print(f"th1 : {threshold}, th2 : {threshold_normalized}")

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
        thresholds = []
        for data in train_data:
            threshold, threshold_normalized = threshold_calculation(data[1])
            thresholds.append((threshold, threshold_normalized))
        save_object(thresholds, THRESHOLD_FILE)

    
    
if __name__ == "__main__":
    # main()
    th = threshold_calculation(16, int(300), 1000)
    print(th)