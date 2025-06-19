#Wavelet
import pywt

import numpy as np

import matplotlib.pyplot as plt
from collections import deque
import pywt.data

import matplotlib.pyplot as plt
import numpy as np

import hdy
import mmt
import pyqtgraph


from collections import deque
import time

FILE_DIR = "/Users/amiyazawa/Library/CloudStorage/Dropbox/01 Research/02 Current Projects/02 AAD/Data/A08/Haem"
FILE_FN = "A08_LVP180_15_07_2022_155306_.zip"
#"AAD03_VVI180_08_11_2021_114608_.zip"

daq_data = hdy.DAQ_File(zip_dir=FILE_DIR, zip_fn=FILE_FN)

print(daq_data)
ecg_detector = mmt.ecg.ECGDetectors(sampling_frequency=1000)

r_peaks = ecg_detector.pan_tompkins_detector(daq_data.plethg) # change to bipolar ecg lead
print(r_peaks)
print(type(r_peaks))

icd_memory = {}
icd_memory['last_r_peak'] = 0
icd_memory['rr_intervals'] = deque(maxlen=40)
icd_memory['onset'] = deque(maxlen=8)
icd_memory['stability'] = deque(maxlen=4)
icd_memory['active_tachy'] = False
icd_memory['wavelet'] = deque(maxlen=8)
icd_memory['wavelet_coeff'] = deque(maxlen=8)


# def check_last_num_identical(deque_items, num):
#     if len(deque_items) >= num:
#         last_num = list(deque_items)[-num:]
#         if len(set(last_num)) == 1 and last_num[0] == "TS":
#             return True
#         else:
#             return False

'''
def extract_beat(signal, qrs_pos, half_win_ms=100, fs=1000, start_beat=100, end_beat=300):
    signal = np.array(signal)
    print(signal.shape)
    beat =  signal[qrs_pos-half_win_ms:qrs_pos+half_win_ms]
    beat_array=np.pad(beat,half_win_ms*2,mode='constant')
    dwt = np.zeros(len(beat))
    win = fs*win_msec//1000 
    cA, cD = pywt.dwt(beat, 'haar')
    maxcoeff_ten = np.argpartition(cA, -10)[-10:]
    maxcoeff_ten = cA[maxcoeff_ten]
    dwt[start_beat-half_win_ms:start_beat+half_win_ms]=
    return beat_array, label  '''
'''
'''

# qrs_pos = [1,100,3500]
# plt.plot(x)
# for q in qrs_pos:
#     y,label = extract_beat(x,q)
#     plt.figure()
#     plt.plot(y)
#     plt.plot(label)

for r_peak in r_peaks:
    print(r_peak)
    rr_interval = r_peak - icd_memory['last_r_peak']
 #   single_wavelet = (r_peak + 100) - (icd_memory['last_r_peak'] - 100)
    icd_memory['last_r_peak'] = r_peak
    icd_memory['rr_intervals'].append(rr_interval)
    print(icd_memory['rr_intervals'])

    wavelet = daq_data.plethg[r_peak-100: r_peak+100]

    icd_memory['wavelet'].append(wavelet) #Creating set of 8 wavelets

    # check_last_num_identical(deque_items=icd_memory['wavelet'], num=2)

    # To check if 6/8 coefficients have > 70% match

#    print(wavelet)
    cA, cD = pywt.dwt(wavelet, 'haar')
    cA4, cD4, cD3, cD2, cD1 = pywt.wavedec(wavelet, 'haar', level=4)
    maxcoeff_ten = np.argpartition(cD4, -10)[-10:]
    maxcoeff_ten = cD4[maxcoeff_ten]
    print('coeff', maxcoeff_ten)

#    icd_memory['wavelet_coeff'].append(maxcoeff_ten)

    # inv_dwt = pywt.waverec(maxcoeff_ten, wavelet=wavelet,  mode='zero', axis=-1)

    #   cA, cD = pywt.dwt(six_beat_wavelets, 'haar')
    # cA: Approximation coefficients
    # cD: Detail coefficients

    # 1 wavelet, 1 beat
    # custom_xlim1 = (r_peak-100, r_peak+100)

    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, gridspec_kw={'width_ratios': [1, 3, 3, 3], 'height_ratios': [0.5]})
    # plt.xlim(custom_xlim1)
    # #    fig.suptitle('R wave and wavelets')
    # # plt.title.set_text('Wavelets')
    # plt.plot(daq_data.plethg)
    # plt.plot(r_peaks, daq_data.plethg[r_peaks], "ro")
    # plt.plot(wavelet)
    # plt.plot(np.round(cD4, 2) * cA4)
    # plt.plot(np.round(maxcoeff_ten))
    #
    # plt.show()

    custom_xlim = (six_beat_wavelets[0] - 100, six_beat_wavelets[5] + 100)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, gridspec_kw = {'width_ratios': [1, 3, 3,3], 'height_ratios': [0.5]})
    plt.setp(ax1, xlim=custom_xlim1)
#    fig.suptitle('R wave and wavelets')
    ax1.title.set_text('ECG Beat')
    ax2.title.set_text('Wavelet')
    ax3.title.set_text('Coefficients')
    ax1.plot(daq_data.ecg)
    ax1.plot(r_peaks, daq_data.ecg[r_peaks], "ro")
    ax2.plot(wavelet)
    ax3.plot(np.round(cD4,2)*cA4)
    ax4.plot(np.round(maxcoeff_ten))

    plt.show()

    print('cd4: ', cD4)

   #
   # if self.icd_memory['rhythm_label'] =='VS':
   #          # Perform Haar wavelet transform on each normal waveform
   #          wavelet_coeffs = []
   #          for waveform in normal_waveforms:
   #              coeffs = pywt.dwt(waveform, 'haar')  # Apply Haar wavelet transform
   #              wavelet_coeffs.append(coeffs[0])  # Store the approximation coefficients
   #
   #          # Calculate the average wavelet coefficients
   #          self.avg_template_coeffs = np.mean(wavelet_coeffs, axis=0)
   #      else:
   #          print("Can't analyse now")
   #          pass
   #
   #  def plot_wavelet(waveform, title, color, plot_widget):
   #      # title = "Wavelet Template"
   #      coeffs = pywt.dwt(waveform, 'haar')  # Compute wavelet coefficients
   #      cA, cD = coeffs  # Approximation (cA) and detail (cD) coefficients
   #
   #      # Plot Approximation (cA)
   #      plot_widget.setData(cA, pen=color, name=f'{title} - Approximation (cA)')
   #
   #      # Plot Detail (cD)
   #      plot_widget.setData(cD, pen=color, style=QtCore.Qt.DashLine, name=f'{title} - Detail (cD)')
   #
