import hdy
import mmt
import numpy as np
import matplotlib.pyplot as plt
import pywt
from PySide6 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg

FILE_DIR = "/Users/amiyazawa/Library/CloudStorage/Dropbox/01 Research/02 Current Projects/02 AAD/Data/A03/Haem"
FILE_FN = "A03_VVI180_08_11_2021_114608_.zip"

daq_data = hdy.DAQ_File(zip_dir=FILE_DIR, zip_fn=FILE_FN)

print(daq_data)
ecg_detector = mmt.ecg.ECGDetectors(sampling_frequency=1000)

r_peaks = ecg_detector.pan_tompkins_detector(daq_data.ecg) # change to bipolar ecg lead
a_peaks = ecg_detector.pan_tompkins_detector(daq_data.boxb) # change to ra lead
rv_peaks = ecg_detector.pan_tompkins_detector(daq_data.bpao) # change to rv bipolar lead
shock_peaks = ecg_detector.pan_tompkins_detector(daq_data.plethg) # change to shock lead

def bipegm(self):
    """Bipolar EGM algorithm."""
    wavelet = 'haar'
    level = 4
    threshold = self.icd_mdt_parameters['rvst_value']
    coeffs = pywt.wavedec(daq_data.bpao, wavelet, level=level)
    threshold_coeffs = [c * (c > threshold) for c in coeffs]

    denoised_signal = pywt.waverec(threshold_coeffs, wavelet)


    binary = np.zeros_like(self.sensing.sqrect_rvbip)
    for i, value in enumerate(self.sensing.sqrect_rvbip):
        if value > 100:
            self.binary[i - 1] = 100

    return denoised_signal, binary

denoised_signal, binary = bipegm(daq_data)
plt.plot(denoised_signal, color='black')
plt.plot(self.sensing.sqrect_rvbip, color='red')
plt.plot(self.sensing.rvbip_data, color='green')
plt.plot(self.binary, color='black')
plt.show()



