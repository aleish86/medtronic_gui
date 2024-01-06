import os
import datetime
import sys
import csv
import pandas as pd
import scipy
import easygui
from PySide6.QtWidgets import QVBoxLayout
from scipy import signal
import numpy as np
import collections
import mmt
from collections import deque
from . import *
from PySide6 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
import hdy
import matplotlib.pyplot as plt
import time
from statistics import mode

class MedtronicAnalysis(object):
    def __init__(self, laser_exp: object, parent: object = None) -> object:
        super().__init__(parent=parent)

        self.MedtronicGui = hdy.MedtronicGui

        self.laser_exp = laser_exp # type: hdy.LaserAnalysis1
        # Fixing and Correlating Data
        # Filtering ECGs
        bipecg_d = self.laser_exp.bipecg.data
        ecg_sos = scipy.signal.butter(5, (0.5, 25), 'band', fs=500, output='sos')
        self.bipecg_d = scipy.signal.sosfilt(ecg_sos, bipecg_d)
        self.ecg3_d = scipy.signal.sosfilt(ecg_sos, self.laser_exp.ecg3.data)
        self.laser_exp.ecg3.data = self.ecg3_d
        self.laser_exp.bipecg.data = self.bipecg_d

        # Fixing Voltage on leads
        self.laser_exp.ralead.data = self.laser_exp.ralead.data * 10
        self.laser_exp.rvbip.data = self.laser_exp.rvbip.data * 10
        self.laser_exp.rvshock.data = self.laser_exp.rvshock.data * 10
        self.laser_exp.lvlead.data = self.laser_exp.lvlead.data * 10

        self.laser_exp.pressure.data = self.laser_exp.pressure.data

        #     Correlate ECGs
        corr = scipy.signal.correlate(self.laser_exp.rvbip.data, self.laser_exp.ecg3.data, mode="full", method="fft")
        lags = scipy.signal.correlation_lags(self.laser_exp.rvbip.data.size, self.laser_exp.ecg3.data.size, mode="full")
        lag = lags[np.argmax(corr)]
        print(f"Best lag: {lag}")

        remove = np.arange(0, abs(lag), step=1)

        if lag < 0:
            print(f"Removing {len(remove)} samples from ECG Dataset")
            self.laser_exp.ecg3.data = np.delete(self.laser_exp.ecg3.data, remove, axis=0)
            self.laser_exp.bipecg.data = np.delete(self.laser_exp.bipecg.data, remove, axis=0)
            self.laser_exp.laser1.data = np.delete(self.laser_exp.laser1.data, remove, axis=0)
            self.laser_exp.laser2.data = np.delete(self.laser_exp.laser2.data, remove, axis=0)
            self.laser_exp.pressure.data = np.delete(self.laser_exp.pressure.data, remove, axis=0)

            self.laser_exp.rvbip.data = self.laser_exp.rvbip.data[0:self.laser_exp.ecg3.data.size]
            self.laser_exp.rvshock.data = self.laser_exp.rvshock.data[0:self.laser_exp.ecg3.data.size]
            self.laser_exp.ralead.data = self.laser_exp.ralead.data[0:self.laser_exp.ecg3.data.size]
            self.laser_exp.lvlead.data = self.laser_exp.lvlead.data[0:self.laser_exp.ecg3.data.size]

        else:
            print(f"Removing {len(remove)} samples from ICD Leads")
            self.laser_exp.rvbip.data = np.delete(self.laser_exp.rvbip.data, remove, axis=0)
            self.laser_exp.rvshock.data = np.delete(self.laser_exp.rvshock.data, remove, axis=0)
            self.laser_exp.ralead.data = np.delete(self.laser_exp.ralead.data, remove, axis=0)
            self.laser_exp.lvlead.data = np.delete(self.laser_exp.lvlead.data, remove, axis=0)

            self.laser_exp.ecg3.data = self.laser_exp.ecg3.data[0:self.laser_exp.rvbip.data.size]
            self.laser_exp.bipecg.data = self.laser_exp.bipecg.data[0:self.laser_exp.rvbip.data.size]
            self.laser_exp.laser1.data = self.laser_exp.laser1.data[0:self.laser_exp.rvbip.data.size]
            self.laser_exp.laser2.data = self.laser_exp.laser2.data[0:self.laser_exp.rvbip.data.size]
            self.laser_exp.pressure.data = self.laser_exp.pressure.data[0:self.laser_exp.rvbip.data.size]

        # Initialize Memory
        self.icd_memory = {}
        self.icd_memory['last_r_peak'] = 0
        self.icd_memory['rr_intervals'] = deque(maxlen=8)

        self.icd_memory['vt_counter'] = 0

        self.icd_memory['onset'] = deque(maxlen=8)
        self.icd_memory['stability'] = deque(maxlen=4)

        self.icd_memory['avg_interval_bin'] = deque(maxlen=4)
        self.icd_memory['nid_vtzone1'] = deque(maxlen=30)
        self.icd_memory['nid_vtzone2'] = deque(maxlen=30)
        self.icd_memory['nid_vf'] = deque(maxlen=30)
        self.icd_memory['nid_nsr'] = deque(maxlen=10)
        self.icd_memory['nid_discard'] = deque(maxlen=10)
        self.icd_memory['vf_check'] = deque(maxlen=8)
        self.icd_memory['median_rr'] = deque(maxlen=12)

        self.icd_memory['wavelet'] = deque(maxlen=8)  # use RV Shock lead

        self.icd_memory['final_rvpeaks'] = deque(maxlen=10000)
        self.icd_memory['rhythm_label'] = deque(maxlen=40)

        self.icd_memory['active_tachy'] = False

        finding_r_peaks(self)

    def finding_rv_peaks(self, data):

        rv_peaks = np.where(data > float(self.rv_sens_threshold.currentText()))
        rv_peaks = np.array(rv_peaks).flatten()
        print("rv_peaks: ", rv_peaks)

        # Rate
        for i, r_peak in enumerate(rv_peaks):

            # RR Interval
            rr_interval = r_peak - icd_memory['last_r_peak']

            # Ignore blanking period after VS
            ignore_blanking = np.where(r_peak < (icd_memory['last_r_peak'] + float(self.rv_sens_threshold.currentText())))
            print("ignore blanking: ", ignore_blanking)

            if i != 0 and ignore_blanking[0].size > 0:
                print("Post V-Sense Blanking at r_peak: " + str(r_peak))
                continue

            elif i == 0 and ignore_blanking[0].size > 0:
                print("Post V-Sense Blanking at r_peak: " + str(r_peak))
                icd_memory['last_r_peak'] = r_peak

                continue
            else:
                pass


            icd_memory['single_beat'] = deque(maxlen=1)
            # Find rough r-peak time range
            start = r_peak - 150
            print("start: ", start)
            end = r_peak + 150
            print("end: ", end)
            x_range = range(start, end)

            # Rough QRS Dataset
            single_beat = data[x_range]
            icd_memory['single_beat'].append(single_beat)
            # Max Voltage
            max_beat = np.max(single_beat)
            print("max beat: ", max_beat)
            # Min Voltage
            min_beat = np.min(single_beat)
            print("min beat: ", min_beat)

            # Amplitude
            amplitude = max_beat + abs(np.min(single_beat))
            print("amplitude: ", amplitude)
            # Rough time of max voltage
            max_x = single_beat.argmax() + start
            print("Maximum Index position: ", max_x)

            icd_memory['final_rvpeaks'].append(max_x)

            new_rr_interval = max_x - icd_memory['last_r_peak']
            icd_memory['rr_intervals'].append(new_rr_interval)
            print("icd_memory rr intervals: ", icd_memory['rr_intervals'])

            icd_memory['last_r_peak'] = r_peak
            print("last r peak: ", icd_memory['last_r_peak'])


            x_range_start = max_x - 100
            print("x range start", x_range_start)
            x_range_end = max_x + 100
            print("x range end", x_range_end)
            x_range_plot = range(x_range_start, x_range_end)
            single_beat_plot = data[x_range_start:x_range_end]

            mode_info = mode(single_beat_plot)
            print("mode: ", mode_info)
            upper_limit_mode = mode_info + 0.1
            lower_limit_mode = mode_info - 0.1
            loc1 = np.argmax(single_beat_plot > upper_limit_mode)
            loc2 = np.argmax(single_beat_plot < lower_limit_mode)
            print("loc1: ", loc1)
            print("loc2: ", loc2)
            if loc1 > loc2:
                qrs_onset = loc2 + x_range_start
                print("qrs onset: ", qrs_onset)

            else:
                qrs_onset = loc1 + x_range_start
                print("qrs onset: ", qrs_onset)

            loc_qrs_onset = np.where(x_range_plot == qrs_onset)
            print("loc qrs onset: ", loc_qrs_onset)
            min_mv = single_beat_plot[loc_qrs_onset]
            delta_time = int(max_x - qrs_onset)
            delta_mv_mode = float(max_beat - mode_info)
            delta_mv = float(max_beat - min_mv)
            print("delta mv: ", delta_mv)
            slew_rate_mode = round(delta_mv / delta_time, 4)
            slew_rate = round(delta_mv / delta_time, 4)
            print("slew rate: ", slew_rate)

            loc_qrs_onset = np.where(x_range_plot == qrs_onset)
            print("loc qrs onset: ", loc_qrs_onset)
            min_mv = single_beat_plot[loc_qrs_onset]
            delta_time = int(max_x - qrs_onset)
            delta_mv_mode = float(max_beat - mode_info)
            delta_mv = float(max_beat - min_mv)
            print("delta mv: ", delta_mv)
            slew_rate_mode = round(delta_mv / delta_time, 4)
            slew_rate = round(delta_mv / delta_time, 4)
            print("slew rate: ", slew_rate)

            plt.title("RV Bipolar: Single Beat")
            plt.plot(x_range_plot, single_beat_plot)
            plt.axvline(max_x, color='red', linestyle="-")
            plt.axvline(qrs_onset, color='red', linestyle="-")
            plt.axhline(max_beat, color='darkgrey', linestyle="--")
            plt.axhline(min_beat, color='darkgrey', linestyle='--')
            plt.axhline(upper_limit_mode, color='grey')
            plt.axhline(lower_limit_mode, color='grey')
            plt.annotate("Amplitude: " + str(round(amplitude, 2)), xy=(x_range_start, 0.7 * max_beat),
                         xytext=(x_range_start, 0.7 * max_beat))
            plt.annotate("Amplitude: " + str(round(amplitude, 2)), xy=(x_range_start, 0.7 * max_beat),
                         xytext=(x_range_start, 0.7 * max_beat))
            plt.annotate("Time to Peak: " + str(delta_time) + " ms", xy=(x_range_start, 0.6 * max_beat),
                         xytext=(x_range_start, 0.6 * max_beat))
            plt.annotate("Slew Rate: " + str(slew_rate) + " mV/ms", xy=(x_range_start, 0.5 * max_beat),
                         xytext=(x_range_start, 0.5 * max_beat))
            plt.annotate("Slew Rate (Mode): " + str(slew_rate_mode) + " mV/ms", xy=(x_range_start, 0.4 * max_beat),
                            xytext=(x_range_start, 0.4 * max_beat))
            plt.annotate("RR Interval: " + str(rr_interval) + " ms", xy=(x_range_start, 0.3* max_beat),
                         xytext=(x_range_start, 0.4 * max_beat))
            plt.show()

