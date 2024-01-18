import scipy
from scipy import signal
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import hdy

class sensing(object):
    def __init__(self, laser_exp:object, parent: object = None) -> object:
        self.laser_exp = laser_exp

        self.used_signals = {}
        for k, v in self.laser_exp.data_source.items():
            if v != 'blank':
                self.used_signals[k] = v
            else:
                pass

        # Fixing and Correlating Data
        self.fix_lag()
        self.resample_data(1000, 512)
        self.ecg_filter()
        # Filtering ECGs
        self.rectifier()
        self.sq_rectifier()
        self.derivatives()
        self.icd_memory = {}
        self.icd_memory['low_sens'] = deque(maxlen=16)
        self.icd_memory['high_sens'] = deque(maxlen=36)
        self.ecg_peaks = []
        self.ecg3_peaks = []
        self.ralead_peaks = []
        self.rvbip_peaks = []
        self.rvshock_peaks = []
        self.lvlead_peaks = []
        self.laser1_peaks = []
        self.laser2_peaks = []

        self.filt_ecg_peaks = []
        self.filt_ecg3_peaks = []
        self.filt_ralead_peaks = []
        self.filt_rvbip_peaks = []
        self.filt_rvshock_peaks = []
        self.filt_lvlead_peaks = []
        self.filt_laser1_peaks = []
        self.filt_laser2_peaks = []


        max_peaks, peak_vals, rr_list, self.icd_memory['low_sens'], self.icd_memory['high_sens'] = self.find_peaks(self.rvbip_rect, 120, 0.3, 75, 512, 1000)
        self.zero_crossings(max_peaks)
        self.fix_lag()
        # self.peak_adaptive_threshold(ecg_data, window_size = 75, threshold = rvst_value, pvsb_value, factor = 0.6)
        print("Sensing Class Initialized")
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)
    def reject_outliers(self, data, m=2):
        return data[abs(data - np.mean(data)) < m * np.std(data)]

    def fix_lag(self):
        #     Correlate ECGs
        # try:
        #     corr = scipy.signal.correlate(self.resampled_shock_data, self.resampled_ecg_data, mode="full",
        #                                   method="fft")
        #     lags = scipy.signal.correlation_lags(len(resampled_shock_data), len(self.resampled_ecg_data),
        #                                      mode="full")
        # except:
        #     corr = scipy.signal.correlate(self.resampled_rv_data, self.resampled_ecg_data, mode="full",
        #                                   method="fft")
        #     lags = scipy.signal.correlation_lags(len(resampled_rv_data), len(self.resampled_ecg_data),
        #                                      mode="full")

        if 'ECG' in self.used_signals.keys():
            self.laser_exp.ecg.calc_ecg_peaks()
            num_peaks = min(10, len(self.laser_exp.ecg.peaks_sample))
            ecg_peaks = np.array([self.laser_exp.ecg.peaks_sample[0:num_peaks]])
            ecg_peak_vals = np.array([self.laser_exp.ecg.data[self.laser_exp.ecg.peaks_sample[0:num_peaks]]])
        else:
            self.laser_exp.ecg3.calc_ecg_peaks()
            num_peaks = min(10, len(self.laser_exp.ecg3.peaks_sample))
            ecg_peaks = np.array([self.laser_exp.ecg3.peaks_sample[0:num_peaks]])
            ecg_peak_vals = np.array([self.laser_exp.ecg3.data[self.laser_exp.ecg3.peaks_sample[0:num_peaks]]])
        if 'RVbip' in self.used_signals.keys():
            self.laser_exp.rvbip.calc_ecg_peaks()
            egm_signal = self.laser_exp.rvbip.data
            num_peaks = min(10, len(self.laser_exp.rvbip.peaks_sample))
            egm_peaks = np.array([self.laser_exp.rvbip.peaks_sample[0:num_peaks]])
            egm_peak_vals = np.array([self.laser_exp.rvbip.data[self.laser_exp.rvbip.peaks_sample[0:num_peaks]]])
        else:
            self.laser_exp.rvshock.calc_ecg_peaks()
            egm_signal = self.laser_exp.rvshock.data
            num_peaks = min(10, len(self.laser_exp.rvshock.peaks_sample))
            egm_peaks = np.array([self.laser_exp.rvshock.peaks_sample[0:num_peaks]])
            egm_peak_vals = np.array([self.laser_exp.rvshock.data[self.laser_exp.rvshock.peaks_sample[0:num_peaks]]])

        ecg_peak_values = self.reject_outliers(ecg_peak_vals)
        egm_peak_values = self.reject_outliers(egm_peak_vals)
        mean_ecg_peak = np.mean(ecg_peak_values)
        mean_egm_peak = np.mean(egm_peak_values)

        peak_diff = np.subtract(ecg_peaks, egm_peaks)
        peak_diff_new = self.reject_outliers(peak_diff)
        time_diff = int(np.mean(peak_diff_new))

        remove = np.arange(0, abs(time_diff), step=1)

        if time_diff < 0:
            print(f"Problem. Should not be removing {time_diff} samples from ECG Dataset")

        else:
            print(f"Removing {time_diff} samples from ICD Leads")
            egms = ['RVbip', 'RVshock', 'RAlead', 'LVlead']
            for k in self.used_signals.keys():
                string = k.lower()
                data_copy = self.laser_exp.data[k].data.copy()
                length = len(data_copy) - time_diff

                if k in egms:
                    if mean_egm_peak < mean_ecg_peak:
                        factor = mean_ecg_peak / mean_egm_peak
                        data_copy = data_copy * factor
                    if ecg_peaks[0][0] < egm_peaks[0][0]:
                        self[string + '_corr'] = np.delete(data_copy, remove, axis=0)
                    else:
                        self[string + '_corr'] = data_copy[:length]


                elif k in ['ECG', 'ECG3']:
                    if mean_ecg_peak < mean_egm_peak:
                        factor = mean_egm_peak / mean_ecg_peak
                        data_copy = data_copy * factor
                    if ecg_peaks[0][0] > egm_peaks[0][0]:
                        self[string + '_corr'] = data_copy[time_diff:len(data_copy)]
                    else:
                        self[string + '_corr'] = data_copy[0:length]

                else:
                    if ecg_peaks[0][0] < egm_peaks[0][0]:
                        self[string + '_corr'] = data_copy[time_diff:len(data_copy)]
                    else:
                        self[string + '_corr'] = data_copy[0:length]

    def resample_data(self, original_fs, desired_fs):
        cardiac_signal = ['ECG', 'ECG3', 'RVbip', 'RVshock', 'RAlead', 'LVlead']
        for k in self.used_signals.keys():
            string = k.lower()
            sec = (self[string + '_corr'].size) / original_fs
            new_length = int(sec * desired_fs)
            self[string + '_resampled'] = scipy.signal.resample(self[string + '_corr'], new_length)
            if k in cardiac_signal:
                self[string + '_resampled_norm'] = self[string + '_resampled'] - min(self[string + '_resampled']) / (np.max(self[string + '_resampled']) - min(self[string + '_resampled']))
                self[string + '_resampled_norm'] = self[string + '_resampled_norm'] - np.mean(self[string + '_resampled_norm'])


    def ecg_filter(self):
        ecg_sos = scipy.signal.butter(5, (0.5, 25), 'band', fs=512, output='sos')
        for k in self.used_signals.keys():
            if k in ['ECG', 'ECG3']:
                string = k.lower()
                self[string + '_filt'] = scipy.signal.sosfilt(ecg_sos, self[string + '_resampled_norm'])

    # def amplifier(self):
    #     # Fixing Voltage on leads
    #     self.ra_data = self.resampled_ra_data * 10
    #     self.rvbip_data  = self.resampled_rv_data * 10
    #     self.rvshock_data = self.resampled_shock_data * 10
    #     self.lv_data = self.resampled_lv_data * 10
    #     self.ecg_data = self.ecg_data * 3
    #     self.ecg3_data = self.ecg3_data * 3

    def rectifier(self):
        # Rectifying the ECG
        cardiac_signal = ['ECG', 'ECG3', 'RVbip', 'RVshock', 'RAlead', 'LVlead']
        for k in self.used_signals.keys():
            if k in cardiac_signal:
                string = k.lower()
                if k in ['ECG', 'ECG3']:
                    self[string + '_rect'] = abs(self[string + '_filt'])
                else:
                    self[string + '_rect'] = abs(self[string + '_resampled_norm'])

    def sq_rectifier(self):
        self.rectifier()
        # Rectifying the ECG
        cardiac_signal = ['ECG', 'ECG3', 'RVbip', 'RVshock', 'RAlead', 'LVlead']
        for k in self.used_signals.keys():
            if k in cardiac_signal:
                string = k.lower()

                self[string + '_sqrect'] = (self[string + '_rect'] ** 2) * 10

    def derivatives(self):
        self.sq_rectifier()
        cardiac_signal = ['ECG', 'ECG3', 'RVbip', 'RVshock', 'RAlead', 'LVlead']
        for k in self.used_signals.keys():
            if k in cardiac_signal:
                string = k.lower()

                self[string + '_gradient'] = np.gradient(self[string + '_sqrect'])

    def zero_crossings(self, max_peaks):
        self.derivatives()
        cardiac_signal = ['ECG', 'ECG3', 'RVbip', 'RVshock', 'RAlead', 'LVlead']
        for k in self.used_signals.keys():
            if k in cardiac_signal:
                string = k.lower()
                self[string + '_zerocross'] = []
                for peak in max_peaks:
                    # Zero Crossings - This detects the point where the gradient changes sign
                    start = max(0, peak - 30)
                    end = min(peak + 30, len(self[string + '_gradient']))
                    zerocross = np.where(np.diff(np.sign(self[string + '_gradient'][start:end])))
                    if len(zerocross[0]) > 0:
                        new_list = np.abs(zerocross - peak).argmin()
                        closest_zero = min(new_list + start, end)
                        self[string + '_zerocross'].append(closest_zero)

    def peak_zerox_diff(self, zero_cross, peaks):
        cardiac_signal = ['ECG', 'ECG3', 'RVbip', 'RVshock', 'RAlead', 'LVlead']
        for k in self.used_signals.keys():
            if k in cardiac_signal:
                string = k.lower()
                self[string + '_peak_zerox'] = []

                peak_zerox_diff = np.array(peaks) - np.array(zero_cross)
                return peak_zerox_diff

    def binary_data(self, zerox, peaks):
        # Define the padding range
        pad_range = 60

        # Initialize an array filled with zeros
        result_array = np.zeros(len(self.rvbip_rect), dtype=int)

        # Mark the specified ranges with 1
        for zero, peak in zip(zerox, peaks):
            result_array[zero:peak + pad_range +1] = 1
        # self.rvbip_binary = [1 if (x > i and x < peak + 60) else 0 for x, y in enumerate(self.rvbip_gradient) for i in
        #                      self.rvbip_zerocross for peak in max_peaks]
    def exp_decay_plot(self, start_thresh, xdata): #This is the correct one
        ylist = []
        for x in xdata:
            y = start_thresh * (1 - 1/312)**x
            ylist.append(y)
        return ylist

    def find_peaks(self, data, pvsb, rvst, half_win_size_ms, sampling_freq, med_rr): #This is the correct one
        if half_win_size_ms > med_rr:
            half_win_size_ms = int(med_rr/2)
        win_size = int(half_win_size_ms * (sampling_freq / 1000))
        start_thresh = rvst
        pvsb = int(pvsb * (sampling_freq / 1000))
        peak_vals = []
        last_peak_index = None

        next_peak = 0
        max_peaks = []
        rr_list = []
        for i, peak_value in enumerate(data):
            if (peak_value > rvst and ((last_peak_index is None) or ((i - last_peak_index) >= pvsb))) or (((last_peak_index is not None) and (i - last_peak_index) >= pvsb) and (next_peak==i) and (next_peak!=0)):
                min_win = max(0, i - win_size)
                max_win = min(i + win_size, len(data))

                max_peak = np.argmax(data[min_win:max_win]) + min_win

                if (last_peak_index is not None) and ((max_peak - last_peak_index)< pvsb):
                    continue
                else:
                    rr_list.append(max_peak)

                max_peaks.append(max_peak)

                peak_val = data[max_peak]
                peak_vals.append(peak_val)

                if last_peak_index is not None:
                    rr_list.append(max_peak - last_peak_index)

                last_peak_index = max_peak

                if peak_val > rvst*4:
                    self.icd_memory['low_sens'].append(1)
                    self.icd_memory['high_sens'].append(1)
                if peak_val < rvst*4 or peak_val > rvst*2.8:
                    self.icd_memory['low_sens'].append(0)
                    self.icd_memory['high_sens'].append(0)
                if peak_val < rvst*2.8:
                    self.icd_memory['low_sens'].append(-1)
                    self.icd_memory['high_sens'].append(-1)

                if peak_val*0.75 > rvst*8:
                    start_thresh = 8 * rvst
                else:
                    start_thresh = peak_val*0.75
                if start_thresh < rvst:
                    start_thresh = rvst

                if len(self.icd_memory['low_sens'])==16 and all(item == -1 for item in self.icd_memory['low_sens']) and self.icd_mdt_parameters['rvst_value'] <=0.6:
                    self.icd_mdt_parameters['rvst_value'] = max(self.icd_mdt_parameters['rvst_value'] - 0.15, 0.15)
                    self.icd_memory['low_sens'] = deque(maxlen=16)
                if len(self.icd_memory['low_sens'])==16 and all(item == -1 for item in self.icd_memory['low_sens']) and self.icd_mdt_parameters['rvst_value'] >0.6:
                    self.icd_mdt_parameters['rvst_value'] = self.icd_mdt_parameters['rvst_value'] - 0.3
                    self.icd_memory['low_sens'] = deque(maxlen=16)
                if len(self.icd_memory['high_sens'])==36 and all(item == 1 for item in self.icd_memory['high_sens']) and self.icd_mdt_parameters['rvst_value'] < 0.6:
                    self.icd_mdt_parameters['rvst_value'] = min(self.icd_mdt_parameters['rvst_value'] + 0.15, 0.6)
                    self.icd_memory['high_sense'] = deque(maxlen=36)
                if len(self.icd_memory['high_sens'])==36 and all(item == 1 for item in self.icd_memory['high_sens']) and self.icd_mdt_parameters['rvst_value'] >= 0.6:
                    self.icd_mdt_parameters['rvst_value'] = min(self.icd_mdt_parameters['rvst_value'] + 0.3, 1.8)
                    self.icd_memory['high_sense'] = deque(maxlen=36)


                start = last_peak_index + int(pvsb * 0.512)
                rr_deque = deque(rr_list, maxlen=8)
                maxwin_nextpeak = max_win + np.median(rr_deque)*3

                exp_decay = self.exp_decay_plot(start_thresh, np.arange(0, maxwin_nextpeak - start))

                for j, val in enumerate(exp_decay):
                    k = min(j+start, len(data)-1)
                    if data[k] > val:
                        next_peak =k

                        print('Peak found near ' + str(next_peak))
                        break
                    else:
                        pass

        return max_peaks, peak_vals, rr_list, self.icd_memory['low_sens'], self.icd_memory['high_sens']