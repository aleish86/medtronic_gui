import scipy
from scipy import signal
import numpy as np
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
        self.resample_data(1000, 512)
        self.fix_lag()
        self.ecg_filter()
        self.amplifier()
        # Filtering ECGs
        self.rectifier()
        self.sq_rectifier()
        self.derivatives()

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

        self.zero_crossings()
        self.peaks_zero_crossings()

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
                    if ecg_peaks[0] < egm_peaks[0]:
                        self[string + '_corr'] = np.delete(data_copy, remove, axis=0)
                    else:
                        self[string + '_corr'] = data_copy[:length]


                elif k in ['ECG', 'ECG3']:
                    if mean_ecg_peak < mean_egm_peak:
                        factor = mean_egm_peak / mean_ecg_peak
                        data_copy = data_copy * factor
                    if ecg_peaks[0] > egm_peaks[0]:
                        self[string + '_corr'] = data_copy[time_diff:len(data_copy)]
                    else:
                        self[string + '_corr'] = data_copy[0:length]

                else:
                    if ecg_peaks[0] > egm_peaks[0]:
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
        # norm_ecg_signal = (ecg_signal - min(ecg_signal)) / (max(ecg_signal) - min(ecg_signal))
        # norm_egm_signal = (egm_signal - min(egm_signal)) / (max(egm_signal) - min(egm_signal))

    def ecg_filter(self):
        ecg_sos = scipy.signal.butter(5, (0.5, 25), 'band', fs=512, output='sos')
        self.ecg_data = scipy.signal.sosfilt(ecg_sos, self.resampled_ecg_data)
        self.ecg3_data = scipy.signal.sosfilt(ecg_sos, self.resampled_ecg3_data)

    def amplifier(self):
        # Fixing Voltage on leads
        self.ra_data = self.resampled_ra_data * 10
        self.rvbip_data  = self.resampled_rv_data * 10
        self.rvshock_data = self.resampled_shock_data * 10
        self.lv_data = self.resampled_lv_data * 10
        self.ecg_data = self.ecg_data * 3
        self.ecg3_data = self.ecg3_data * 3

    def rectifier(self):
        # Rectifying the ECG
        self.rect_ecg3 = abs(self.ecg3_data)
        self.rect_ecg = abs(self.ecg_data)
        self.rect_ralead = abs(self.ra_data)
        self.rect_rvbip = abs(self.rvbip_data)
        self.rect_rvshock = abs(self.rvshock_data)
        self.rect_lvlead = abs(self.lv_data)
        self.rect_laser1 = abs(self.resampled_laser1_data)
        self.rect_laser2 = abs(self.resampled_laser2_data)

    def sq_rectifier(self):
        self.rectifier()
        # Rectifying the ECG
        self.sqrect_ecg3 = (self.rect_ecg3 ** 2) * 10
        self.sqrect_ecg = (self.rect_ecg ** 2) * 10
        self.sqrect_ralead = (self.rect_ralead ** 2) * 10
        self.sqrect_rvbip = (self.rect_rvbip ** 2) * 10
        self.sqrect_rvshock = (self.rect_rvshock ** 2) * 10
        self.sqrect_lvlead = (self.rect_lvlead ** 2) * 10
        self.sqrect_laser1 = (self.rect_laser1 ** 2) * 10
        self.sqrect_laser2 = (self.rect_laser2 ** 2) * 10

    def derivatives(self):
        self.sq_rectifier()
        self.gradient_ecg = np.gradient(self.sqrect_ecg)
        self.gradient_ecg3 = np.gradient(self.sqrect_ecg3)
        self.gradient_ralead = np.gradient(self.sqrect_ralead)
        self.gradient_rvbip = np.gradient(self.sqrect_rvbip)
        self.gradient_rvshock = np.gradient(self.sqrect_rvshock)
        self.gradient_lvlead = np.gradient(self.sqrect_lvlead)
        self.gradient_laser1 = np.gradient(self.sqrect_laser1)
        self.gradient_laser2 = np.gradient(self.sqrect_laser2)

    def zero_crossings(self):
        self.derivatives()
        # Zero Crossings - This detects the point where the gradient changes sign
        self.zero_cross_ecg = np.where(np.diff(np.sign(self.gradient_ecg)))[0]
        self.zero_cross_ecg3 = np.where(np.diff(np.sign(self.gradient_ecg3)))[0]
        self.zero_cross_ralead = np.where(np.diff(np.sign(self.gradient_ralead)))[0]
        self.zero_cross_rvbip = np.where(np.diff(np.sign(self.gradient_rvbip)))[0]
        self.zero_cross_rvshock = np.where(np.diff(np.sign(self.gradient_rvshock)))[0]
        self.zero_cross_lvlead = np.where(np.diff(np.sign(self.gradient_lvlead)))[0]
        self.zero_cross_laser1 = np.where(np.diff(np.sign(self.gradient_laser1)))[0]
        self.zero_cross_laser2 = np.where(np.diff(np.sign(self.gradient_laser2)))[0]

    def peaks_zero_crossings(self):
        self.zero_crossings()
        items = ['ecg', 'ecg3', 'ralead', 'rvbip', 'rvshock', 'lvlead', 'laser1', 'laser2']
        for item in items:
            try:
                prev_crossing = None
                for crossing in getattr(self, 'zero_cross_' + item):
                    if prev_crossing is None or crossing - prev_crossing > 62:  # 1000/512 = 1.95, therefore 120/1.95 = 61.5
                        peak_index = np.argmax(getattr(self, 'sqrect_' + item)[crossing - 75:crossing + 75]) + crossing
                        getattr(self, item + '_peaks').append(peak_index)
                        prev_crossing = crossing

                for i in range(1, len(getattr(self, item + '_peaks'))):
                    diff = np.gradient(getattr(self, 'sqrect_' + item).data[getattr(self, item + '_peaks')[i - 1]:getattr(self, item + '_peaks')[i]])
                    if np.max(np.abs(diff)) > 0:
                        getattr(self, 'filt_' + item + '_peaks').append(getattr(self, item + '_peaks')[i])
                print('filt' + item +  'peak', getattr(self, 'filt_' + item + '_peaks'))
            except:
                continue

    def peak_adaptive_threshold(self, ecg_data, window_size, threshold, pvsb, factor):
        detected_peaks = []

        last_peak_index = 0
        for i, sample in enumerate(ecg_data):
            if sample > threshold:
                if ((i - last_peak_index) >= pvsb):
                    detected_peaks.append(i)
                    last_peak_index = i
            if i >= window_size:
                window_start = i - window_size
                window_end = i
                window_mean = np.mean(ecg_data[window_start:window_end])
                threshold = window_mean * factor
        return detected_peaks

    # def adaptive_threshold(self):
    #     time = 480 #ms
    #     rate = 1 mV per 312 ms = 3.125 hz
    #     last_rpeak_value * (e**(-3.125*(t/480)))

# def data_list(data, rv_sens_threshold, post_vs_blanking):
#     # def data_list(data):
#     max_index_list = []
#     amplitude_list = []
#     rr_list = []
#     rpeaks = []
#     post_vs_blanking = post_vs_blanking
#     abs_data = abs(data)**2
#
#     icd_memory = {}
#     icd_memory['last_r_peak'] = 0
#
#     rv_peaks = np.where(data > rv_sens_threshold)
#     # rv_peaks = np.where(data > 0.3)
#     rv_peaks = np.array(rv_peaks).flatten()
#     print("rv_peaks: ", rv_peaks)
#     for i, r_peak in enumerate(rv_peaks):
    #     if i == 0:
    #         last_rpeak = 0
    #     else:
    #         pass
    #     if (r_peak - last_rpeak) > post_vs_blanking:
    #         rpeaks.append(r_peak)
    #     else:
    #         pass
    #     last_rpeak = r_peak
    #     rpeaks = list(set(rpeaks))
    # rpeaks = np.sort(rpeaks)
    # time = 480  # ms
    #     #     rate = 1 mV per 312 ms = 3.125 hz
    # last_rpeak_value = rpeaks[0]
    # peak_values = data[rpeaks]
    # decay_rate = -3.125
    # e = 2.71828
    #
    # low_sens_limit = rv_sens_threshold * 2.8
    # high_sens_limit = rv_sens_threshold * 4
    # time_d = []