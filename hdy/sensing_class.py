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
        # Fixing and Correlating Data
        self.resample_data(1000, 512)
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
        self.used_signals = {}
        for k, v in self.laser_exp.data_source.items():
            if v != 'blank':
                self.used_signals[k] = v
            else:
                pass

        if 'ECG' in self.used_signals.keys():
            ecg_signal = self.laser_exp.ecg.data[0:6000]
            num_peaks = min(10, len(self.laser_exp.ecg.peaks_sample))
            ecg_peaks = np.array([self.laser_exp.ecg.peaks_sample[0:num_peaks]])
        else:
            ecg_signal = self.laser_exp.ecg3.data[0:6000]
            num_peaks = min(10, len(self.laser_exp.ecg3.peaks_sample))
            ecg_peaks = np.array([self.laser_exp.ecg3.peaks_sample[0:num_peaks]])

        if 'RVbip' in self.used_signals.keys():
            egm_signal = self.laser_exp.rvbip.data
            num_peaks = min(10, len(self.laser_exp.rvbip.peaks_sample))
            egm_peaks = np.array([self.laser_exp.rvbip.peaks_sample[0:num_peaks]])
        else:
            egm_signal = self.laser_exp.rvshock.data
            num_peaks = min(10, len(self.laser_exp.rvshock.peaks_sample))
            egm_peaks = np.array([self.laser_exp.rvshock.peaks_sample[0:num_peaks]])

        # norm_ecg_signal = (ecg_signal - min(ecg_signal)) / (max(ecg_signal) - min(ecg_signal))
        # norm_egm_signal = (egm_signal - min(egm_signal)) / (max(egm_signal) - min(egm_signal))
        peak_diff = np.subtract(ecg_peaks, egm_peaks)
        peak_diff_new = self.reject_outliers(peak_diff)
        time_diff = round(np.mean(peak_diff_new),0)
        # corr = scipy.signal.correlate(norm_egm_signal, norm_ecg_signal, mode="full")
        # lags = scipy.signal.correlation_lags(len(norm_egm_signal), len(norm_ecg_signal), mode="full")
        # lag = lags[np.argmax(corr)]
        # print(f"Best lag: {lag}")

        remove = np.arange(0, abs(time_diff), step=1)

        if lag < 0:
            print(f"Problem. Should not be removing {len(remove)} samples from ECG Dataset")

        else:
            print(f"Removing {len(remove)} samples from ICD Leads")
            egms = ['RVbip', 'RVshock', 'RAlead', 'LVlead']
            for k in self.used_signals.keys():
                string = str(str(k).lower())
                data_copy = self.laser_exp.data[k].data.copy()

                if k in egms:
                    self[string + '_corr'] = np.delete(data_copy, remove, axis=0)

                else:
                    length = len(egm_signal)
                    self[string + '_corr'] = data_copy[0:length]

    def resample_data(self, original_fs, desired_fs):
        sec = (self.laser_exp.ecg.data.size) / original_fs
        new_length = int(sec * desired_fs)
        self.resampled_bp_data = scipy.signal.resample(self.laser_exp.pressure.data, new_length)
        self.resampled_ecg3_data = scipy.signal.resample(self.laser_exp.ecg3.data, new_length)
        self.resampled_ecg_data = scipy.signal.resample(self.laser_exp.ecg.data, new_length)
        self.resampled_laser1_data = scipy.signal.resample(self.laser_exp.laser1.data, new_length)
        self.resampled_laser2_data = scipy.signal.resample(self.laser_exp.laser2.data, new_length)
        self.resampled_ra_data = (scipy.signal.resample(self.laser_exp.ralead.data, new_length))
        self.resampled_rv_data = scipy.signal.resample(self.laser_exp.rvbip.data, new_length)
        self.resampled_shock_data = scipy.signal.resample(self.laser_exp.rvshock.data, new_length)
        self.resampled_lv_data = scipy.signal.resample(self.laser_exp.lvlead.data, new_length)
        self.laser1_data = self.resampled_laser1_data
        self.laser2_data = self.resampled_laser2_data
        self.bp_data = self.resampled_bp_data

        # resampled_bp_data = scipy.signal.resample(self.laser_exp.pressure.data, new_length)
        # resampled_ecg3_data = scipy.signal.resample(self.laser_exp.ecg3.data, new_length)
        # resampled_ecg_data = scipy.signal.resample(self.laser_exp.ecg.data, new_length)
        # resampled_laser1_data = scipy.signal.resample(self.laser_exp.laser1.data, new_length)
        # resampled_laser2_data = scipy.signal.resample(self.laser_exp.laser2.data, new_length)
        # resampled_ra_data = (scipy.signal.resample(self.laser_exp.ralead.data, new_length))
        # resampled_rv_data = scipy.signal.resample(self.laser_exp.rvbip.data, new_length)
        # resampled_shock_data = scipy.signal.resample(self.laser_exp.rvshock.data, new_length)
        # resampled_lv_data = scipy.signal.resample(self.laser_exp.lvlead.data, new_length)

        # log_bp_data = np.log(resampled_bp_data+10)
        # log_ecg3_data = np.log(resampled_ecg3_data+10)
        # log_ecg_data = np.log(resampled_ecg_data+10)
        # log_laser1_data = np.log(resampled_laser1_data+10)
        # log_laser2_data = np.log(resampled_laser2_data+10)
        # log_ra_data = np.log(resampled_ra_data+10)
        # log_rv_data = np.log(resampled_rv_data+10)
        # log_shock_data = np.log(resampled_shock_data+10)
        # log_lv_data = np.log(resampled_lv_data+10)
        #
        # scaler = MinMaxScaler(feature_range=(-1, 1))
        # self.resampled_bp_data = scaler.fit_transform(log_bp_data.reshape(-1, 1)).reshape(-1)
        # self.resampled_ecg3_data = scaler.fit_transform(log_ecg3_data.reshape(-1, 1)).reshape(-1)
        # self.resampled_ecg_data = scaler.fit_transform(log_ecg_data.reshape(-1, 1)).reshape(-1)
        # self.resampled_laser1_data = scaler.fit_transform(log_laser1_data.reshape(-1, 1)).reshape(-1)
        # self.resampled_laser2_data = scaler.fit_transform(log_laser2_data.reshape(-1, 1)).reshape(-1)
        # self.resampled_ra_data = scaler.fit_transform(log_ra_data.reshape(-1, 1)).reshape(-1)
        # self.resampled_rv_data = scaler.fit_transform(log_rv_data.reshape(-1, 1)).reshape(-1)
        # self.resampled_shock_data = scaler.fit_transform(log_shock_data.reshape(-1, 1)).reshape(-1)
        # self.resampled_lv_data = scaler.fit_transform(log_lv_data.reshape(-1, 1)).reshape(-1)


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