import csv
import pandas as pd
import scipy
from scipy import signal
import numpy as np
import mmt
from collections import defaultdict
from collections import deque
from collections import Counter
from . import *
from PySide6 import QtGui, QtCore, QtWidgets
import math
import pyqtgraph as pg
import pywt

class Wavelet_GUI(QtWidgets.QWidget):
    def __init__(self, laser_exp: object, parent: object = None)-> object:
        super().__init__(parent=parent)
        self.laser_exp = laser_exp
        self.trace_view_wavelet = 0
        self.wavelet_match_score = 0
        self.pct_match_score = 0

        self.icd_mdt_parameters = {}
        self.icd_mdt_parameters['vt_tcl'] = 0 # Nominal OFF
        self.icd_mdt_parameters['vf_tcl'] = 320 # 188 bpm
        self.icd_mdt_parameters['rvst_value'] = 0.3 # 188 bpm
        self.icd_mdt_parameters['pvsb_value'] = 120 # Nominal
        self.icd_mdt_parameters['vt_min_nid'] = 24 # Nominal
        self.icd_mdt_parameters['vf_max_nid'] = 40 # Nominal 30/40
        self.icd_mdt_parameters['wavelet_match_threshold'] = 70 # 70% match Nominal Setting

        self.icd_memory = {}
        self.icd_memory['r_peak'] = deque(maxlen=8)
        self.icd_memory['last_r_peak'] = 0
        self.icd_memory['rhythm_label'] = deque(maxlen=max(self.icd_mdt_parameters['vt_min_nid'], self.icd_mdt_parameters['vf_max_nid']))
        self.icd_memory['wavelet_beats'] = deque(maxlen=8)
        self.icd_memory['wavelet_match_beats'] = deque(maxlen=8)

        self.x_marker = 0

        self.init_ui()
        self.setup_wavelet_plots()

    def init_ui(self):
        pg.setConfigOptions(antialias=True, background='w')

        self.wavelet_vbox = QtWidgets.QVBoxLayout()
        self.wavelet_overall_hbox = QtWidgets.QHBoxLayout()
        self.wavelet_hbox = QtWidgets.QVBoxLayout()
        self.wavelet_template_grid = QtWidgets.QHBoxLayout()
        self.wavelet_template_box = QtWidgets.QHBoxLayout()

        self.wavelet_lbl_w = QtWidgets.QLabel("Wavelet")
        self.wavelet_lbl_w.setStyleSheet("font-weight: bold;")
        self.wavelet_lbl_w.setFixedWidth(75)
        self.spacer_lbl = QtWidgets.QLabel()
        self.media_control_lbl = QtWidgets.QLabel()
        self.media_control_lbl.setText("\nMedia Controls")
        self.media_control_lbl.setStyleSheet("font-weight: bold")


        # Creating Viewing Toggles
        self.start_wavelet_btn = QtWidgets.QPushButton("Start")
        self.start_wavelet_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.start_wavelet_btn.clicked.connect(self.start_btn_clicked_w)

        self.pause_wavelet_btn = QtWidgets.QPushButton("Pause")
        self.pause_wavelet_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.pause_wavelet_btn.clicked.connect(self.pause_btn_clicked_w)

        self.collect_template_btn = QtWidgets.QPushButton('Collect Template')
        self.collect_template_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.collect_template_btn.clicked.connect(self.collect_template_btn_clicked)

        self.validate_template_btn = QtWidgets.QPushButton('Validate Template')
        self.validate_template_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.validate_template_btn.clicked.connect(self.validate_template_btn_clicked)

        self.check_wavelet_match_btn = QtWidgets.QPushButton('Check Wavelet Match')
        self.check_wavelet_match_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.check_wavelet_match_btn.clicked.connect(self.check_wavelet_match_btn_clicked)

        self.rvbip_wavelet_toggle = QtWidgets.QCheckBox("View RV Bipolar")
        self.rvbip_wavelet_toggle.setChecked(False)
        self.rvbip_wavelet_toggle.stateChanged.connect(self.rvbip_wavelet_toggle_changed)

        self.rvshock_wavelet_toggle = QtWidgets.QCheckBox("View RV Shock")
        self.rvshock_wavelet_toggle.setChecked(True)
        self.rvshock_wavelet_toggle.stateChanged.connect(self.rvshock_wavelet_toggle_changed)

        self.bipecg_wavelet_toggle = QtWidgets.QCheckBox("View Bipolar ECG")
        self.bipecg_wavelet_toggle.setChecked(False)
        self.bipecg_wavelet_toggle.stateChanged.connect(self.bipecg_wavelet_toggle_changed)

        self.bipecg_wavelet_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.bipecg_wavelet_pw.resize(200, 800)
        self.bipecg_wavelet_pi = self.bipecg_wavelet_pw.getPlotItem()
        self.bipecg_wavelet_pi.setTitle("Bipolar ECG", color='k', size='16pt')
        self.bipecg_wavelet_pi.setLabel(axis='left')
        self.bipecg_wavelet_plt = self.bipecg_wavelet_pi.plot()
        self.bipecg_peak_wavelet_plt = self.bipecg_wavelet_pi.plot()
        self.bipecg_wavelet_pw.hide()

        self.rvbip_wavelet_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.rvbip_wavelet_pw.resize(200, 800)
        self.rvbip_wavelet_pi = self.rvbip_wavelet_pw.getPlotItem()
        self.rvbip_wavelet_pi.setTitle("RV Bipolar", color='k', size='16pt')
        self.rvbip_wavelet_pi.setLabel(axis='left')
        self.rvbip_wavelet_plt = self.rvbip_wavelet_pi.plot()
        self.rvbip_wavelet_pw.hide()

        self.rvshock_wavelet_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.rvshock_wavelet_pw.resize(50, 200)
        self.rvshock_wavelet_pi = self.rvshock_wavelet_pw.getPlotItem()
        self.rvshock_wavelet_pi.setTitle("RV Shock", color='k', size='16pt')
        self.rvshock_wavelet_pi.setLabel(axis='left')
        self.rvshock_wavelet_plt = self.rvshock_wavelet_pi.plot()

        self.wavelet_template_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.wavelet_template_pi = self.wavelet_template_pw.getPlotItem()
        self.wavelet_template_pi.setTitle("Wavelet Template", color='k', size='16pt')
        self.wavelet_template_pi.setLabel(axis='left')
        self.wavelet_template_plt = self.wavelet_template_pi.plot()

        self.wavelet_check_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.wavelet_check_pi = self.wavelet_check_pw.getPlotItem()
        self.wavelet_check_pi.setTitle("Wavelet Check", color='k', size='16pt')
        self.wavelet_check_pi.setLabel(axis='left')
        self.wavelet_check_plt = self.wavelet_check_pi.plot()

        self.maxcoeffs_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.maxcoeffs_pi = self.maxcoeffs_pw.getPlotItem()
        self.maxcoeffs_pi.setTitle("Max Wavelet Coefficients", color='k', size='16pt')
        self.maxcoeffs_pi.setLabel(axis='left')
        self.maxcoeffs_plt = self.maxcoeffs_pi.plot()
        # Enable interactive mode for the PlotWidget
        self.maxcoeffs_pw.setMouseEnabled(x=True, y=True)

        self.wavelet_vbox.addWidget(self.wavelet_lbl_w)
        self.wavelet_hbox.addWidget(self.bipecg_wavelet_pw)
        self.wavelet_hbox.addWidget(self.rvbip_wavelet_pw)
        self.wavelet_hbox.addWidget(self.rvshock_wavelet_pw)

        self.wavelet_vbox.addWidget(self.rvshock_wavelet_toggle)
        self.wavelet_vbox.addWidget(self.rvbip_wavelet_toggle)
        self.wavelet_vbox.addWidget(self.bipecg_wavelet_toggle)
        self.wavelet_vbox.addWidget(self.spacer_lbl)
        self.wavelet_vbox.addWidget(self.media_control_lbl)
        self.wavelet_vbox.addWidget(self.start_wavelet_btn)
        self.wavelet_vbox.addWidget(self.pause_wavelet_btn)
        self.wavelet_vbox.addWidget(self.spacer_lbl)
        self.wavelet_template_grid.addWidget(self.wavelet_template_pw)
        self.wavelet_template_grid.addWidget(self.maxcoeffs_pw)
        self.wavelet_template_grid.addWidget(self.wavelet_check_pw)

        self.wavelet_template_box.addWidget(self.collect_template_btn)
        self.wavelet_template_box.addWidget(self.validate_template_btn)
        self.wavelet_template_box.addWidget(self.check_wavelet_match_btn)
        self.wavelet_template_box.setAlignment(QtCore.Qt.AlignLeft)
        self.wavelet_template_grid.setSpacing(0)
        self.wavelet_template_grid.setContentsMargins(0, 0, 0, 0)
        self.wavelet_hbox.setSpacing(10)
        self.wavelet_hbox.setContentsMargins(10, 10, 10, 500)
        self.wavelet_vbox.setSpacing(10)
        self.wavelet_vbox.setContentsMargins(10, 10, 10, 850)
        self.wavelet_overall_hbox.addLayout(self.wavelet_vbox)
        self.wavelet_hbox.addLayout(self.wavelet_template_grid)
        self.wavelet_hbox.addLayout(self.wavelet_template_box)
        self.wavelet_overall_hbox.setSpacing(0)
        self.wavelet_overall_hbox.setContentsMargins(0, 0, 0, 0)
        self.wavelet_overall_hbox.addLayout(self.wavelet_hbox)

        # Setting up the Timer for Updating Plots
        self.timer_w = QtCore.QTimer()
        self.timer_w.timeout.connect(self.update_wavelet_plots)

        # self.max_x_list, self.rr_list, self.amplitude_list, self.rpeaks = data_list(self.laser_exp.rvbip.data,
        #                                                                self.icd_mdt_parameters['rvst_value'],
        #                                                                self.icd_mdt_parameters['pvsb_value'])
        self.max_x_list, self.rr_list = data_list(self.laser_exp.rvbip.data,
                                                                       self.icd_mdt_parameters['rvst_value'],
                                                                       self.icd_mdt_parameters['pvsb_value'])
        self.setLayout(self.wavelet_overall_hbox)


    def setup_wavelet_plots(self):
        self.rvbip_wavelet_pi.getAxis('left').setWidth(w=50)
        self.rvbip_wavelet_pi.getAxis('left').setStyle(showValues=False)
        self.rvshock_wavelet_pi.getAxis('left').setWidth(w=50)
        self.rvshock_wavelet_pi.getAxis('left').setStyle(showValues=False)
        self.bipecg_wavelet_pi.getAxis('left').setWidth(w=50)
        self.bipecg_wavelet_pi.getAxis('left').setStyle(showValues=False)
        self.wavelet_check_pi.getAxis('left').setWidth(w=50)
        self.wavelet_check_pi.getAxis('left').setStyle(showValues=False)
        self.wavelet_template_pi.getAxis('left').setWidth(w=50)
        self.wavelet_template_pi.getAxis('left').setStyle(showValues=False)
        self.maxcoeffs_pi.getAxis('left').setWidth(w=50)
        self.maxcoeffs_pi.getAxis('left').setStyle(showValues=False)

    def update_wavelet_plots(self):
        pg.setConfigOptions(antialias=True, background='w')
        samples = self.laser_exp.pressure.data.shape[0]
        ecg_hint = self.laser_exp.hints['Period']

        m_beat_end = 500 + self.trace_view_wavelet
        m_beat_start = max(0, m_beat_end - 10000)

        m_wavelet_range = np.arange(m_beat_start, m_beat_end)

        print('m_beat_start: ', m_beat_start)
        print('m_beat_end: ', m_beat_end)

        rvbip_wavelet_data = self.laser_exp.rvbip.data[m_beat_start:m_beat_end]
        rvshock_wavelet_data = self.laser_exp.rvshock.data[m_beat_start:m_beat_end]
        bipecg_wavelet_data = self.laser_exp.bipecg.data[m_beat_start:m_beat_end]

        self.trace_view_wavelet += 500

        for i, r_peak in enumerate(self.max_x_list):
            if m_beat_start <= r_peak <= m_beat_end:
                if not self.icd_memory['r_peak'] or r_peak > self.icd_memory['r_peak'][-1] + self.icd_mdt_parameters['pvsb_value']:

                    self.rr_int_wavelet = r_peak - self.icd_memory['last_r_peak']

                    self.x_marker = r_peak

                    if self.rr_int_wavelet <= self.icd_mdt_parameters['vf_tcl']:
                        self.marker_label = 'FS'

                    if self.icd_mdt_parameters['vt_tcl'] > self.rr_int_wavelet > self.icd_mdt_parameters['vf_tcl']:
                        self.marker_label = 'TS'

                    else:
                        self.marker_label = 'VS'

                    self.icd_memory['rhythm_label'].append(self.marker_label)

                    min_wavelet_x = r_peak - 100
                    max_wavelet_x = r_peak + 100
                    self.beat_plot = self.laser_exp.rvshock.data[min_wavelet_x: max_wavelet_x]
                    self.icd_memory['wavelet_beats'].append(self.beat_plot)
                    self.icd_memory['wavelet_match_beats'].append(self.beat_plot)

                    if m_beat_start < 0:
                        m_beat_start = 0
                    if m_beat_end > samples:
                        m_beat_end = samples
                        self.timer_w.stop()

        self.rvbip_wavelet_plt.setData(x=m_wavelet_range, y=rvbip_wavelet_data, pen='#732F9B', symbol=None,
                                       antialise=True,
                                       autoDownsample=True, clipToView=True)

        self.rvshock_wavelet_plt.setData(x=m_wavelet_range, y=rvshock_wavelet_data, pen='#732F9B', symbol=None,
                                         antialise=True,
                                         autoDownsample=True, clipToView=True)

        self.bipecg_wavelet_plt.setData(x=m_wavelet_range, y=bipecg_wavelet_data, pen='#0FA00F', symbol=None,
                                        antialise=True,
                                        autoDownsample=True, clipToView=True)

    def start_btn_clicked_w(self):
        self.timer_w.start(1000)

    def pause_btn_clicked_w(self):
        self.timer_w.stop()

    def rvbip_wavelet_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.rvbip_wavelet_pw.show()
            self.rvshock_wavelet_pw.hide()
            self.rvshock_wavelet_toggle.setChecked(False)
            self.bipecg_wavelet_pw.hide()
            self.bipecg_wavelet_toggle.setChecked(False)

            self.update_wavelet_plots()

    def rvshock_wavelet_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.rvshock_wavelet_pw.show()
            self.rvbip_wavelet_toggle.setChecked(False)
            self.rvbip_wavelet_pw.hide()
            self.bipecg_wavelet_toggle.setChecked(False)
            self.bipecg_wavelet_pw.hide()

            self.update_wavelet_plots()

    def bipecg_wavelet_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.bipecg_wavelet_pw.show()
            self.rvbip_wavelet_pw.hide()
            self.rvbip_wavelet_toggle.setChecked(False)
            self.rvshock_wavelet_pw.hide()
            self.rvshock_wavelet_toggle.setChecked(False)

            self.update_wavelet_plots()

    def collect_template_btn_clicked(self):
        print("Collecting template")
        normal_waveforms = self.icd_memory['wavelet_beats']

        num_non_empty_arrays = sum(1 for arr in normal_waveforms if len(arr) > 0)
        print("length normal waveforms", num_non_empty_arrays)

        window_length = 200
        wavelet = 'haar'  # 'db4'  # Choose a wavelet type (e.g., Daubechies-4)
        level = 4  # Choose the level of decomposition
        num_beats = 8
        dwt_coeffs = []
        idwt_beats = []
        self.wavelet_template_pw.clear()

        if len(normal_waveforms) < num_beats:
            print("Not enough beats to create wavelet template")
            return

        for beat in normal_waveforms:
            coeffs = pywt.wavedec(beat, wavelet, level=level)
            dwt_coeffs.append(coeffs)


        for coeffs in dwt_coeffs:
            beat = pywt.waverec(coeffs, wavelet)
            idwt_beats.append(beat)

        # Plot the reconstructed beats
        for i in range(num_beats):
            beat = idwt_beats[i]
            x = np.arange(len(beat))
            self.wavelet_template_pw.plot(x, beat, pen='k')

        saved_temp_coeffs = np.mean(dwt_coeffs, axis=0)
        # self.saved_wavelet_template = saved_temp_coeffs[1]
        # print('saved template', self.saved_wavelet_template)

        self.saved_wavelet_template = np.mean(idwt_beats, axis=0)


        color = QtGui.QColor(255, 0, 0, 127)  # Red color with 50% transparency
        pen = pg.mkPen(color=color, width=5)
        self.wavelet_template_pw.plot(x, self.saved_wavelet_template, pen=pen)

    def select_N_largest_coeffs(self, coeffs, N):
        abs_coeffs = np.abs(coeffs)  # Calculate the absolute values of coefficients
        print('abs_coeffs', abs_coeffs)
        x_axis = np.arange(len(coeffs))  # Generate x-axis values
        self.maxcoeffs_pw.clear()

        if N > len(abs_coeffs):
            N = len(abs_coeffs)

        maxcoeff_ten_indices = np.argpartition(abs_coeffs, -N)[-N:]
        maxcoeff_ten_values = np.sort(abs_coeffs[maxcoeff_ten_indices])
        print('maxcoeff_ten', maxcoeff_ten_values)

        # Create a BarGraphItem to plot the coefficients
        bar_graph = pg.BarGraphItem(x=x_axis, height=abs_coeffs, width=0.8, brush='grey')
        self.maxcoeffs_pw.addItem(bar_graph)

        # Create a BarGraphItem to plot the selected coefficients in red
        color = QtGui.QColor(255, 0, 0, 127)  # Red color with 50% transparency
        brush = pg.mkBrush(color=color)
        selected_bar_graph = pg.BarGraphItem(x=maxcoeff_ten_indices, height=maxcoeff_ten_values, width=0.8, brush=brush)
        self.maxcoeffs_pw.addItem(selected_bar_graph)

        # Set plot labels
        self.maxcoeffs_pw.setLabel('left', 'Absolute Coefficient Value')
        self.maxcoeffs_pw.setLabel('bottom', 'Coefficient Index')

        return maxcoeff_ten_values  # coeffs[selected_indices]

    def find_key(self, dictionary, value):
        self.key_list = []
        value = float(value)
        for key, values in dictionary.items():
            if values == value:
                self.key_list.append(key)
            else:
                pass
        return None


    def plot_wavelet_transform(self, template_coeffs, new_waveform):
        print('length new waveform', len(new_waveform))
        coeffs = pywt.wavedec(new_waveform, 'haar', level=4)

        recons_cA = pywt.waverec([coeffs], 'haar')
        recons_cD1 = pywt.waverec([coeffs[4]], 'haar')
        recons_cD2 = pywt.waverec([coeffs[3]], 'haar')
        recons_cD3 = pywt.waverec([coeffs[2]], 'haar')
        recons_cD4 = pywt.waverec([coeffs[1]], 'haar')

        self.wavelet_check_pw.clear()
        self.wavelet_check_pw.plot(coeffs[0], pen='k')
        self.wavelet_check_pw.plot(recons_cA, pen='y')
        self.wavelet_check_pw.plot(recons_cD1, pen='g')
        self.wavelet_check_pw.plot(recons_cD2, pen='b')
        self.wavelet_check_pw.plot(recons_cD3, pen='d')
        self.wavelet_check_pw.plot(recons_cD4, pen='r')
        self.wavelet_check_pw.plot(template_coeffs, pen='cyan')

    def compute_match_score(self, template_coeffs, unknown_coeffs):

        self.plot_wavelet_transform(template_coeffs, unknown_coeffs)
        # Select the 10 largest template coefficients
        largest_temp_coeffs = self.select_N_largest_coeffs(template_coeffs, 10)
        print('largest_temp_coeffs', largest_temp_coeffs)
        # Rank the template coefficients by finding location of the 10 largest coefficients
        rank_temp_coeffs = np.argsort(template_coeffs)[-10:]
        print('rank_temp_coeffs', rank_temp_coeffs)

        # Weighted match score
        simple_rank = np.arange(1, 11)

        template_simp_rank_dict = dict(zip(rank_temp_coeffs, simple_rank))


        # Create data dictionary for template coefficients
        template_coeff_dict = dict(zip(rank_temp_coeffs, np.round(largest_temp_coeffs, 2)))
        # template_coeff_dict = dict(zip(rank_temp_coeffs, largest_temp_coeffs))

        # Compute DWT of unknown waveform
        coeffs = pywt.wavedec(unknown_coeffs, 'haar', level=4)
        beat = pywt.waverec(coeffs, 'haar')

        # Select the 10 largest unknown coefficients
        largest_unknown_coeffs = self.select_N_largest_coeffs(beat, 10)
        # Rank the unknown coefficients
        rank_unknown_coeffs = np.argsort(beat)[-10:]

        unknown_coeff_dict = dict(zip(rank_unknown_coeffs, np.round(largest_unknown_coeffs, 2)))
        # unknown_coeff_dict = dict(zip(rank_unknown_coeffs, largest_unknown_coeffs))

        print('template_coeff_dict', template_coeff_dict)
        print('unknown_coeff_dict', unknown_coeff_dict)
        print(template_simp_rank_dict)
        simp_rank_list = []


        for rank, coeff in template_coeff_dict.items():
            # If largest unknown coefficients has coeff as one of its elements, then
            # if the absolute value of the rank of coeff - rank of coeff in unknown coeffs is = 1
            # if coeff in unknown_coeffs [for unknown_ranks, unknown_coeffs in unknown_coeff_dict.items()]:
            #     if abs(rank - unknown_ranks) <= 1:
            #
            if coeff in unknown_coeff_dict.values():
                self.find_key(unknown_coeff_dict, coeff)

                for key in self.key_list:
                    if abs(rank - key) <= 1:
                        # add the coefficient rank to the match score (i.e. match_score += coeff_rank)
            #     match_score = sum of simple rank for coeff in template_coeff_dict
                        print('coeff', coeff)
                        print('rank', rank)
                        # print('unknown coeff', unknown_coeff_dict[coeff])
                        print('key', key)
                        print('match score', template_simp_rank_dict[rank])

                        simp_rank_list.append(template_simp_rank_dict[rank])

        print('simp rank list', simp_rank_list)
        len_rank_list = len(simp_rank_list) + 1
        for i in np.arange(1,len_rank_list):
             if i == max(simp_rank_list):
                sum_ranks = sum(simp_rank_list)

                print('sum ranks', sum_ranks)
                prop_sum_ranks = sum_ranks / 55
                print('prop sum ranks', prop_sum_ranks)
                self.pct_match_score = int(prop_sum_ranks * 100)
                print('pct match score', self.pct_match_score)

        self.wavelet_match_score = min(self.pct_match_score, 100)

        print('match score', self.wavelet_match_score)

    def validate_template(self, template_coeffs, normal_waveforms):
        num_matches = 0
        text_items = []

        print('template coeffs', template_coeffs)
        print('normal waveforms', normal_waveforms)

        for i, waveform in enumerate(normal_waveforms):

            self.compute_match_score(template_coeffs, waveform)


            print('waveform', waveform)
            print('wavelet threshold: ', self.icd_mdt_parameters['wavelet_match_threshold'])
            print('match score: ', self.wavelet_match_score)
            print(int(self.icd_mdt_parameters['wavelet_match_threshold']))
            # Create a text item for each match score
            text_item = pg.TextItem(text=f"Match score:\n {self.wavelet_match_score}", color='k', anchor=(0, 0))
            text_item.setPos(self.x_marker, max(self.beat_plot))  # Set the position of the text item
            text_items.append(text_item)

            self.rvshock_wavelet_pw.addItem(text_item)

        if np.round(self.wavelet_match_score, 2) >= int(self.icd_mdt_parameters['wavelet_match_threshold']):  # Set the match threshold to 70%
            num_matches += 1



    def check_recent_waveforms(self, template_coeffs, recent_waveforms):
        mismatch_count = 0
        for waveform in recent_waveforms:
            self.compute_match_score(template_coeffs, waveform)
            # self.wavelet_match_score = min(self.pct_match_score, 100)

            if self.wavelet_match_score < int(self.icd_mdt_parameters['wavelet_match_threshold']):  # Set the match threshold to 70%
                mismatch_count += 1


                if mismatch_count >= 6:

                    print("Morphology Matched: VT detected")
                    self.rvshock_wavelet_pw.clear()
                    text_item = pg.TextItem(text="Morphology Matched: VT detected", color='k', anchor=(0, 0))
                    self.rvshock_wavelet_pw.addItem(text_item)


    def validate_template_btn_clicked(self):
        self.validate_template(self.saved_wavelet_template, self.icd_memory['wavelet_beats'])

    def check_wavelet_match_btn_clicked(self):
        self.check_recent_waveforms(self.saved_wavelet_template, self.icd_memory['wavelet_match_beats'])
        coeffs = pywt.wavedec(unknown_coeffs, 'haar', level=4)
        beat = pywt.waverec(coeffs, 'haar')
        self.wavelet_check_pw.clear()
        self.wavelet_check_pw.clear()
        self.wavelet_check_pw.plot(self.saved_wavelet_template, pen=pg.mkPen(color='r', width=3))
        self.wavelet_check_pw.plot(self.beat_plot, pen=pg.mkPen(color='k', width=2))
        self.wavelet_check_pw.plot(beat, pen=pg.mkPen(color='b', width=2))
        self.rvshock_wavelet_pw.clear()
        text_item  = pg.TextItem(text=f"Match score:\n {self.wavelet_match_score:.2f}")
        self.rvshock_wavelet_pw.addItem(text_item)