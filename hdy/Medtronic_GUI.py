import csv
import pandas as pd
import scipy
from scipy import signal
import numpy as np
import mmt
from collections import deque
from collections import Counter
from . import *
from PySide6 import QtGui, QtCore, QtWidgets
import math
import pyqtgraph as pg
import pywt
import time
import os
import datetime
import sys
import hdy


class TimeAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fs = 1000
    def tickStrings(self, values, scale, spacing):
        if spacing > self.fs:
            result = [self.ms_to_mmss(value) for value in values]
        else:
            result = [self.ms_to_mmssmm(value) for value in values]
        return result

    def ms_to_mmss(self, value):
        sec = int(value / self.fs) % 60
        min = int(value / (self.fs * 60))
        return "{min}:{sec:02d}".format(min=min, sec=sec)

    def ms_to_mmssmm(self, value):
        ms = int(value) % self.fs
        sec = int(value / self.fs) % 60
        min = int(value / (self.fs * 60))
        return "{min}:{sec:02d}.{ms:03d}".format(min=min, sec=sec, ms=ms)

class MagiQuantLaser(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.statusBar().showMessage('Ready')

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')

        openFile = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.showFileOpen)

        fileMenu.addAction(openFile)
        menubar.addMenu(fileMenu)

        self.lasergui = LaserGui()
        self.setCentralWidget(self.lasergui)
        self.setWindowTitle('MagiQuant Laser')
        self.show()

class MedtronicGui(QtWidgets.QMainWindow):

    def __init__(self, laser_exp: object, parent: object = None) -> object:

        super().__init__(parent=parent)

        self.laser_exp = laser_exp  # LaserAnalysis1 object
        self.index1 = 0  # index of current data to update vvi plots
        self.trace_view = 0  # index of current data to update vvi plots

        # Initialize Parameters
        self.icd_mdt_parameters = {}
        self.icd_mdt_parameters['rvst_value'] = 0.3  # value of rv_sens_threshold
        self.icd_mdt_parameters['pvsb_value'] = 120  # value of postvsblanking
        self.icd_mdt_parameters['sensitivity'] = 0.75  # Initial sensitivity at 75% of peak EGM amplitude
        self.icd_mdt_parameters['amplitude'] = 0  # Placeholder for the amplitude of the sensed R-wave
        self.icd_mdt_parameters['onset_pct'] = 0.81  # Nominal setting
        self.icd_mdt_parameters['stability'] = 40  # Nominal Setting
        self.icd_mdt_parameters['vf_tcl'] = 320  # 188 bpm
        self.icd_mdt_parameters['fvt_tcl'] = 0  # OFF
        self.icd_mdt_parameters['vt_tcl'] = 0  # OFF
        self.icd_mdt_parameters['vf_min_nid'] = 30  # nominal
        self.icd_mdt_parameters['vf_max_nid'] = 40  # nominal
        self.icd_mdt_parameters['vt_nid'] = 0  # OFF
        self.icd_mdt_parameters['monitor_tcl'] = 0  # OFF

        self._haemStable = False
        self._stability = False
        self._haem_comp = False
        self._atp_del1 = False
        self._atp_del2 = False
        self.vt_shock_counter = 0
        self.haem_comp_alert_viewed = False
        self.cancel_tx_viewed = False
        self._atp1_viewed = False
        self._atp2_viewed = False
        self.hasrun = False

        self.x_marker = 0
        self.next_r = 0

        # Fixing and Correlating Data
        # self.resample_data(1000, 512)
        print("shape", self.laser_exp.rvbip.data.shape)
        print("size", self.laser_exp.rvbip.data.size)
        print("length, ", len(self.laser_exp.rvbip.data))

        # Filtering ECGs
        self.ecg_filter()
        # self.fix_lag()
        self.amplifier()
        # self.rectifier()

        # Initialize Memory
        self.icd_memory = {}
        self.icd_memory['r_peak'] = deque(maxlen=8)
        self.icd_memory['last_r_peak'] = 0
        self.icd_memory['rr_intervals'] = deque(maxlen=8)
        self.icd_memory['onset'] = deque(maxlen=8)
        self.icd_memory['stability'] = deque(maxlen=4)
        self.icd_memory['high_sens'] = deque(maxlen=36)
        self.icd_memory['low_sens'] = deque(maxlen=16)


        self.vt_counter = 0

        self.icd_memory['avg_interval_bin'] = deque(maxlen=4)
        # self.icd_memory['nid_vtzone1'] = deque(maxlen=30)
        # self.icd_memory['nid_vtzone2'] = deque(maxlen=30)
        # self.icd_memory['nid_vf'] = deque(maxlen=30)
        # self.icd_memory['nid_nsr'] = deque(maxlen=10)
        # self.icd_memory['nid_discard'] = deque(maxlen=10)
        self.icd_memory['vf_check'] = deque(maxlen=8)
        self.icd_memory['median_rr_ints'] = deque(maxlen=12)

        self.icd_memory['wavelet'] = deque(maxlen=8)  # use RV Shock lead
        self.icd_memory['haem_label'] = deque(maxlen=10)
        self.icd_memory['rhythm_label'] = deque(maxlen=max(self.icd_mdt_parameters['vt_nid'] , self.icd_mdt_parameters['vf_max_nid'] ))

        self.icd_memory['active_tachy'] = False

        #  GUI Setups
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab1 = QtWidgets.QWidget()

        self.wavelet_gui = hdy.Wavelet_GUI(self.laser_exp)
        self.beatplot_gui = hdy.BeatPlot_GUI(self.laser_exp)
        self.m_vvi_gui = hdy.MedtronicVVI_GUI(self.laser_exp)
        self.therapies_gui = hdy.vtherapies_GUI(self.laser_exp)
        self.sensing = hdy.sensing(self.laser_exp)

        self.tab_widget.addTab(self.tab1, "Plot Overview")
        self.tab_widget.addTab(self.beatplot_gui, "Beat Plot")
        self.tab_widget.addTab(self.m_vvi_gui, "Medtronic VVI Settings")
        self.tab_widget.addTab(self.wavelet_gui, "Wavelet")
        self.tab_widget.addTab(self.therapies_gui, "VF/VT Therapies")
        self.setup_gui()
        self.setup_plots()
        self.update_data()
        self.setCentralWidget(self.tab_widget)
        # Setting main window as Central Widget
        self.resize(800, 600)

        self.show()

    def setup_gui(self):
        # Setting PyQtGraph Configurations
        pg.setConfigOptions(antialias=True, background='w')

        # Setting Window Title
        self.setWindowTitle('Medtronic ICD')

        # Creating Plot and Button Layouts
        self.plot_layout = QtWidgets.QVBoxLayout()
        self.btn_layout = QtWidgets.QVBoxLayout()

        # Toggles for viewer settins
        self.ralead_toggle = QtWidgets.QCheckBox("View RA")
        self.ralead_toggle.setChecked(True)
        self.ralead_toggle.stateChanged.connect(self.ralead_toggle_changed)

        self.rvbip_toggle = QtWidgets.QCheckBox("View RV Bipolar")
        self.rvbip_toggle.setChecked(True)
        self.rvbip_toggle.stateChanged.connect(self.rvbip_toggle_changed)

        self.rvshock_toggle = QtWidgets.QCheckBox("View RV Shock")
        self.rvshock_toggle.setChecked(True)
        self.rvshock_toggle.stateChanged.connect(self.rvshock_toggle_changed)

        self.lvlead_toggle = QtWidgets.QCheckBox("View LV Lead")
        self.lvlead_toggle.setChecked(False)
        self.lvlead_toggle.stateChanged.connect(self.lvlead_toggle_changed)

        self.bipecg_toggle = QtWidgets.QCheckBox("View Bipolar ECG")
        self.bipecg_toggle.setChecked(True)
        self.bipecg_toggle.stateChanged.connect(self.bipecg_toggle_changed)

        self.ecg3_toggle = QtWidgets.QCheckBox("View 3 lead ECG")
        self.ecg3_toggle.setChecked(False)
        self.ecg3_toggle.stateChanged.connect(self.ecg3_toggle_changed)

        self.bp_toggle = QtWidgets.QCheckBox("View BP")
        self.bp_toggle.setChecked(False)
        self.bp_toggle.stateChanged.connect(self.bp_toggle_changed)

        self.laser1_toggle = QtWidgets.QCheckBox("View Laser1")
        self.laser1_toggle.setChecked(True)
        self.laser1_toggle.stateChanged.connect(self.laser1_toggle_changed)

        self.laser2_toggle = QtWidgets.QCheckBox("View Laser2")
        self.laser2_toggle.setChecked(False)
        self.laser2_toggle.stateChanged.connect(self.laser2_toggle_changed)

        # self.hrv_toggle = QtWidgets.QCheckBox("Show RR Interval")
        # self.hrv_toggle.stateChanged.connect(self.hrv_toggle_changed)

        # Combo boxes for viewer settings
        self.hrv_combo = QtWidgets.QComboBox()
        self.hrv_combo.addItems("No-RR-Intervals BipECG 3L-ECG RVbip RVShock".split())
        self.hrv_combo.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        self.hrv_combo.currentIndexChanged.connect(self.update_data)

        self.rect_combo = QtWidgets.QComboBox()
        self.rect_combo.addItems("None Rectifier Derivative Zero-Crossing Peaks SqRectifier".split())
        self.rect_combo.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        self.rect_combo.currentIndexChanged.connect(self.update_data)

        self.laser_combo = QtWidgets.QComboBox()
        self.laser_combo.addItems("Raw Filtered Log-Filtered new new-convolved min-max-norm".split())
        self.laser_combo.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        self.laser_combo.currentIndexChanged.connect(self.update_data)

        # Buttons for Calculations
        self.calc_move_btn = QtWidgets.QPushButton("Move Calc ROI")
        self.calc_move_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        self.calc_move_btn.clicked.connect(self.calc_region_move)

        self.calc_10s_btn = QtWidgets.QPushButton("Set 10s")
        self.calc_10s_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        self.calc_10s_btn.clicked.connect(self.calc_10s)

        self.calc_btn = QtWidgets.QPushButton("Calc")
        self.calc_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        self.calc_btn.clicked.connect(self.calc_results)

        # Creating Labels
        self.begin_lbl = QtWidgets.QLabel()
        self.end_lbl = QtWidgets.QLabel()
        self.laser1_value = QtWidgets.QLabel()
        self.laser2_value = QtWidgets.QLabel()
        self.laser1_conf = QtWidgets.QLabel()
        self.laser2_conf = QtWidgets.QLabel()
        self.hr_value = QtWidgets.QLabel()
        self.rr_value = QtWidgets.QLabel()
        self.sbp_value = QtWidgets.QLabel()
        self.map_value = QtWidgets.QLabel()
        self.m_settings_lbl = QtWidgets.QLabel()
        self.spacer_lbl = QtWidgets.QLabel()
        self.viewing_options_lbl = QtWidgets.QLabel()
        self.calc_lbl = QtWidgets.QLabel()
        self.zipfl_lbl = QtWidgets.QLabel()

        # Adding Widgets to Button Layout
        # Viewing Option Widgets in Button Layout
        self.btn_layout.addWidget(self.viewing_options_lbl)
        self.btn_layout.addWidget(self.ralead_toggle)
        self.btn_layout.addWidget(self.rvbip_toggle)
        self.btn_layout.addWidget(self.rvshock_toggle)
        self.btn_layout.addWidget(self.lvlead_toggle)
        self.btn_layout.addWidget(self.bipecg_toggle)
        self.btn_layout.addWidget(self.ecg3_toggle)
        self.btn_layout.addWidget(self.bp_toggle)
        self.btn_layout.addWidget(self.laser1_toggle)
        self.btn_layout.addWidget(self.laser2_toggle)
        self.btn_layout.addWidget(self.hrv_combo)
        self.btn_layout.addWidget(self.rect_combo)
        self.btn_layout.addWidget(self.laser_combo)
        self.btn_layout.addWidget(self.spacer_lbl)

        # Calculations Widgets in Button Layout
        self.btn_layout.addWidget(self.calc_lbl)
        self.btn_layout.addWidget(self.calc_move_btn)
        self.btn_layout.addWidget(self.calc_10s_btn)
        self.btn_layout.addWidget(self.calc_btn)
        self.btn_layout.addWidget(self.spacer_lbl)

        # Results Widgets in Button Layout
        self.btn_layout.addWidget(self.begin_lbl)
        self.btn_layout.addWidget(self.end_lbl)
        self.btn_layout.addWidget(self.laser1_value)
        self.btn_layout.addWidget(self.laser1_conf)
        self.btn_layout.addWidget(self.laser2_value)
        self.btn_layout.addWidget(self.laser2_conf)
        self.btn_layout.addWidget(self.rr_value)
        self.btn_layout.addWidget(self.hr_value)
        self.btn_layout.addWidget(self.sbp_value)
        self.btn_layout.addWidget(self.map_value)

        # Plot Widgets
        time_axis = TimeAxisItem(orientation='bottom')

        self.bipecg_pw = pg.PlotWidget(axisItems={'bottom': time_axis})
        self.bipecg_pi = self.bipecg_pw.getPlotItem()
        self.bipecg_pi.setLabel(axis='left', text="Bipolar ECG")
        self.bipecg_plt = self.bipecg_pi.plot()
        self.bipecg_peak_plt = self.bipecg_pi.plot()

        self.rvbip_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.rvbip_pi = self.rvbip_pw.getPlotItem()
        self.rvbip_pi.setLabel(axis='left', text="RV Bip")
        self.rvbip_plt = self.rvbip_pi.plot()
        self.rvbip_peak_plt = self.rvbip_pi.plot()

        self.rvshock_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.rvshock_pi = self.rvshock_pw.getPlotItem()
        self.rvshock_pi.setLabel(axis='left', text="RV Shock")
        self.rvshock_plt = self.rvshock_pi.plot()
        self.rvshock_peak_plt = self.rvshock_pi.plot()

        self.ralead_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.ralead_pi = self.ralead_pw.getPlotItem()
        self.ralead_pi.setLabel(axis='left', text="RA")
        self.ralead_plt = self.ralead_pi.plot()
        self.ralead_peak_plt = self.ralead_pi.plot()

        self.lvlead_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.lvlead_pi = self.lvlead_pw.getPlotItem()
        self.lvlead_pi.setLabel(axis='left', text="LV")
        self.lvlead_plt = self.lvlead_pi.plot()
        self.lvlead_pw.hide()

        self.ecg3_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.ecg3_pi = self.ecg3_pw.getPlotItem()
        self.ecg3_pi.setLabel(axis='left', text="3-Lead ECG")
        self.ecg3_plt = self.ecg3_pi.plot()
        self.ecg3_peak_plt = self.ecg3_pi.plot()
        self.ecg3_pw.hide()

        self.pressure_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.pressure_pi = self.pressure_pw.getPlotItem()
        self.pressure_pi.setLabel(axis='left', text="BP")
        self.pressure_plt = self.pressure_pi.plot()
        self.pressure_pw.hide()

        self.laser1_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.laser1_pi = self.laser1_pw.getPlotItem()
        self.laser1_pi.setLabel(axis='left', text="LDPM1")
        self.laser1_plt = self.laser1_pi.plot()
        self.laser1_peak_plt = self.laser1_pi.plot()
        self.envelope1_plt = self.laser1_pi.plot()

        self.laser2_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.laser2_pi = self.laser2_pw.getPlotItem()
        self.laser2_pi.setLabel(axis='left', text="LDPM2")
        self.laser2_plt = self.laser2_pi.plot()
        self.envelope2_plt = self.laser2_pi.plot()
        self.laser2_peak_plt = self.laser2_pi.plot()
        self.laser2_pw.hide()

        self.overview_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.overview_pi = self.overview_pw.getPlotItem()
        self.overview_pi.setLabel(axis='left', text="Overview")
        self.overview_plt = self.overview_pi.plot()

        # Adding Widgets to Plot Layout
        self.plot_layout.addWidget(self.rvbip_pw, stretch=1)
        self.plot_layout.addWidget(self.rvshock_pw, stretch=1)
        self.plot_layout.addWidget(self.ralead_pw, stretch=1)
        self.plot_layout.addWidget(self.lvlead_pw, stretch=1)
        self.plot_layout.addWidget(self.bipecg_pw, stretch=1)
        self.plot_layout.addWidget(self.ecg3_pw, stretch=1)
        self.plot_layout.addWidget(self.pressure_pw, stretch=1)
        self.plot_layout.addWidget(self.laser1_pw, stretch=1)
        self.plot_layout.addWidget(self.laser2_pw, stretch=1)
        self.plot_layout.addWidget(self.overview_pw, stretch=1)
        self.plot_layout.addWidget(self.zipfl_lbl)

        # Adding Label Text to Layout
        self.zipfl_lbl.setText(str(self.laser_exp.zip_fl))

        # Fixing Plot and Button Layouts
        self.plot_layout.setSpacing(0)
        self.plot_layout.setContentsMargins(0, 0, 0, 200)

        self.btn_layout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        # Creating Main Layout
        self.main_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addLayout(self.btn_layout)
        self.main_layout.addLayout(self.plot_layout)
        # self.main_layout.setStretch(0, 1)
        self.main_layout.setStretch(1, 5)
        self.main_layout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        self.tab1.setLayout(self.main_layout)

    # Setup GUI and Plots
    def setup_plots(self):

        self.overview_region = pg.LinearRegionItem()
        self.overview_region.setZValue(10)
        self.overview_pi.addItem(self.overview_region)
        self.overview_region.sigRegionChanged.connect(self.overview_region_changed)
        self.overview_region.setRegion((0, self.laser_exp.rvbip.data.shape[0]))

        self.calc_region = pg.LinearRegionItem()
        self.calc_region.setZValue(-10)
        self.rvbip_pi.addItem(self.calc_region)
        self.calc_region.sigRegionChanged.connect(self.calc_region_changed)

        self.rvbip_pw.sigRangeChanged.connect(self.overview_region_update)
        self.ralead_pw.sigRangeChanged.connect(self.overview_region_update)
        self.rvshock_pw.sigRangeChanged.connect(self.overview_region_update)
        self.lvlead_pw.sigRangeChanged.connect(self.overview_region_update)
        self.bipecg_pw.sigRangeChanged.connect(self.overview_region_update)
        self.ecg3_pw.sigRangeChanged.connect(self.overview_region_update)
        self.pressure_pw.sigRangeChanged.connect(self.overview_region_update)
        self.laser1_pw.sigRangeChanged.connect(self.overview_region_update)
        self.laser2_pw.sigRangeChanged.connect(self.overview_region_update)

        self.rvbip_pi.getAxis('left').setWidth(w=50)
        self.rvbip_pi.getAxis('left').setStyle(showValues=False)
        self.ralead_pi.getAxis('left').setWidth(w=50)
        self.ralead_pi.getAxis('left').setStyle(showValues=False)
        self.rvshock_pi.getAxis('left').setWidth(w=50)
        self.rvshock_pi.getAxis('left').setStyle(showValues=False)
        self.lvlead_pi.getAxis('left').setWidth(w=50)
        self.lvlead_pi.getAxis('left').setStyle(showValues=False)
        self.bipecg_pi.getAxis('left').setWidth(w=50)
        self.bipecg_pi.getAxis('left').setStyle(showValues=False)
        self.ecg3_pi.getAxis('left').setWidth(w=50)
        self.ecg3_pi.getAxis('left').setStyle(showValues=False)
        self.pressure_pi.getAxis('left').setWidth(w=50)
        self.laser1_pi.getAxis('left').setWidth(w=50)
        self.laser2_pi.getAxis('left').setWidth(w=50)
        self.overview_pi.getAxis('left').setWidth(w=50)
        self.overview_pi.getAxis('left').setStyle(showValues=False)

        # self.x_marker = self.max_x_list[self.index1]
        try:
            # self.max_x_list, self.rr_list, self.amplitude_list, self.rpeaks = data_list(self.laser_exp.rvbip.data,self.icd_mdt_parameters['rvst_value'], self.icd_mdt_parameters['pvsb_value'])
            # self.max_x_list, self.rr_list, self.amplitude_list, self.rpeaks = data_list(self.laser_exp.rvbip.data,
            #                                                                self.icd_mdt_parameters['rvst_value'],
            #                                                                self.icd_mdt_parameters['pvsb_value'])
            self.max_x_list, self.rr_list = data_list(self.laser_exp.rvbip.data,
                                                      self.icd_mdt_parameters['rvst_value'],
                                                      self.icd_mdt_parameters['pvsb_value'])
        except:
            # self.max_x_list, self.rr_list, self.amplitude_list, self.rpeaks = data_list(self.laser_exp.bipecg.data,self.icd_mdt_parameters['rvst_value'], self.icd_mdt_parameters['pvsb_value'])
            # self.max_x_list, self.rr_list, self.amplitude_list, self.rpeaks = data_list(self.laser_exp.rvbip.data,
            #                                                                self.icd_mdt_parameters['rvst_value'],
            #                                                                self.icd_mdt_parameters['pvsb_value'])
            self.max_x_list, self.rr_list = data_list(self.laser_exp.bipecg.data,
                                                      self.icd_mdt_parameters['rvst_value'],
                                                      self.icd_mdt_parameters['pvsb_value'])

        # self.max_x_list, self.rr_list = data_list(self.laser_exp.rvbip.data)

    def update_data(self):
        samples = self.laser_exp.pressure.data.shape[0]
        ecg_hint = self.laser_exp.hints['Period']
        print(ecg_hint)

        self.pressure_plt.setData(x=np.arange(samples), y=self.laser_exp.pressure.data, pen='#fc0303', antialise=True,
                                  autoDownsample=True, clipToView=True)

        if self.hrv_combo.currentIndex() == 0:
            self.rvshock_peak_plt.clear()
            self.rvbip_peak_plt.clear()
            self.ecg3_peak_plt.clear()
            self.bipecg_peak_plt.clear()

        elif self.hrv_combo.currentIndex() == 1:
            self.laser_exp.bipecg.calc_ecg_peaks(ecg_hint=ecg_hint)
            self.hrv_xs = self.laser_exp.bipecg.peaks_sample[1:]
            self.hrv_ys = 60000 / np.diff(self.laser_exp.bipecg.peaks_sample)
            self.bipecg_peaks_x = self.laser_exp.bipecg.peaks_sample
            self.bipecg_peaks_y = self.laser_exp.bipecg.peaks_value

            self.bipecg_peak_plt.setData(x=self.bipecg_peaks_x, y=self.bipecg_peaks_y,
                                         pen=None, symbol='o', size=4, pxMode=True,
                                         antialise=True, autoDownsample=False, downsampleMethod='peak',
                                         clipToView=False)

            self.rvshock_peak_plt.clear()
            self.rvbip_peak_plt.clear()
            self.ecg3_peak_plt.clear()

        elif self.hrv_combo.currentIndex() == 2:
            self.laser_exp.ecg3.calc_ecg_peaks(ecg_hint=ecg_hint)
            self.hrv_xs = self.laser_exp.ecg3.peaks_sample[1:]
            self.hrv_ys = 60000 / np.diff(self.laser_exp.ecg3.peaks_sample)
            self.ecg3_peaks_x = self.laser_exp.ecg3.peaks_sample
            self.ecg3_peaks_y = self.laser_exp.ecg3.peaks_value

            self.ecg3_peak_plt.setData(x=self.ecg3_peaks_x, y=self.ecg3_peaks_y,
                                       pen=None, symbol='o', size=4, pxMode=True,
                                       antialise=True, autoDownsample=False, downsampleMethod='peak', clipToView=False)
            self.rvshock_peak_plt.clear()
            self.rvbip_peak_plt.clear()
            self.bipecg_peak_plt.clear()
        elif self.hrv_combo.currentIndex() == 3:
            self.laser_exp.rvbip.calc_ecg_peaks(ecg_hint=ecg_hint)
            self.hrv_xs1 = self.laser_exp.rvbip.peaks_sample[1:]
            self.hrv_ys1 = 60000 / np.diff(self.laser_exp.rvbip.peaks_sample)
            self.bipecg_peaks_x1 = self.laser_exp.rvbip.peaks_sample
            self.bipecg_peaks_y1 = self.laser_exp.rvbip.peaks_value

            bipecg_peaks_x = self.bipecg_peaks_x1
            bipecg_peaks_y = self.bipecg_peaks_y1

            self.rvbip_peak_plt.setData(x=bipecg_peaks_x, y=bipecg_peaks_y,
                                        pen=None, symbol='o', size=4, pxMode=True,
                                        antialise=True, autoDownsample=False, downsampleMethod='peak', clipToView=False)
            self.rvshock_peak_plt.clear()
            self.ecg3_peak_plt.clear()
            self.bipecg_peak_plt.clear()
        elif self.hrv_combo.currentIndex() == 4:

            self.laser_exp.rvshock.calc_ecg_peaks(ecg_hint="rvshock")
            self.hrv_xs = self.laser_exp.rvshock.peaks_sample[1:]
            self.hrv_ys = 60000 / np.diff(self.laser_exp.rvshock.peaks_sample)
            self.ecg_peaks_x = self.laser_exp.rvshock.peaks_sample
            self.ecg_peaks_y = self.laser_exp.rvshock.peaks_value


            self.rvshock_peak_plt.setData(x=self.ecg_peaks_x, y=self.ecg_peaks_y,
                                          pen=None, symbol='o', size=4, pxMode=True,
                                          antialise=True, autoDownsample=False, downsampleMethod='peak',
                                          clipToView=False)

            self.rvbip_peak_plt.clear()
            self.ecg3_peak_plt.clear()
            self.bipecg_peak_plt.clear()

        if self.rect_combo.currentIndex() == 0:
            bip_ecg = self.laser_exp.bipecg.data.copy()
            ecg3 = self.laser_exp.ecg3.data.copy()
            rvshock = self.laser_exp.rvshock.data.copy()
            rvbip = self.laser_exp.rvbip.data.copy()
            ra = self.laser_exp.ralead.data.copy()
            lv = self.laser_exp.lvlead.data.copy()

        elif self.rect_combo.currentIndex() == 1:
            self.rectifier()
            bip_ecg = self.rect_bipecg
            ecg3 = self.rect_ecg3
            rvshock = self.rect_rvshock
            rvbip = self.rect_rvbip
            ra = self.rect_ralead
            lv = self.rect_lvlead

        elif self.rect_combo.currentIndex() == 2:
            self.derivatives()
            bip_ecg = self.gradient_bipecg
            ecg3 = self.gradient_ecg3
            rvshock = self.gradient_rvshock
            rvbip = self.gradient_rvbip
            ra = self.gradient_ralead
            lv = self.gradient_lvlead

        elif self.rect_combo.currentIndex() == 3:
            bip_ecg = self.laser_exp.bipecg.data
            ecg3 = self.laser_exp.ecg3.data
            rvshock = self.laser_exp.rvshock.data
            rvbip = self.laser_exp.rvbip.data
            ra = self.laser_exp.ralead.data
            lv = self.laser_exp.lvlead.data
            self.sq_rectifier()
            self.derivatives()
            self.zero_crossings()

            self.peaks_zero_crossings()
            self.bipecg_peak_plt.setData(y=bip_ecg[self.bip_peaks], x=self.bip_peaks,
                                            pen=None, symbol='o', size=4, pxMode=True,
                                            antialise=True, autoDownsample=False, downsampleMethod='peak',
                                            clipToView=False)


        elif self.rect_combo.currentIndex() == 4:
            bip_ecg = self.laser_exp.bipecg.data
            ecg3 = self.laser_exp.ecg3.data
            rvshock = self.laser_exp.rvshock.data
            rvbip = self.laser_exp.rvbip.data
            ra = self.laser_exp.ralead.data
            lv = self.laser_exp.lvlead.data
            self.sq_rectifier()
            self.derivatives()
            self.zero_crossings()
            self.peaks_zero_crossings()
            self.bipecg_peak_plt.setData(y=bip_ecg[self.filt_bip_peaks], x=self.filt_bip_peaks,
                                         pen=None, symbol='o', size=4, pxMode=True,
                                         antialise=True, autoDownsample=False, downsampleMethod='peak',
                                         clipToView=False)


        elif self.rect_combo.currentIndex() == 5:
            self.sq_rectifier()
            bip_ecg = self.sqrect_bipecg
            ecg3 = self.sqrect_ecg3
            rvshock = self.sqrect_rvshock
            rvbip = self.sqrect_rvbip
            ra = self.sqrect_ralead
            lv = self.sqrect_lvlead

        self.bipecg_plt.setData(x=np.arange(samples), y=bip_ecg, pen='02B83A', symbol=None,
                                antialise=True,
                                autoDownsample=True, clipToView=True)
        self.ecg3_plt.setData(x=np.arange(samples), y=ecg3, pen='#02B83A', antialise=True,
                              autoDownsample=True,
                              clipToView=True)
        self.ralead_plt.setData(x=np.arange(samples), y=ra, pen='#732F9B', antialise=True,
                                autoDownsample=True,
                                clipToView=True)
        self.rvshock_plt.setData(x=np.arange(samples), y=rvshock, pen='#732F9B', antialise=True,
                                 autoDownsample=True, clipToView=True)
        self.rvbip_plt.setData(x=np.arange(samples), y=rvbip, pen='#732F9B', antialise=True,
                               autoDownsample=True,
                               clipToView=True)
        self.lvlead_plt.setData(x=np.arange(samples), y=lv, pen='#732F9B', antialise=True,
                                autoDownsample=True,
                                clipToView=True)

        if self.laser_combo.currentIndex() == 0:
            laser1_data = self.laser_exp.laser1.data
            laser2_data = self.laser_exp.laser2.data
        elif self.laser_combo.currentIndex() == 1:
            laser1_data = mmt.butter_bandpass_filter(self.laser_exp.laser1.data, 0.5, 25, 1000, order=2)
            laser2_data = mmt.butter_bandpass_filter(self.laser_exp.laser2.data, 0.5, 25, 1000, order=2)
            envelope1_data = np.abs(scipy.signal.hilbert(laser1_data))
            envelope1_data = mmt.butter_lowpass_filter(envelope1_data, 1, 1000, 2)
            envelope2_data = np.abs(scipy.signal.hilbert(laser2_data))
            envelope2_data = mmt.butter_lowpass_filter(envelope2_data, 1, 1000, 2)
        elif self.laser_combo.currentIndex() == 2:
            laser1_data = mmt.butter_bandpass_filter(
                np.log(self.laser_exp.laser1.data + 20) + self.laser_exp.laser1.data / 100, 0.5, 25, 1000, order=2)
            laser2_data = mmt.butter_bandpass_filter(
                np.log(self.laser_exp.laser2.data + 20) + self.laser_exp.laser2.data / 100, 0.5, 25, 1000, order=2)
            envelope1_data = np.abs(scipy.signal.hilbert(laser1_data))
            envelope1_data = mmt.butter_lowpass_filter(envelope1_data, 1, 1000, 2)
            envelope2_data = np.abs(scipy.signal.hilbert(laser2_data))
            envelope2_data = mmt.butter_lowpass_filter(envelope2_data, 1, 1000, 2)
        elif self.laser_combo.currentIndex() == 3:
            sos = scipy.signal.iirfilter(4, Wn=[0.1, 2.5], fs=500, btype="bandpass",
                                         ftype="butter", output="sos")
            laser1_data = scipy.signal.sosfilt(sos, self.laser_exp.laser1.data)
            laser2_data = scipy.signal.sosfilt(sos, self.laser_exp.laser2.data)
        elif self.laser_combo.currentIndex() == 4:
            sos = scipy.signal.iirfilter(4, Wn=[0.1, 2.5], fs=1000, btype="bandpass",
                                         ftype="butter", output="sos")
            laser1_data = np.convolve(self.laser_exp.laser1.data, np.ones((11,)) / 11, mode='valid')
            laser2_data = np.convolve(self.laser_exp.laser2.data, np.ones((11,)) / 11, mode='valid')
            laser1_data = scipy.signal.sosfilt(sos, laser1_data)
            laser2_data = scipy.signal.sosfilt(sos, laser2_data)
        elif self.laser_combo.currentIndex() == 5:
            self.norm_laser()
            laser1_data = self.norm_laser1
            laser2_data = self.norm_laser2

        self.laser1_plt.setData(x=np.arange(samples), y=laser1_data, pen='#03c6fc', pensize=50, antialise=True,
                                autoDownsample=True, clipToView=True)
        self.laser2_plt.setData(x=np.arange(samples), y=laser2_data, pen='#03c6fc', antialise=True, autoDownsample=True,
                                clipToView=True)

        self.overview_plt.setData(x=np.arange(samples), y=self.laser_exp.rvbip.data, pen='#0FA00F', symbol=None,
                                  antialise=True, autoDownsample=True, clipToView=True)

    # FIXING DATA
    def fix_lag(self):
        #     Correlate ECGs
        corr = scipy.signal.correlate(self.laser_exp.rvshock.data, self.laser_exp.bipecg.data, mode="full")
        lags = scipy.signal.correlation_lags(self.laser_exp.rvshock.data.size, self.laser_exp.bipecg.data.size,
                                             mode="full")
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

            self.laser_exp.rvbip.data = self.laser_exp.rvbip.data[0:self.laser_exp.bipecg.data.size]
            self.laser_exp.rvshock.data = self.laser_exp.rvshock.data[0:self.laser_exp.bipecg.data.size]
            self.laser_exp.ralead.data = self.laser_exp.ralead.data[0:self.laser_exp.bipecg.data.size]
            self.laser_exp.lvlead.data = self.laser_exp.lvlead.data[0:self.laser_exp.bipecg.data.size]

        else:
            print(f"Removing {len(remove)} samples from ICD Leads")
            self.laser_exp.rvbip.data = np.delete(self.laser_exp.rvbip.data, remove, axis=0)
            self.laser_exp.rvshock.data = np.delete(self.laser_exp.rvshock.data, remove, axis=0)
            self.laser_exp.ralead.data = np.delete(self.laser_exp.ralead.data, remove, axis=0)
            self.laser_exp.lvlead.data = np.delete(self.laser_exp.lvlead.data, remove, axis=0)

            self.laser_exp.ecg3.data = self.laser_exp.ecg3.data[0:self.laser_exp.rvshock.data.size]
            self.laser_exp.bipecg.data = self.laser_exp.bipecg.data[0:self.laser_exp.rvshock.data.size]
            self.laser_exp.laser1.data = self.laser_exp.laser1.data[0:self.laser_exp.rvshock.data.size]
            self.laser_exp.laser2.data = self.laser_exp.laser2.data[0:self.laser_exp.rvshock.data.size]
            self.laser_exp.pressure.data = self.laser_exp.pressure.data[0:self.laser_exp.rvshock.data.size]

    def resample_data(self, original_fs, desired_fs):
        #        resample_ratio = desired_fs / original_fs
        sec = self.laser_exp.rvbip.data.size / original_fs
        new_length = int(sec * desired_fs)
        self.laser_exp.pressure.data = scipy.signal.resample(self.laser_exp.pressure.data, new_length)
        self.laser_exp.ecg3.data = scipy.signal.resample(self.laser_exp.ecg3.data, new_length)
        self.laser_exp.bipecg.data = scipy.signal.resample(self.laser_exp.bipecg.data, new_length)
        self.laser_exp.laser1.data = scipy.signal.resample(self.laser_exp.laser1.data, new_length)
        self.laser_exp.laser2.data = scipy.signal.resample(self.laser_exp.laser2.data, new_length)
        self.laser_exp.ralead.data = (scipy.signal.resample(self.laser_exp.ralead.data, new_length))
        self.laser_exp.rvbip.data = scipy.signal.resample(self.laser_exp.rvbip.data, new_length)
        self.laser_exp.rvshock.data = scipy.signal.resample(self.laser_exp.rvshock.data, new_length)
        self.laser_exp.lvlead.data = scipy.signal.resample(self.laser_exp.lvlead.data, new_length)

    def ecg_filter(self):
        bipecg_d = self.laser_exp.bipecg.data
        ecg_sos = scipy.signal.butter(5, (0.5, 25), 'band', fs=512, output='sos')
        try:
            self.laser_exp.ecg3.data = scipy.signal.sosfilt(ecg_sos, bipecg_d)
        except:
            self.laser_exp.bipecg.data = scipy.signal.sosfilt(ecg_sos, self.laser_exp.ecg3.data)

    def amplifier(self):
        # Fixing Voltage on leads
        self.laser_exp.ralead.data = self.laser_exp.ralead.data * 10
        self.laser_exp.rvbip.data  = self.laser_exp.rvbip.data * 10
        self.laser_exp.rvshock.data = self.laser_exp.rvshock.data * 10
        self.laser_exp.lvlead.data = self.laser_exp.lvlead.data * 10

    def norm_laser(self):
        # self.norm_laser1 = self.laser_exp.laser1.data / np.max(self.laser_exp.laser1.data)
        # self.norm_laser2 = self.laser_exp.laser2.data / np.max(self.laser_exp.laser2.data)
        min_laser1 = np.min(self.laser_exp.laser1.data )
        min_laser2 = np.min(self.laser_exp.laser2.data)
        max_laser1 = np.max(self.laser_exp.laser1.data )
        max_laser2 = np.max(self.laser_exp.laser2.data)

        self.norm_laser1 = (self.laser_exp.laser1.data - min_laser1) / (max_laser1 - min_laser1)
        self.norm_laser2 = (self.laser_exp.laser2.data - min_laser2) / (max_laser2 - min_laser2)

    def rectifier(self):
        # Rectifying the ECG
        self.rect_ecg3 = abs(self.laser_exp.ecg3.data)
        self.rect_bipecg = abs(self.laser_exp.bipecg.data)
        self.rect_ralead = abs(self.laser_exp.ralead.data)
        self.rect_rvbip = abs(self.laser_exp.rvbip.data)
        self.rect_rvshock = abs(self.laser_exp.rvshock.data)
        self.rect_lvlead = abs(self.laser_exp.lvlead.data)
        self.rect_laser1 = abs(self.laser_exp.laser1.data)
        self.rect_laser2 = abs(self.laser_exp.laser2.data)
    
    def sq_rectifier(self):
        self.rectifier()
        # Rectifying the ECG
        self.sqrect_ecg3 = (self.rect_ecg3 ** 2)*10
        self.sqrect_bipecg = (self.rect_bipecg ** 2)*10
        self.sqrect_ralead = (self.rect_ralead ** 2)*10
        self.sqrect_rvbip = (self.rect_rvbip ** 2)*10
        self.sqrect_rvshock = (self.rect_rvshock ** 2)*10
        self.sqrect_lvlead = (self.rect_lvlead ** 2)*10

    def derivatives(self):
        self.sq_rectifier()
        self.gradient_bipecg = np.gradient(self.sqrect_bipecg)
        self.gradient_ecg3 = np.gradient(self.sqrect_ecg3)
        self.gradient_ralead = np.gradient(self.sqrect_ralead)
        self.gradient_rvbip = np.gradient(self.sqrect_rvbip)
        self.gradient_rvshock = np.gradient(self.sqrect_rvshock)
        self.gradient_lvlead = np.gradient(self.sqrect_lvlead)

    def zero_crossings(self):
        self.derivatives()
        # Zero Crossings - This detects the point where the gradient changes sign
        self.zero_cross_bipecg = np.where(np.diff(np.sign(self.gradient_bipecg)))[0]
        self.zero_cross_ecg3 = np.where(np.diff(np.sign(self.gradient_ecg3)))[0]
        self.zero_cross_ralead = np.where(np.diff(np.sign(self.gradient_ralead)))[0]
        self.zero_cross_rvbip = np.where(np.diff(np.sign(self.gradient_rvbip)))[0]
        self.zero_cross_rvshock = np.where(np.diff(np.sign(self.gradient_rvshock)))[0]
        self.zero_cross_lvlead = np.where(np.diff(np.sign(self.gradient_lvlead)))[0]

    def peaks_zero_crossings(self):
        self.zero_crossings()
        self.bip_peaks = []
        self.filt_bip_peaks = []
        try:
            for crossing in self.zero_cross_bipecg:
                peak_index = np.argmax(self.rect_bipecg[crossing-10:crossing+10]) + crossing
                self.bip_peaks.append(peak_index)

                for i in range(1, len(self.bip_peaks)):
                    diff = np.gradient(self.laser_exp.bipecg.data[self.bip_peaks[i-1]:self.bip_peaks[i]])
                    if np.max(np.abs(diff)) > 0:
                        self.filt_bip_peaks.append(self.bip_peaks[i])
                    print('filt bip peak', self.filt_bip_peaks)
                else:
                    pass

            self.ecg3_peaks = []
            for crossing in self.zero_cross_ecg3:
                peak_index = np.argmax(self.rect_ecg3[crossing-10:crossing+10]) + crossing
                self.ecg3_peaks.append(peak_index)

            self.ralead_peaks = []
            for crossing in self.zero_cross_ralead:
                peak_index = np.argmax(self.rect_ralead[crossing-10:crossing+10]) + crossing
                self.ralead_peaks.append(peak_index)

            self.rvbip_peaks = []
            for crossing in self.zero_cross_rvbip:
                peak_index = np.argmax(self.rect_rvbip[crossing-10:crossing+10]) + crossing
                self.rvbip_peaks.append(peak_index)

            self.rvshock_peaks = []
            for crossing in self.zero_cross_rvshock:
                peak_index = np.argmax(self.rect_rvshock[crossing-10:crossing+10]) + crossing
                self.rvshock_peaks.append(peak_index)

            self.lvlead_peaks = []
            for crossing in self.zero_cross_lvlead:
                peak_index = np.argmax(self.rect_lvlead[crossing-10:crossing+10]) + crossing
                self.lvlead_peaks.append(peak_index)
        except:
            pass

    # Overview Regions
    def overview_region_changed(self):
        self.overview_region.setZValue(10)
        minX, maxX = self.overview_region.getRegion()

        self.rvbip_pw.plotItem.setXRange(minX, maxX, padding=0)
        self.ralead_pw.plotItem.setXRange(minX, maxX, padding=0)
        self.rvshock_pw.plotItem.setXRange(minX, maxX, padding=0)
        self.lvlead_pw.plotItem.setXRange(minX, maxX, padding=0)
        self.bipecg_pw.plotItem.setXRange(minX, maxX, padding=0)
        self.ecg3_pw.plotItem.setXRange(minX, maxX, padding=0)
        self.pressure_pw.plotItem.setXRange(minX, maxX, padding=0)
        self.laser1_pw.plotItem.setXRange(minX, maxX, padding=0)
        self.laser2_pw.plotItem.setXRange(minX, maxX, padding=0)

    def overview_region_update(self, window, viewRange):
        rgn = viewRange[0]
        self.overview_region.setRegion(rgn)

    # Toggle Changes
    def laser1_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.laser1_pw.show()

        else:
            self.laser1_pw.hide()

    def laser2_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.laser2_pw.show()
        else:
            self.laser2_pw.hide()

    def ecg3_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.ecg3_pw.show()
        else:
            self.ecg3_pw.hide()

    def bipecg_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.bipecg_pw.show()
        else:
            self.bipecg_pw.hide()

    def bp_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.pressure_pw.show()
        else:
            self.pressure_pw.hide()

    def rvshock_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.rvshock_pw.show()
        else:
            self.rvshock_pw.hide()


    def rvbip_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.rvbip_pw.show()
        else:
            self.rvbip_pw.hide()


    def lvlead_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.lvlead_pw.show()
        else:
            self.lvlead_pw.hide()

    def ralead_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.ralead_pw.show()
        else:
            self.ralead_pw.hide()

    # Calc Region Changes
    def calc_region_changed(self):
        roi_begin, roi_end = self.calc_region.getRegion()
        self.begin_lbl.setText("Begin: " + str(int(roi_begin)))
        self.end_lbl.setText("End: " + str(int(roi_end)))

    def calc_region_move(self):
        x, y = self.overview_region.getRegion()
        print(x, y)

        rgn = (x + (y - x) / 4, y - (y - x) / 4)
        self.calc_region.setRegion(rgn)

    def calc_10s(self):
        roi_begin, roi_end = self.calc_region.getRegion()
        roi_end = roi_begin + 10000
        self.calc_region.setRegion((roi_begin, roi_end))

    def calc_hrv(self):
        ecg_hint = self.laser_exp.hints['Period']
        print(ecg_hint)

        self.laser_exp.rvbip.calc_ecg_peaks(ecg_hint=ecg_hint)
        self.hrv_xs = self.laser_exp.rvbip.peaks_sample[1:]
        self.hrv_ys = 60000 / np.diff(self.laser_exp.rvbip.peaks_sample)
        self.ecg_peaks_x = self.laser_exp.rvbip.peaks_sample
        self.ecg_peaks_y = self.laser_exp.rvbip.peaks_value

    def calc_results(self):
        roi_begin, roi_end = self.calc_region.getRegion()
        self.laser_exp.begin = int(roi_begin)
        self.laser_exp.end = int(roi_end)

        try:
            self.laser_exp.process()
            print(self.laser_exp.results)
        except Exception as e:
            print("Problem in calculation")
            print(e)
        else:

            self.laser1_value.setText(
                "Laser1: " + str(round(self.laser_exp.results['Laser1_Magic'], 4)) + "(Laser1 Conf: " + str(
                    round(self.laser_exp.results['Laser1_Conf'], 4)) + ")")
            self.laser2_value.setText(
                "Laser2: " + str(round(self.laser_exp.results['Laser2_Magic'], 4)) + "(Laser2 Conf: " + str(
                    round(self.laser_exp.results['Laser2_Conf'], 4)) + ")")

            self.rr_value.setText(
                "RR (RV Bipolar Lead): " + str(int(np.mean(np.diff(self.laser_exp.rvbip.peaks_sample)))))
            self.hr_value.setText("HR: " + str(int(60000 / np.mean(np.diff(self.laser_exp.rvbip.peaks_sample)))))

            try:
                self.sbp_value.setText("SBP: " + str(self.laser_exp.results['SBP_Mean']))
                self.map_value.setText("MAP: " + str(self.laser_exp.results['MAP_Mean']))

            except Exception as e:
                print(e)

            dual_laser_window = DualLaserWindow(
                laser1_array_ys=100 * (np.exp(self.laser_exp.laser1_magic_data_all) - 1),
                laser1_sum=100 * (np.exp(self.laser_exp.laser1_magic_data) - 1),
                laser2_array_ys=100 * (np.exp(self.laser_exp.laser2_magic_data_all) - 1),
                laser2_sum=100 * (np.exp(self.laser_exp.laser2_magic_data) - 1),
                parent=self)
            dual_laser_window.show()

    # Time Stamp Period Changes
    # def time_stamp_lbl_changed(self):
    #     self.timestamp_lbl_text = self.timestamp_lbl.text()
    #     print(self.timestamp_lbl_text)
    #
    #     if self.timestamp_lbl_text == "" or self.timestamp_lbl_text == None:
    #         print("No label for time stamp has been selected")
    #     else:
    #         print("Label for time stamp has been selected: " + str(self.timestamp_lbl.text()))

    # def csv_input_changed(self):
    #
    #     if self.csv_input.text == "" or self.csv_input.text == None:
    #         print("No csv file has been selected")
    #         self.csv_file == None
    #     else:
    #         print("CSV file has been selected: " + str(self.csv_input.text))
    #         csv_f = pd.read_csv(str(self.csv_input.text))
    #         num = len(csv_f)
    #         self.csv_file = str(self.csv_input.text)
    #         Exp = 'Exp' + str(num + 1)
    #         file_lbl = str(self.laser_exp.zip_fl).split("Haem/")[-1]

    # def time_stamp_period(self):
    #     roi_begin, roi_end = self.calc_region.getRegion()
    #     print(self.csv_input)
    #
    #     header = ['Patient', 'File', 'Experiment', 'Period', 'Begin', 'End', 'BipECG', 'BP', 'Laser1', 'Laser2', 'ECG3',
    #               'RVshock', 'Rvbip', 'LVlead', 'RAlead']
    #     file_lbl = str(self.laser_exp.zip_fl).split("Haem/")[-1]
    #     if self.csv_file != None:
    #
    #         csv_f = pd.read_csv(str(self.csv_file))
    #         num = len(csv_f)
    #
    #         Exp = 'Exp' + str(num + 1)
    #
    #         data = {'Patient': str(self.laser_exp.patient), 'File': str(file_lbl),
    #                 'Experiment': str(Exp), 'Period': str(self.time_stamp_label),
    #                 'Begin': str(int(roi_begin)), 'End': str(int(roi_end)), 'BipECG': str(self.laser_exp.bipecg_source),
    #                 'BP': str(self.laser_exp.pressure_source), 'Laser1': str(self.laser_exp.laser1_source),
    #                 'Laser2': str(self.laser_exp.laser2_source),
    #                 'ECG3': str(self.laser_exp.ecg3_source), 'RVshock': str(self.laser_exp.rvshock_source),
    #                 'Rvbip': str(self.laser_exp.rvbip_source),
    #                 'LVlead': str(self.laser_exp.lvlead_source), 'RAlead': str(self.laser_exp.ralead_source)}
    #
    #         with open(str(self.csv_file), 'a', encoding='UTF8', newline='') as f:
    #             writer = csv.DictWriter(f, fieldnames=header, lineterminator='\n')
    #             # Append the data
    #             writer.writerow(data)
    #
    #             f.close()
    #     else:
    #         print("No csv file has been selected./nCreating new CSV")
    #         with open('time_period.csv', 'a', encoding='UTF8', newline='') as f:
    #
    #             writer = csv.DictWriter(f, fieldnames=header, lineterminator='\n')
    #
    #             # write the header
    #             writer.writeheader()
    #
    #             Exp = 'Exp1'
    #
    #             # write the data
    #             data = {'Patient': str(self.laser_exp.patient), 'File': str(file_lbl),
    #                     'Experiment': Exp, 'Period': str(self.timestamp_lbl_text),
    #                     'Begin': str(int(roi_begin)), 'End': str(int(roi_end)),
    #                     'BipECG': str(self.laser_exp.bipecg_source),
    #                     'BP': str(self.laser_exp.pressure_source), 'Laser1': str(self.laser_exp.laser1_source),
    #                     'Laser2': str(self.laser_exp.laser2_source),
    #                     'ECG3': str(self.laser_exp.ecg3_source), 'RVshock': str(self.laser_exp.rvshock_source),
    #                     'Rvbip': str(self.laser_exp.rvbip_source),
    #                     'LVlead': str(self.laser_exp.lvlead_source), 'RAlead': str(self.laser_exp.ralead_source)}
    #
    #             writer.writerow(data)
    #             f.close()

    def plot_single_rv_peaks(self, data):

        x_range_start = self.max_x - 100
        print("x range start", x_range_start)
        x_range_end = max_x + 100
        print("x range end", x_range_end)
        x_range_plot = range(x_range_start, x_range_end)
        single_beat_plot = data[x_range_start:x_range_end]

        self.mode_info = mode(single_beat_plot)
        print("mode: ", self.mode_info)
        self.upper_limit_mode = self.mode_info + 0.1
        self.lower_limit_mode = self.mode_info - 0.1
        loc1 = np.argmax(single_beat_plot > float(self.upper_limit_mode))
        loc2 = np.argmax(single_beat_plot < float(self.lower_limit_mode))
        print("loc1: ", loc1)
        print("loc2: ", loc2)
        if loc1 > loc2:
            self.qrs_onset = loc2 + x_range_start
            print("qrs onset: ", self.qrs_onset)

        else:
            self.qrs_onset = loc1 + x_range_start
            print("qrs onset: ", self.qrs_onset)

        loc_qrs_onset = np.where(x_range_plot == int(self.qrs_onset))
        print("loc qrs onset: ", loc_qrs_onset)
        self.min_mv = single_beat_plot[loc_qrs_onset]
        self.delta_time = int(self.max_x - self.qrs_onset)
        self.delta_mv_mode = float(self.max_beat - self.mode_info)
        self.delta_mv = float(self.max_beat - self.min_mv)
        print("delta mv: ", str(self.delta_mv))
        self.slew_rate_mode = round(self.delta_mv / self.delta_time, 4)
        self.slew_rate = round(self.delta_mv / self.delta_time, 4)
        print("slew rate: ", self.slew_rate)

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
        plt.annotate("Slew Rate (Mode): " + str(slew_rate_mode) + " mV/ms", xy=(x_range_start, 0.4 * max_beat))
        plt.annotate("RR Interval: " + str(rr_interval) + " ms", xy=(x_range_start, 0.3 * max_beat),
                     xytext=(x_range_start, 0.4 * max_beat))
        plt.show()

class DualLaserWindow(QtWidgets.QDialog):
    def __init__(self, laser1_array_ys, laser1_sum, laser2_array_ys, laser2_sum, parent=None):
        super().__init__(parent=parent)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtCore.Qt.white)
        p.setColor(self.foregroundRole(), QtCore.Qt.black)

        laser_sum_pen = pg.mkPen('k', width=8)

        laser1_array_xs = np.empty_like(laser1_array_ys)
        laser1_array_xs[:] = np.arange(laser1_array_ys.shape[1])[np.newaxis, :]
        laser1_multilines = MultiLine(laser1_array_xs, laser1_array_ys, pen_args={'color': (150, 150, 150), 'width': 2})

        laser1_sum_xs = np.arange(laser1_sum.shape[0])

        laser2_array_xs = np.empty_like(laser2_array_ys)
        laser2_array_xs[:] = np.arange(laser2_array_ys.shape[1])[np.newaxis, :]
        laser2_multilines = MultiLine(laser2_array_xs, laser2_array_ys, pen_args={'color': (150, 150, 150), 'width': 2})

        laser2_sum_xs = np.arange(laser2_sum.shape[0])

        self.laser1_pw = pg.PlotWidget()
        self.laser1_pw.setBackground('w')

        self.laser2_pw = pg.PlotWidget()
        self.laser2_pw.setBackground('w')

        self.plot_layout = QtWidgets.QVBoxLayout()
        self.plot_layout.addWidget(self.laser1_pw)
        self.plot_layout.addWidget(self.laser2_pw)

        self.setLayout(self.plot_layout)

        self.laser1_pw.addItem(laser1_multilines)
        self.laser1_pw.plot(laser1_sum_xs, laser1_sum, pen=laser_sum_pen)

        self.laser2_pw.addItem(laser2_multilines)
        self.laser2_pw.plot(laser2_sum_xs, laser2_sum, pen=laser_sum_pen)


class HaemCompromise(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtCore.Qt.white)
        p.setColor(self.foregroundRole(), QtCore.Qt.black)
        self.setWindowTitle("HAEMODYNAMIC ASSESSMENT")

        self.message = QtWidgets.QLabel()
        icon_path = "/Users/amiyazawa/Library/CloudStorage/Dropbox/01 Projects/Medtronic Algorithm/Medtronic Algorithm Python Code/icons/checkbox.png"  # Replace with the path to your icon image
        self.tick_icon = QtGui.QPixmap(icon_path)  # Replace with the path to your icon image
        icon_html = f'<img src="{icon_path}" width="{self.tick_icon.width()}" height="{self.tick_icon.height()}">'
        text = "&nbsp;&nbsp;HAEMODYNAMIC<br>&nbsp;&nbsp;COMPROMISE"
        label_text = f'<span style="vertical-align:bottom;">{icon_html}</span>{text}'
        self.message.setText(label_text)
        self.message.setStyleSheet("font-weight: bold; color: black; font-size: 20px;")

        self.submit_btn = QtWidgets.QPushButton("Acknowledged")
        self.submit_btn.clicked.connect(self.close)
        # time.sleep(5)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.message)

        self.layout.addWidget(self.submit_btn)
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.setLayout(self.layout)

class ShockDelivered(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtCore.Qt.white)
        p.setColor(self.foregroundRole(), QtCore.Qt.black)
        self.setWindowTitle("SHOCK")

        self.message = QtWidgets.QLabel()
        icon_path = "/Users/amiyazawa/Library/CloudStorage/Dropbox/01 Projects/Medtronic Algorithm/Medtronic Algorithm Python Code/icons/shock_icon.png"  # Replace with the path to your icon image
        self.tick_icon = QtGui.QPixmap(icon_path)  # Replace with the path to your icon image
        icon_html = f'<img src="{icon_path}" width="{self.tick_icon.width()}" height="{self.tick_icon.height()}">'
        text = "&nbsp;&nbsp;SHOCK<br>&nbsp;&nbsp;DELIVERED"
        label_text = f'<span style="vertical-align:bottom;">{icon_html}</span>{text}'
        self.message.setText(label_text)
        self.message.setStyleSheet("font-weight: bold; color: black; font-size: 20px;")

        self.submit_btn = QtWidgets.QPushButton("Acknowledged")
        self.submit_btn.clicked.connect(self.close)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.message)

        self.layout.addWidget(self.submit_btn)
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.setLayout(self.layout)

class ATP_Delivered(QtWidgets.QDialog):
    def __init__(self, atp_timer, parent=None):
        super().__init__(parent=parent)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtCore.Qt.white)
        p.setColor(self.foregroundRole(), QtCore.Qt.black)
        self.setWindowTitle("ATP")
        self.atp_timer = atp_timer
        self.count = int(self.atp_timer)

        self.message = QtWidgets.QLabel()
        icon_path = "/Users/amiyazawa/Library/CloudStorage/Dropbox/01 Projects/Medtronic Algorithm/Medtronic Algorithm Python Code/icons/atp_icon.png"  # Replace with the path to your icon image
        self.tick_icon = QtGui.QPixmap(icon_path)  # Replace with the path to your icon image
        icon_html = f'<img src="{icon_path}" width="{self.tick_icon.width()}" height="{self.tick_icon.height()}">'
        text = "&nbsp;&nbsp;DELIVERING<br>&nbsp;&nbsp;ATP"
        label_text = f'<span style="vertical-align:bottom;">{icon_html}</span><br>{text}<br>'
        self.message.setText(label_text)
        self.message.setStyleSheet("font-weight: bold; color: black; font-size: 20px;")

        # creating label to show the seconds
        self.timer_lbl = QtWidgets.QLabel()

        # creating a timer object
        timer_view = QtCore.QTimer(self)

        # adding action to timer
        timer_view.timeout.connect(self.timer)

        # update the timer every tenth second
        timer_view.start(100)

        self.update_ui()

        self.submit_btn = QtWidgets.QPushButton("Acknowledged")
        self.submit_btn.clicked.connect(self.close)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.message)
        self.layout.addWidget(self.timer_lbl)

        self.layout.addWidget(self.submit_btn)
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.setLayout(self.layout)

    def timer(self):
        # incrementing the counter
        self.count -= 100

        # timer is completed
        if self.count <= 0:
            # setting text to the label
            self.message.setText("ATP DELIVERED")

    def update_ui(self):
        self.timer_lbl.setText(str(self.count))

class Therapy_Cancelled(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtCore.Qt.white)
        p.setColor(self.foregroundRole(), QtCore.Qt.black)
        self.setWindowTitle("THERAPY CANCELLED")

        self.message = QtWidgets.QLabel()
        icon_path = "/Users/amiyazawa/Library/CloudStorage/Dropbox/01 Projects/Medtronic Algorithm/Medtronic Algorithm Python Code/icons/cancel.png"  # Replace with the path to your icon image
        self.tick_icon = QtGui.QPixmap(icon_path)  # Replace with the path to your icon image
        icon_html = f'<img src="{icon_path}" width="{self.tick_icon.width()}" height="{self.tick_icon.height()}">'
        text = "&nbsp;&nbsp;THERAPY<br>&nbsp;&nbsp;&nbsp;CANCELLED"
        label_text = f'<span style="vertical-align:bottom;">{icon_html}</span>{text}'
        self.message.setText(label_text)
        self.message.setStyleSheet("font-weight: bold; color: black; font-size: 20px;")

        self.submit_btn = QtWidgets.QPushButton("Acknowledged")
        self.submit_btn.clicked.connect(self.close)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.message)

        self.layout.addWidget(self.submit_btn)
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.setLayout(self.layout)

class SelectDeviceWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtCore.Qt.white)
        p.setColor(self.foregroundRole(), QtCore.Qt.black)
        self.setWindowTitle("Select Device")

        self.message = QtWidgets.QLabel("Please Select Device Type")

        self.vvi_btn = QtWidgets.QRadioButton('VVI')
        self.vvi_btn.setChecked(False)
        self.vvi_btn.toggled.connect(lambda: self.vvi_selected(self.vvi_btn))

        self.ddd_btn = QtWidgets.QRadioButton('DDD')
        self.ddd_btn.setChecked(False)
        self.ddd_btn.toggled.connect(lambda: self.ddd_selected(self.ddd_btn))

        self.crt_btn = QtWidgets.QRadioButton('CRT')
        self.crt_btn.setChecked(False)
        self.crt_btn.toggled.connect(lambda: self.crt_selected(self.crt_btn))

        self.submit_btn = QtWidgets.QPushButton("Submit")
        self.submit_btn.clicked.connect(self.close)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.message)
        self.layout.addWidget(self.vvi_btn)
        self.layout.addWidget(self.ddd_btn)
        self.layout.addWidget(self.crt_btn)
        self.layout.addWidget(self.submit_btn)
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.setLayout(self.layout)

    def vvi_selected(self, btn):
        if btn.isChecked() == True:
            self.device_type = "VVI"
            print("VVI Selected")
            return "VVI"
        else:
            print("VVI Not Selected")

    def ddd_selected(self, btn):
        if btn.isChecked() == True:
            self.device_type = "DDD"
            print("DDD Selected")
            return "DDD"
        else:
            print("DDD Not Selected")

    def crt_selected(self, btn):
        if btn.isChecked() == True:
            self.device_type = "CRT"
            print("CRT Selected")
            return "CRT"
        else:
            print("CRT Not Selected")

    def get_device_type(self):
        output = str(self.device_type)
        return output

class SecondaryPreventionWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        p = self.palette()
        p.setColor(self.backgroundRole(), QtCore.Qt.white)
        p.setColor(self.foregroundRole(), QtCore.Qt.black)

        self.setWindowTitle("Secondary Prevention Settings")
        self.message = QtWidgets.QLabel("Please Enter the Known VT cycle length in ms")

        self.secondary_vt_cycle_length = QtWidgets.QLineEdit()
        self.secondary_vt_cycle_length.setPlaceholderText("330")
        self.secondary_vt_cycle_length.setFixedWidth(200)
        self.secondary_vt_cycle_length.setValidator(QtGui.QIntValidator(100, 600))

        self.submit_btn = QtWidgets.QPushButton("Submit")
        self.submit_btn.clicked.connect(self.close)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.message)
        self.layout.addWidget(self.secondary_vt_cycle_length)
        self.layout.addWidget(self.submit_btn)
        self.layout.setSpacing(20)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.setLayout(self.layout)

        self.output = None

    def get_secondary_vt_cycle_length(self):
        self.output = int(float(self.secondary_vt_cycle_length.text()))

class MultiLine(QtWidgets.QGraphicsPathItem):
    def __init__(self, x, y, pen_args={'color': 'k'}):
        """x and y are 2D arrays of shape (Nplots, Nsamples)"""
        connect = np.ones(x.shape, dtype=bool)
        connect[:, -1] = 0  # don't draw the segment between each trace
        self.path = pg.arrayToQPath(x.flatten(), y.flatten(), connect.flatten())
        QtWidgets.QGraphicsPathItem.__init__(self, self.path)
        self.setPen(pg.mkPen(**pen_args))

    # def shape(self):  # override because QGraphicsPathItem.shape is too expensive.
    #     return QtWidgets.QGraphicsItem.shape(self)
    #
    # def boundingRect(self):
    #     return self.path.boundingRect()

def peak_detection(data, rvst, pvsb):
    window_size = 150
    start_index = max(0, peak_detected - window_size // 2)
    end_index = min(len(data), peak_detected + window_size //2)
    centred_window = data[start_index:end_index]

def data_list(data, rv_sens_threshold, post_vs_blanking):
    # def data_list(data):
    max_index_list = []
    rr_list = []

    icd_memory = {}
    icd_memory['last_r_peak'] = 0

    rv_peaks = np.where(data > rv_sens_threshold)
    # rv_peaks = np.where(data > 0.3)
    rv_peaks = np.array(rv_peaks).flatten()
    print("rv_peaks: ", rv_peaks)

    # Rate
    for i, r_peak in enumerate(rv_peaks):

        # RR Interval
        # rr_interval = r_peak - icd_memory['last_r_peak']

        # Ignore blanking period after VS
        ignore_blanking = np.where(r_peak < (icd_memory['last_r_peak'] + post_vs_blanking))
        # ignore_blanking = np.where(r_peak < (icd_memory['last_r_peak'] + 120))
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

        # amplitude, max_x, max_beat, min_beat, single_beat = amplitude(r_peak, rv_d)
        if len(data) >= 300:
            # Find rough r-peak time range
            start = r_peak - 150
            print("start: ", start)
            end = r_peak + 150
            print("end: ", end)
            x_range = range(start, end)

            # Rough QRS Dataset
            single_beat = data[start:end]
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
            max_index = single_beat.argmax() + start
            print("Maximum Index position: ", max_index)
            rrinterval = max_index - icd_memory['last_r_peak']
            icd_memory['last_r_peak'] = r_peak

            rr_list.append(rrinterval)
            max_index_list.append(max_index)
        # print(max_x_list)

    return max_index_list, rr_list

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
    #
    # if len(data) >= 300:
    #     # Find rough r-peak time range
    #     start = max(0, rpeak - 150)
    #     print("start: ", start)
    #     end = min(rpeak + 150, len(data))
    #     print("end: ", end)
    #     x_range = range(start, end)
    # first_threshold_peak = np.where(data>rv_sens_threshold)[0][0]
    # time_d.append(first_threshold_peak)
    #
    # for i, peak in enumerate(data):
    #     if peak > rv_sens_threshold:
    #         print(i, peak)
    #         time_d.append(i)
    #         next_time = i + post_vs_blanking
    #         if next_time is True and data[next_time] < low_sens_limit:
    #             print(low_sens_limit)
    #
    #
    # adj_sens_peak = last_rpeak_value * (e**(decay_rate*(i/time)))


    icd_memory['single_beat'] = deque(maxlen=1)

    # amplitude, max_x, max_beat, min_beat, single_beat = amplitude(r_peak, rv_d)
    if len(data) >= 150:
        # Find rough r-peak time range
        start = max(0, r_peak - 150)
        print("start: ", start)
        end = min(r_peak + 150, len(data))
        print("end: ", end)
        x_range = range(start, end)

        # Rough QRS Dataset
        single_beat = data[x_range]
        icd_memory['single_beat'].append(single_beat)
        # Max Voltage
        abs_data_range = abs_data[x_range]
        max_beat = np.max(abs_data_range)
        # print("max beat: ", max_beat)
        # Min Voltage
        min_beat = np.min(single_beat)
        # print("min beat: ", min_beat)

        # Amplitude
        amplitude = max_beat + abs(min_beat)
        # print("amplitude: ", amplitude)
        # Rough time of max voltage
        max_index = abs_data_range.argmax() + start
        # print("Maximum Index position: ", max_index)
        rrinterval = max_index - icd_memory['last_r_peak']

        icd_memory['last_r_peak'] = r_peak


        rr_list.append(rrinterval)
        max_index_list.append(max_index)
        amplitude_list.append(amplitude)
        # print(max_x_list)

    return max_index_list, rr_list, amplitude_list, rpeaks
