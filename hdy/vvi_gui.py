# Medtronic VVI
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
import hdy
import pywt
import matplotlib.pyplot as plt

class TimeAxisItemMdtc(pg.AxisItem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fs = 512

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

class MedtronicVVI_GUI(QtWidgets.QWidget):
    def __init__(self, laser_exp: object, parent: object = None)-> object:
        super().__init__(parent=parent)
        self.laser_exp = laser_exp
        self.trace_view = 0

        self.icd_mdt_parameters = {}

        # Initialize Parameters
        self.icd_mdt_parameters = {}
        self.icd_mdt_parameters['rvst_value'] = 0.3  # value of rv_sens_threshold
        self.icd_mdt_parameters['pvsb_value'] = 120  # value of postvsblanking
        self.icd_mdt_parameters['sensitivity'] = 0.75  # Initial sensitivity at 75% of peak EGM amplitude
        self.icd_mdt_parameters['amplitude'] = 0  # Placeholder for the amplitude of the sensed R-wave
        self.icd_mdt_parameters['onset_pct'] = 0.81  # Nominal setting
        self.icd_mdt_parameters['stability'] = 40  # Nominal Setting
        self.icd_mdt_parameters['vf_tcl'] = 320   # 188 bpm
        self.icd_mdt_parameters['fvt_tcl'] = 0  # OFF
        self.icd_mdt_parameters['vt_tcl'] = 0  # OFF
        self.icd_mdt_parameters['vf_min_nid'] = 30  # nominal
        self.icd_mdt_parameters['vf_max_nid'] = 40  # nominal
        self.icd_mdt_parameters['vt_nid'] = 0  # OFF
        self.icd_mdt_parameters['monitor_tcl'] = 0  # OFF

        self.icd_memory = {}
        self.icd_memory['r_peak'] = deque(maxlen=8)
        self.icd_memory['last_r_peak'] = 0
        self.icd_memory['rr_intervals'] = deque(maxlen=8)
        self.icd_memory['onset'] = deque(maxlen=8)
        self.icd_memory['stability'] = deque(maxlen=4)

        self.vt_counter = 0
        self.icd_memory['avg_interval_bin'] = deque(maxlen=4)
        self._fd_triggered = False
        self._tf_triggered = False
        self._fvt_vtz_triggered = False
        self._vt_rate_triggered = False
        self.icd_memory['vf_check'] = deque(maxlen=8)
        self.icd_memory['median_rr_ints'] = deque(maxlen=12)
        self._stability=False
        self.icd_memory['wavelet'] = deque(maxlen=8)  # use RV Shock lead
        self.icd_memory['haem_label'] = deque(maxlen=10)
        self.icd_memory['rhythm_label'] = deque(
            maxlen=max(self.icd_mdt_parameters['vt_nid'], self.icd_mdt_parameters['vf_max_nid']))

        self.icd_memory['active_tachy'] = False
        
        self.sensing = hdy.sensing(laser_exp)

        # self.sensing.

        self.medtronic_vvi_gui()
        self.setup_m_vvi_plots()
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def medtronic_vvi_gui(self) -> object:
        pg.setConfigOptions(antialias=True, background='w')

        # Creating Layouts
        self.m_vvi_btn_layout = QtWidgets.QVBoxLayout()
        self.m_vvi_program_settings_layout = QtWidgets.QVBoxLayout()
        self.m_vvi_comb_layout = QtWidgets.QVBoxLayout()
        self.m_vvi_leads_layout = QtWidgets.QVBoxLayout()
        self.m_vvi_settings_layout = QtWidgets.QVBoxLayout()
        self.m_vvi_main_layout = QtWidgets.QHBoxLayout()
        self.m_vvi_settings_plots_layout = QtWidgets.QVBoxLayout()

        # Creating Viewing Option Toggles to Button Layout
        self.start_btn_vvi = QtWidgets.QPushButton("")
        self.start_btn_vvi.setIcon(
            QtGui.QIcon('/Users/amiyazawa/Library/CloudStorage/Dropbox/01 Projects/Medtronic Algorithm/Medtronic Algorithm Python Code/icons/play02.png'))
        self.start_btn_vvi.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.start_btn_vvi.clicked.connect(self.start_btn_clicked_vvi)

        self.pause_btn_vvi = QtWidgets.QPushButton("")
        self.pause_btn_vvi.setIcon(
            QtGui.QIcon('/Users/amiyazawa/Library/CloudStorage/Dropbox/01 Projects/Medtronic Algorithm/Medtronic Algorithm Python Code/icons/pause02.png'))

        self.pause_btn_vvi.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.pause_btn_vvi.clicked.connect(self.pause_btn_clicked_vvi)

        self.restart_btn_vvi = QtWidgets.QPushButton("")
        self.restart_btn_vvi.setIcon(QtGui.QIcon(
            '/Users/amiyazawa/Library/CloudStorage/Dropbox/01 Projects/Medtronic Algorithm/Medtronic Algorithm Python Code/icons/restart02.png'))
        self.restart_btn_vvi.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.restart_btn_vvi.clicked.connect(self.restart_btn_clicked_vvi)

        self.rvbip_toggle_vvi = QtWidgets.QCheckBox("View RV Bipolar")
        self.rvbip_toggle_vvi.setChecked(True)
        self.rvbip_toggle_vvi.stateChanged.connect(self.rvbip_toggle_changed)

        self.rvbip_threshold_line = QtWidgets.QCheckBox("View RV Sensing Threshold")
        self.rvbip_threshold_line.setChecked(False)
        self.rvbip_threshold_line.stateChanged.connect(self.rvbip_threshold_line_toggle_changed)

        self.rvshock_toggle_vvi = QtWidgets.QCheckBox("View RV Shock")
        self.rvshock_toggle_vvi.setChecked(True)
        self.rvshock_toggle_vvi.stateChanged.connect(self.rvshock_toggle_changed)

        self.marker_toggle_vvi = QtWidgets.QCheckBox("View Marker Line")
        self.marker_toggle_vvi.setChecked(True)
        self.marker_toggle_vvi.stateChanged.connect(self.marker_toggle_changed)

        self.ecgmarker_toggle_vvi = QtWidgets.QCheckBox("View Bipolar ECG Marker Line")
        self.ecgmarker_toggle_vvi.setChecked(True)
        self.ecgmarker_toggle_vvi.stateChanged.connect(self.ecgmarker_toggle_changed)

        self.ecg_toggle_vvi = QtWidgets.QCheckBox("View Bipolar ECG")
        self.ecg_toggle_vvi.setChecked(True)
        self.ecg_toggle_vvi.stateChanged.connect(self.ecg_toggle_changed)

        self.laser1_toggle_vvi = QtWidgets.QCheckBox("View Laser1")
        self.laser1_toggle_vvi.setChecked(True)
        self.laser1_toggle_vvi.stateChanged.connect(self.laser1_toggle_changed)

        self.laser2_toggle_vvi = QtWidgets.QCheckBox("View Laser2")
        self.laser2_toggle_vvi.setChecked(False)
        self.laser2_toggle_vvi.stateChanged.connect(self.laser2_toggle_changed)

        self.overview_toggle_vvi = QtWidgets.QCheckBox("View Overview")
        self.overview_toggle_vvi.setChecked(True)
        self.overview_toggle_vvi.stateChanged.connect(self.overview_toggle_changed)

        # CREATING LABELS
        self.begin_lbl_vvi = QtWidgets.QLabel()
        self.end_lbl_vvi = QtWidgets.QLabel()
        self.laser1_value_vvi = QtWidgets.QLabel()
        self.laser2_value_vvi = QtWidgets.QLabel()
        self.spacer_lbl = QtWidgets.QLabel('\n')
        self.hr_value_vvi = QtWidgets.QLabel()
        self.rr_value_vvi = QtWidgets.QLabel()
        self.sbp_value_vvi = QtWidgets.QLabel()
        self.map_value_vvi = QtWidgets.QLabel()
        self.zipfl_lbl_vvi = QtWidgets.QLabel()
        self.m_settings_lbl_vvi = QtWidgets.QLabel()
        self.viewing_lbl_vvi = QtWidgets.QLabel()
        self.indication_lbl_vvi = QtWidgets.QLabel()
        self.onset_lbl_vvi = QtWidgets.QLabel()
        self.stability_lbl_vvi = QtWidgets.QLabel()
        self.wavelet_lbl_vvi = QtWidgets.QLabel()
        self.rv_sens_threshold_lbl_vvi = QtWidgets.QLabel()
        self.postvsblanking_lbl_vvi = QtWidgets.QLabel()
        self.pacing_dependent_lbl_vvi = QtWidgets.QLabel()
        self.fvt_tcl_program_lbl_vvi = QtWidgets.QLabel()
        self.test_lbl = QtWidgets.QLabel()
        self.media_control_lbl = QtWidgets.QLabel()

        # LABELS
        zip_fl_string = self.laser_exp.zip_fl
        zipfl_lbl_text = zip_fl_string.split("/")[-1]
        self.zipfl_lbl_vvi.setText(str(zipfl_lbl_text))
        self.zipfl_lbl_vvi.setAlignment(QtCore.Qt.AlignRight)

        self.m_settings_lbl_vvi.setText("Medtronic VVI Settings")
        self.m_settings_lbl_vvi.setStyleSheet("font-weight: bold")
        self.viewing_lbl_vvi.setText("EGM Views")
        self.viewing_lbl_vvi.setStyleSheet("font-weight: bold")
        self.indication_lbl_vvi.setText("ICD Indication")
        self.wavelet_lbl_vvi.setText("Wavelet")
        self.rv_sens_threshold_lbl_vvi.setText("RV Sensing Threshold")
        self.postvsblanking_lbl_vvi.setText("Post Ventricular (VS) Blanking")
        self.pacing_dependent_lbl_vvi.setText("Pacing Dependent")
        self.test_lbl.setText("Test")
        self.test_lbl.setStyleSheet("font-weight: bold")
        self.media_control_lbl.setText("\nMedia Controls")
        self.media_control_lbl.setStyleSheet("font-weight: bold")

        # Adding Toggles to Button Layout
        self.m_vvi_btn_layout.addWidget(self.viewing_lbl_vvi)
        self.m_vvi_btn_layout.addWidget(self.rvbip_toggle_vvi)
        self.m_vvi_btn_layout.addWidget(self.rvbip_threshold_line)
        self.m_vvi_btn_layout.addWidget(self.rvshock_toggle_vvi)
        self.m_vvi_btn_layout.addWidget(self.marker_toggle_vvi)
        self.m_vvi_btn_layout.addWidget(self.ecgmarker_toggle_vvi)
        self.m_vvi_btn_layout.addWidget(self.ecg_toggle_vvi)
        self.m_vvi_btn_layout.addWidget(self.laser1_toggle_vvi)
        self.m_vvi_btn_layout.addWidget(self.laser2_toggle_vvi)
        self.m_vvi_btn_layout.addWidget(self.overview_toggle_vvi)

        # Creating ICD Settings Widgets
        self.indication = QtWidgets.QComboBox()
        self.indication.addItems("Primary Secondary AMTest".split())
        self.indication.setCurrentIndex(0)
        self.indication.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        self.indication.currentIndexChanged.connect(self.indication_changed)

        self.rvst1 = QtWidgets.QRadioButton("0.15")
        self.rvst2 = QtWidgets.QRadioButton("0.3")
        self.rvst2.setChecked(True)
        self.rvst3 = QtWidgets.QRadioButton("0.45")
        self.rvst4 = QtWidgets.QRadioButton("0.6")
        self.rvst5 = QtWidgets.QRadioButton("0.9")
        self.rvst6 = QtWidgets.QRadioButton("1.2")
        self.rvst7 = QtWidgets.QRadioButton("1.5")
        self.rvst8 = QtWidgets.QRadioButton("1.8")

        self.rvst_group = QtWidgets.QButtonGroup()
        self.rvst_group.addButton(self.rvst1, 1)
        self.rvst_group.addButton(self.rvst2, 2)
        self.rvst_group.addButton(self.rvst3, 3)
        self.rvst_group.addButton(self.rvst4, 4)
        self.rvst_group.addButton(self.rvst5, 5)
        self.rvst_group.addButton(self.rvst6, 6)
        self.rvst_group.addButton(self.rvst7, 7)
        self.rvst_group.addButton(self.rvst8, 8)
        self.rvst_group.buttonClicked.connect(self.rv_sens_threshold_changed)

        self.pvsb1 = QtWidgets.QRadioButton("80")
        self.pvsb2 = QtWidgets.QRadioButton("100")
        self.pvsb3 = QtWidgets.QRadioButton("120")
        self.pvsb3.setChecked(True)
        self.pvsb4 = QtWidgets.QRadioButton("140")

        self.pvsb_group = QtWidgets.QButtonGroup()
        self.pvsb_group.addButton(self.pvsb1, 80)
        self.pvsb_group.addButton(self.pvsb2, 100)
        self.pvsb_group.addButton(self.pvsb3, 120)
        self.pvsb_group.addButton(self.pvsb4, 140)
        self.pvsb_group.buttonClicked.connect(self.postvsblanking_changed)

        self.pacing_dependent = QtWidgets.QComboBox()
        self.pacing_dependent.addItems("True False".split())
        self.pacing_dependent.setCurrentIndex(1)
        self.pacing_dependent.setMaximumWidth(120)
        self.pacing_dependent.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        self.pacing_dependent.currentIndexChanged.connect(self.pacing_dep_changed)

        # MEDTRONIC VVI SETTINGS AND Labels
        self.m_vvi_settings_layout.addWidget(self.m_settings_lbl_vvi)
        self.m_vvi_settings_layout.addWidget(self.indication_lbl_vvi)
        self.m_vvi_settings_layout.addWidget(self.indication)
        self.m_vvi_settings_layout.addWidget(self.pacing_dependent_lbl_vvi)
        self.m_vvi_settings_layout.addWidget(self.pacing_dependent)

        # RESULTS LABELS LAYOUT
        self.media_control_hbox = QtWidgets.QHBoxLayout()

        self.m_vvi_btn_layout.addWidget(self.media_control_lbl)
        self.media_control_hbox.addWidget(self.start_btn_vvi)
        self.media_control_hbox.addWidget(self.pause_btn_vvi)
        self.media_control_hbox.addWidget(self.restart_btn_vvi)
        self.media_control_hbox.addWidget(self.spacer_lbl)
        self.m_vvi_btn_layout.addLayout(self.media_control_hbox)
        # self.m_vvi_btn_layout.addWidget(self.laser1_value_vvi)
        # self.m_vvi_btn_layout.addWidget(self.laser2_value_vvi)
        # self.m_vvi_btn_layout.addWidget(self.rr_value_vvi)
        # self.m_vvi_btn_layout.addWidget(self.hr_value_vvi)



        # PLOT LAYOUTS
        self.rvbip_vvi_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItemMdtc(orientation='bottom')})
        self.rvbip_vvi_pi = self.rvbip_vvi_pw.getPlotItem()
        self.rvbip_vvi_pi.setLabel(axis='left', text="RV Bip")
        self.rvbip_vvi_plt = self.rvbip_vvi_pi.plot()
        self.rvbip_peak_vvi_plt = self.rvbip_vvi_pi.plot()

        self.rvshock_vvi_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItemMdtc(orientation='bottom')})
        self.rvshock_vvi_pi = self.rvshock_vvi_pw.getPlotItem()
        self.rvshock_vvi_pi.setLabel(axis='left', text="RV Shock")
        self.rvshock_vvi_plt = self.rvshock_vvi_pi.plot()
        self.rvshock_peak_vvi_plt = self.rvshock_vvi_pi.plot()

        self.markerecg_vvi_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItemMdtc(orientation='bottom')})
        self.markerecg_vvi_pi = self.markerecg_vvi_pw.getPlotItem()
        self.markerecg_vvi_pi.setLabel(axis='left', text="Marker Bipolar ECG")
        self.markerecg_vvi_plt = self.markerecg_vvi_pi.plot()
        self.markerecg_peak_vvi_plt = self.markerecg_vvi_pi.plot()

        self.ecg_vvi_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItemMdtc(orientation='bottom')})
        self.ecg_vvi_pi = self.ecg_vvi_pw.getPlotItem()
        self.ecg_vvi_pi.setLabel(axis='left', text="Bipolar ECG")
        self.ecg_vvi_plt = self.ecg_vvi_pi.plot()
        self.ecg_peak_vvi_plt = self.ecg_vvi_pi.plot()

        self.laser1_vvi_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItemMdtc(orientation='bottom')})
        self.laser1_vvi_pi = self.laser1_vvi_pw.getPlotItem()
        self.laser1_vvi_pi.setLabel(axis='left', text="LDPM1")
        self.laser1_vvi_plt = self.laser1_vvi_pi.plot()
        self.envelope1_vvi_plt = self.laser1_vvi_pi.plot()

        self.laser2_vvi_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItemMdtc(orientation='bottom')})
        self.laser2_vvi_pi = self.laser2_vvi_pw.getPlotItem()
        self.laser2_vvi_pi.setLabel(axis='left', text="LDPM2")
        self.laser2_vvi_plt = self.laser2_vvi_pi.plot()
        self.laser2_vvi_pw.hide()
        self.envelope2_vvi_plt = self.laser2_vvi_pi.plot()

        self.marker_vvi_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItemMdtc(orientation='bottom')})
        self.marker_vvi_pi = self.marker_vvi_pw.getPlotItem()
        self.marker_vvi_pi.setLabel(axis='left', text="Marker")
        self.marker_vvi_plt = self.marker_vvi_pi.plot()
        self.envelope2_vvi_plt = self.marker_vvi_pi.plot()

        self.overview_vvi_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItemMdtc(orientation='bottom')})
        self.overview_vvi_pi = self.overview_vvi_pw.getPlotItem()
        self.overview_vvi_pi.setLabel(axis='left', text="Overview")
        self.overview_vvi_plt = self.overview_vvi_pi.plot()

        # Sharing X axis in plots
        self.rvbip_vvi_pw.setXLink(self.rvshock_vvi_pw)
        self.rvbip_vvi_pw.setXLink(self.marker_vvi_pw)
        self.rvbip_vvi_pw.setXLink(self.ecg_vvi_pw)
        self.rvbip_vvi_pw.setXLink(self.laser1_vvi_pw)
        self.rvbip_vvi_pw.setXLink(self.laser2_vvi_pw)
        self.rvbip_vvi_pw.setXLink(self.markerecg_vvi_pw)

        self.rvshock_vvi_pw.setXLink(self.rvbip_vvi_pw)
        self.rvshock_vvi_pw.setXLink(self.marker_vvi_pw)
        self.rvshock_vvi_pw.setXLink(self.ecg_vvi_pw)
        self.rvshock_vvi_pw.setXLink(self.laser1_vvi_pw)
        self.rvshock_vvi_pw.setXLink(self.laser2_vvi_pw)
        self.rvshock_vvi_pw.setXLink(self.markerecg_vvi_pw)

        self.marker_vvi_pw.setXLink(self.rvbip_vvi_pw)
        self.marker_vvi_pw.setXLink(self.rvshock_vvi_pw)
        self.marker_vvi_pw.setXLink(self.ecg_vvi_pw)
        self.marker_vvi_pw.setXLink(self.laser1_vvi_pw)
        self.marker_vvi_pw.setXLink(self.laser2_vvi_pw)
        self.marker_vvi_pw.setXLink(self.markerecg_vvi_pw)

        self.markerecg_vvi_pw.setXLink(self.rvbip_vvi_pw)
        self.markerecg_vvi_pw.setXLink(self.rvshock_vvi_pw)
        self.markerecg_vvi_pw.setXLink(self.marker_vvi_pw)
        self.markerecg_vvi_pw.setXLink(self.laser1_vvi_pw)
        self.markerecg_vvi_pw.setXLink(self.laser2_vvi_pw)
        self.markerecg_vvi_pw.setXLink(self.ecg_vvi_pw)

        self.ecg_vvi_pw.setXLink(self.rvbip_vvi_pw)
        self.ecg_vvi_pw.setXLink(self.rvshock_vvi_pw)
        self.ecg_vvi_pw.setXLink(self.marker_vvi_pw)
        self.ecg_vvi_pw.setXLink(self.laser1_vvi_pw)
        self.ecg_vvi_pw.setXLink(self.laser2_vvi_pw)
        self.ecg_vvi_pw.setXLink(self.markerecg_vvi_pw)

        self.laser1_vvi_pw.setXLink(self.rvbip_vvi_pw)
        self.laser1_vvi_pw.setXLink(self.rvshock_vvi_pw)
        self.laser1_vvi_pw.setXLink(self.marker_vvi_pw)
        self.laser1_vvi_pw.setXLink(self.ecg_vvi_pw)
        self.laser1_vvi_pw.setXLink(self.laser2_vvi_pw)
        self.laser1_vvi_pw.setXLink(self.markerecg_vvi_pw)

        self.laser2_vvi_pw.setXLink(self.rvbip_vvi_pw)
        self.laser2_vvi_pw.setXLink(self.rvshock_vvi_pw)
        self.laser2_vvi_pw.setXLink(self.marker_vvi_pw)
        self.laser2_vvi_pw.setXLink(self.ecg_vvi_pw)
        self.laser2_vvi_pw.setXLink(self.laser1_vvi_pw)
        self.laser2_vvi_pw.setXLink(self.markerecg_vvi_pw)

        # Adding Plots to VVI Leads Layout
        self.m_vvi_leads_layout.addWidget(self.rvbip_vvi_pw, stretch=1)
        self.m_vvi_leads_layout.addWidget(self.rvshock_vvi_pw, stretch=1)
        self.m_vvi_leads_layout.addWidget(self.marker_vvi_pw, stretch=1)
        self.m_vvi_leads_layout.addWidget(self.markerecg_vvi_pw, stretch=1)
        self.m_vvi_leads_layout.addWidget(self.ecg_vvi_pw, stretch=1)
        self.m_vvi_leads_layout.addWidget(self.laser1_vvi_pw, stretch=1)
        self.m_vvi_leads_layout.addWidget(self.laser2_vvi_pw, stretch=1)
        self.m_vvi_leads_layout.addWidget(self.overview_vvi_pw, stretch=1)
        self.m_vvi_leads_layout.addWidget(self.zipfl_lbl_vvi)
        self.m_vvi_program_settings_layout.addWidget(self.test_lbl)

        # Setting up the Timer for Updating Plots
        self.timer_m = QtCore.QTimer()
        self.timer_m.timeout.connect(self.update_medtronic_plots)

        # Adding Widgets to VVI Program Settings Layout in Group Boxes
        self.rvst_groupbox = QtWidgets.QGroupBox("RV Sensitivity Threshold")
        self.rvst_groupbox.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.rvst_vbox = QtWidgets.QVBoxLayout()
        self.rvst_vbox.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.rvst_groupbox.setLayout(self.rvst_vbox)
        self.rvst_hbox1 = QtWidgets.QHBoxLayout()
        self.rvst_hbox1.addWidget(self.rvst1)
        self.rvst_hbox1.addWidget(self.rvst2)
        self.rvst_hbox1.addWidget(self.rvst3)
        self.rvst_hbox1.addWidget(self.rvst4)
        self.rvst_hbox2 = QtWidgets.QHBoxLayout()
        self.rvst_hbox2.addWidget(self.rvst5)
        self.rvst_hbox2.addWidget(self.rvst6)
        self.rvst_hbox2.addWidget(self.rvst7)
        self.rvst_hbox2.addWidget(self.rvst8)
        self.rvst_vbox.addLayout(self.rvst_hbox1)
        self.rvst_vbox.addLayout(self.rvst_hbox2)
        self.m_vvi_settings_layout.addWidget(self.rvst_groupbox)

        self.pvsb_groupbox = QtWidgets.QGroupBox("Post VS Blanking")
        self.pvsb_groupbox.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)

        self.m_vvi_settings_layout.addWidget(self.pvsb_groupbox)
        self.pvsb_hbox = QtWidgets.QHBoxLayout(self.pvsb_groupbox)
        self.pvsb_hbox.addWidget(self.pvsb1)
        self.pvsb_hbox.addWidget(self.pvsb2)
        self.pvsb_hbox.addWidget(self.pvsb3)
        self.pvsb_hbox.addWidget(self.pvsb4)
        self.pvsb_hbox.addWidget(self.spacer_lbl)

        # Aligning Layouts to the Top Left
        self.m_vvi_btn_layout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.m_vvi_settings_layout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.m_vvi_comb_layout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.m_vvi_leads_layout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.m_vvi_main_layout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)

        # Setting Spacing of Layouts
        self.m_vvi_comb_layout.setSpacing(0)
        self.m_vvi_comb_layout.setContentsMargins(0, 0, 0, 0)
        self.m_vvi_leads_layout.setSpacing(0)
        self.m_vvi_leads_layout.setContentsMargins(10, 0, 10, 250)
        self.m_vvi_btn_layout.setSpacing(10)
        self.m_vvi_btn_layout.setContentsMargins(0, 0, 0, 0)
        self.m_vvi_settings_layout.setSpacing(0)
        self.m_vvi_settings_layout.setContentsMargins(0, 20, 5, 20)
        self.m_vvi_main_layout.setSpacing(0)
        self.m_vvi_main_layout.setContentsMargins(0, 0, 0, 0)

        self.v_detection_gui()

        # Setting up the Final Layout
        self.m_vvi_settings_layout.addLayout(self.other_enhance_grid)
        self.m_vvi_comb_layout.addLayout(self.m_vvi_btn_layout)
        self.m_vvi_comb_layout.addLayout(self.m_vvi_settings_layout)
        self.m_vvi_main_layout.addLayout(self.m_vvi_comb_layout)
        self.m_vvi_leads_layout.addLayout(self.v_detection_layout)
        self.m_vvi_main_layout.addLayout(self.m_vvi_leads_layout)

        # self.max_x_list = self.sensing.peak_adaptive_threshold(self.sensing.sqrect_rvbip, window_size = 10, threshold = self.icd_mdt_parameters['rvst_value'], pvsb=self.icd_mdt_parameters['pvsb_value'], factor=0.6)
        # self.max_x_list = self.sensing.filt_rvbip_peaks

        # self.max_x_list, self.rr_list, self.amplitude_list, self.rpeaks = data_list(self.sensing.rvbip_data,self.icd_mdt_parameters['rvst_value'],self.icd_mdt_parameters['pvsb_value'])
        # self.max_x_list, self.rr_list, self.amplitude_list, self.rpeaks = data_list(self.laser_exp.rvbip.data,
        #                                                                self.icd_mdt_parameters['rvst_value'],
        #                                                                self.icd_mdt_parameters['pvsb_value'])
        # self.max_x_list, self.rr_list = data_list(self.laser_exp.rvbip.data,
        #                                                                self.icd_mdt_parameters['rvst_value'],
        #                                                                self.icd_mdt_parameters['pvsb_value'])
        # Setting up the Final Layout
        self.setLayout(self.m_vvi_main_layout)

    def setup_m_vvi_plots(self):
        self.rvbip_vvi_pi.getAxis('left').setWidth(w=40)
        self.rvbip_vvi_pi.getAxis('left').setStyle(showValues=False)
        self.marker_vvi_pi.getAxis('left').setWidth(w=40)
        self.marker_vvi_pi.getAxis('left').setStyle(showValues=False)
        self.rvshock_vvi_pi.getAxis('left').setWidth(w=40)
        self.rvshock_vvi_pi.getAxis('left').setStyle(showValues=False)
        self.markerecg_vvi_pi.getAxis('left').setWidth(w=40)
        self.markerecg_vvi_pi.getAxis('left').setStyle(showValues=False)
        self.ecg_vvi_pi.getAxis('left').setWidth(w=40)
        self.ecg_vvi_pi.getAxis('left').setStyle(showValues=False)
        self.laser1_vvi_pi.getAxis('left').setWidth(w=40)
        self.laser2_vvi_pi.getAxis('left').setWidth(w=40)
        self.overview_vvi_pi.getAxis('left').setWidth(w=50)
        self.overview_vvi_pi.getAxis('left').setStyle(showValues=False)

    def update_medtronic_plots(self):
        # 8 beats 88% VTCL, with S1 and S2. PPI to determine
        pg.setConfigOptions(antialias=True, background='w')

        signal_mapping = {
            'RVbip': 'rvbip_resampled_norm',
            'RVshock': 'rvshock_resampled_norm',
            'ECG': 'ecg_filt',
            'ECG3': 'ecg3_filt',
            'LVlead': 'lvlead_resampled_norm',
            'RAlead': 'ralead_resampled_norm',
            'Laser1': 'laser1_resampled'
        }

        for signal, attribute in signal_mapping.items():
            if any(signal in k for k in self.sensing.used_signals):
                data = getattr(self.sensing, attribute)
                samples = data.shape[0]
                self.main_ecg_signal = signal.lower()
                break  # Exit the loop if a match is found

        ecg_hint = self.laser_exp.hints['Period']

        m_beat_end =int((500 + self.trace_view) *0.512)
        m_beat_start = int(max(0, self.trace_view - 10000)*0.512)

        self.trace_view += int(500*0.512)

        m_vvi_range = np.arange(m_beat_start, m_beat_end)

        print('m_beat_start: ', m_beat_start)
        print('m_beat_end: ', m_beat_end)
        psa_data = {}
        for k in self.sensing.used_signals.keys():
            signal = k.lower()
            if signal in ['rvbip', 'rvshock']:
                vvi_data = getattr(self.sensing, signal + '_resampled_norm')
                psa_data[signal + '_vvi_data'] = vvi_data[m_beat_start:m_beat_end]
            elif signal in ['ecg', 'ecg3']:
                if 'ECG' not in self.sensing.used_signals.keys() and signal=='ecg3':
                    vvi_data = getattr(self.sensing, 'ecg3_filt')
                    ecg_signal = signal
                    psa_data['ecg_vvi_data'] = vvi_data[m_beat_start:m_beat_end]
                elif signal=='ecg':
                    ecg_signal = signal
                    vvi_data = getattr(self.sensing, signal + '_filt')
                    psa_data[signal + '_vvi_data'] = vvi_data[m_beat_start:m_beat_end]
                else:
                    pass
            elif signal in ['laser1', 'laser2']:
                vvi_data = getattr(self.sensing, signal + '_resampled')
                psa_data[signal + '_vvi_data'] = vvi_data[m_beat_start:m_beat_end]
            else:
                pass


# RATES
        rrints = getattr(self.sensing, self.main_ecg_signal + '_rrints')
        rpeaks = getattr(self.sensing, self.main_ecg_signal + '_maxpeaks')

        for i, rpeak in enumerate(rpeaks):
            if i == 0:
                continue
            if m_beat_start < rpeak < m_beat_end:


                index = max(0, i - 1)
                rrint = rrints[index]
                self.icd_memory['rr_intervals'].append(rrint)
                self.icd_memory['rr_intervals'].append(rrint)
                self.icd_memory['onset'].append(rrint)
                self.icd_memory['median_rr_ints'].append(rrint)

                if rrint <= self.icd_mdt_parameters['vf_tcl']*0.512:
                    self.marker_label = 'FS'
                    self.icd_memory['rhythm_label'].append(self.marker_label)
                    fs_count = sum(1 for rhythm in self.icd_memory['rhythm_label'] if rhythm == 'FS')
                    if len(self.icd_memory['rhythm_label']) == self.icd_mdt_parameters['vf_max_nid'] and fs_count == self.icd_mdt_parameters['vf_min_nid']:
                        self.vf_rate_trigger()
                        print("WARNING: VF\nDETECTED")

                elif self.icd_mdt_parameters['fvt_tcl']*0.512 >= rrint >= self.icd_mdt_parameters['vf_tcl']*0.512 and (
                        self.fvt_combobox.currentText() != "OFF"):
                    self.vt_counter += 1
                    self.marker_label = 'TS\n' + str(self.vt_counter)
                    self.icd_memory['rhythm_label'].append(self.marker_label)
                    ts_vf_count = sum(1 for rhythm in self.icd_memory['rhythm_label'] if 'TS' in rhythm)
                    if len(self.icd_memory['rhythm_label']) == self.icd_mdt_parameters['vf_max_nid'] and ts_vf_count == self.icd_mdt_parameters['vf_min_nid'] and self.fvt_combobox.currentText() == "via-VF":
                        self.fvt_vfz_rate()

                        print("WARNING: FAST VT\nDETECTED in VF Zone")
                    if len(self.icd_memory['rhythm_label']) >= self.icd_mdt_parameters['vt_nid'] and self.check_consecutive_vt(self.icd_memory['rhythm_label'], self.icd_mdt_parameters['vt_nid'])==True and self.fvt_combobox.currentText() == "via-VT":
                        self.fvt_vtz_trigger()
                        print("WARNING: FAST VT\nDETECTED in VT Zone")

                elif rrint <= self.icd_mdt_parameters['vf_tcl']*0.512 > max(self.icd_mdt_parameters['fvt_tcl']*0.512,
                                                                                self.icd_mdt_parameters['vf_tcl']*0.512) and (
                        self.fvt_combobox.currentText() == "OFF"):
                    print('vt_counter', self.vt_counter)

                    self.marker_label = 'TS\n' + str(self.vt_counter)
                    self.icd_memory['rhythm_label'].append(self.marker_label)
                    if self.vt_rate_trigger() == True:
                        print("WARNING: VT\nDETECTED")

                else:
                    self.vt_counter = 0
                    self.marker_label = 'VS'
                    self.icd_memory['rhythm_label'].append(self.marker_label)

    # ONSET DATA
                if self.onset_combobox != "OFF" and ((rrint <= int(self.icd_mdt_parameters['vt_tcl']*0.512)) or (
                        rrint <= int(
                    self.icd_mdt_parameters['fvt_tcl']*0.512) and self.fvt_tcl_combobox.currentText() == 'via-VT')) and (
                        len(self.icd_memory['onset']) == 8) and int(self.vt_counter) >= 3:

                    print('vt counter: ', self.vt_counter)
                    # print("icd_memory onset: ", self.icd_memory['onset'])
                    mdtc_onset_seq = np.array(self.icd_memory['onset'])

                    print("medtronic onset sequence: ", mdtc_onset_seq)
                    # print(type(mdtc_onset_seq))
                    a = mdtc_onset_seq[:4]
                    b = mdtc_onset_seq[4:]

                    print("a: ", a)
                    print("b: ", b)
                    onset_trigger = self.mdtc_onset_trigger(a, b)

                    if (onset_trigger == True) and self.icd_memory['active_tachy'] == False:
                        print("onset trigger: ", onset_trigger)

                        print("\033[31mVT onset criteria has been met: \033[0m")
                        self.icd_memory['active_tachy'] = True
                        self.onset_met_lbl.show()
                    else:
                        pass

                    if onset_trigger == True:

                        if self.stability_combobox != "OFF" and int(self.vt_counter) == 3:
                            mdtc_stabil_seq = self.icd_memory['stability'] = np.array(
                                self.icd_memory['rr_intervals'])[4:]

                            last_CL = mdtc_stabil_seq[-1]
                            rest = mdtc_stabil_seq[0:2]

                            for i in rest:
                                stabil_eq = abs(last_CL - i)
                                print('stabil eq: ', stabil_eq)

                                if stabil_eq > self.icd_mdt_parameters['stability']:
                                    print('NOT VT')

                                elif stabil_eq < self.icd_mdt_parameters['stability']:
                                    self.stability_met_lbl.show()
                                    self._stability = True
                        else:
                            pass

                    if self._stability == True and self.vt_counter >= 15:
                        rr_int_sum = np.sum(self.icd_memory['rr_intervals'])
                        print(rr_int_sum)
                        m_begin = m_beat_end - rr_int_sum
                        print("begin: ", m_begin)

                        self.calc_results_beats(begin=m_begin, end=m_beat_end)

                        print("laser1_val: ", self.laser1_val_vt)
                        print("laser1_conf: ", self.laser1_conf_vt)
                        print("laser2_val: ", self.laser2_val_vt)
                        print("laser2_conf: ", self.laser2_conf_vt)

                        if self.laser1_conf_vt > self.laser2_conf_vt:
                            laser_val_vt = self.laser1_val_vt
                            laser_conf_vt = self.laser1_conf_vt

                        else:
                            laser_val_vt = self.laser2_val_vt
                            laser_conf_vt = self.laser2_conf_vt

                        if laser_val_vt > 5 and laser_conf_vt > 25:
                            print("Haemodynamically Stable")
                            self._haem_comp = False
                            self.haem_comp_alert_viewed = False
                            if self._haem_comp:
                                self.haem_unstable_lbl.hide()

                        if laser_val_vt <= 5:
                            print("Haemodynamically Unstable")
                            self._haem_comp = True
                            self.haem_unstable_lbl.show()

                            if self.haem_comp_alert_viewed:
                                pass
                            else:
                                self.haem_unstable_lbl.show()
                                self.haem_comp_alert = HaemCompromise()
                                self.haem_comp_alert.show()
                                self.haem_comp_alert_viewed = True

                        if self._haem_comp == False and self.vt_counter > self.icd_mdt_parameters[
                            'vt_nid'] and self.marker_label != 'VS':
                            print(self.vt_counter)
                            if self.cancel_tx_viewed == True:
                                pass
                            else:
                                self.cancel_tx = Therapy_Cancelled()
                                self.cancel_tx.show()
                                self.cancel_tx_viewed = True

                        if self._haem_comp == True and self.vt_counter > self.icd_mdt_parameters['vt_nid']:

                            if self._atp1_viewed == False:
                                print(self.median_rr)
                                median_rr = int(self.median_rr)
                                atp_ms = int(median_rr * 0.81 * 8.2)
                                print("atp duration (ms): ", atp_ms)

                                self.atp_del = ATP_Delivered(atp_ms)
                                self.atp_del.show()
                                self._atp_del1 = True
                                self._atp1_viewed = True

                        if self._haem_comp == True and self._atp_del1 == True:

                            if self._atp2_viewed == False:
                                atp_ms = int(self.median_rr * 0.81 * 8.2)
                                print("atp duration (ms): ", atp_ms)

                                self.atp_del = ATP_Delivered(atp_ms)
                                self.atp_del.show()
                                self._atp_del2 = True

                        if self._haem_comp == True and self._atp_del2 == True:
                            self.vt_shock = ShockDelivered()
                            self.vt_shock.show()
                            self.vt_shock_counter += 1


                else:
                    pass

        if len(self.icd_memory['median_rr_ints']) == 12:
            self.median_rr = list(self.icd_memory['median_rr_ints'])
            self.median_rr = np.median(np.sort(self.median_rr))
            print("Median RR: ", self.median_rr)

        self.marker_vvi_plt.setData(x=None, y=None, pen=None, symbol=None,
                                    antialise=True,
                                    autoDownsample=True, clipToView=True)
        self.marker_vvi_pw.setXRange(m_beat_start, m_beat_end, padding=0)
        self.marker_vvi_pw.addItem(pg.InfiniteLine(pos=0, angle=0, pen='k', movable=False))

        self.markerecg_vvi_plt.setData(x=None, y=None, pen=None, symbol=None,
                                    antialise=True,
                                    autoDownsample=True, clipToView=True)
        self.markerecg_vvi_pw.setXRange(m_beat_start, m_beat_end, padding=0)
        self.markerecg_vvi_pw.addItem(pg.InfiniteLine(pos=0, angle=0, pen='k', movable=False))

        ecg_peaks = self.sensing[ecg_signal + '_maxpeaks']

        for num, rpeak in enumerate(rpeaks):
            if m_beat_start < rpeak < m_beat_end:
                if not any(label in self.marker_label for label in ['TF', 'TD', 'FD']):
                    pen_col = pg.mkPen('k', width=1)
                else:
                    pen_col = pg.mkPen('r', width=1)
                marker_inf_line = pg.InfiniteLine(pos=rpeak, angle=90, movable=False,
                                                  label=self.marker_label, markers='v',
                                                  pen=pen_col, span=(0.5, 1))

                marker_inf_line.label.setPosition(-0.2)
                self.marker_vvi_pw.addItem(marker_inf_line)
                    #     self.marker_vvi_pw.addItem(pg.InfiniteLine(pos=rpeak, angle=90, movable=False, markers='v', pen='r', span=(0.5, 1)))
                #     # self.marker_vvi_pi.addItem(pg.InfiniteLine(pos=rpeak, label=self.marker_label, angle=90, span=(0.5, 1), pen=pg.mkPen('k', width=1)))
                # else:
        prev_ecgpeak = 0

        for num, rpeak in enumerate(ecg_peaks):
            if m_beat_start < rpeak < m_beat_end:

                # markerecg_inf_line.label.setPosition(-0.5)
                ecg_rrint = rpeak - prev_ecgpeak
                prev_ecgpeak = rpeak
                if ecg_rrint <= self.icd_mdt_parameters['vf_tcl'] * 0.512:
                    self.ecgmarker_label = 'FS'
                elif self.icd_mdt_parameters['fvt_tcl'] * 0.512 >= rrint >= self.icd_mdt_parameters[
                    'vf_tcl'] * 0.512 and (self.fvt_combobox.currentText() != "OFF"):
                    self.ecgmarker_label = 'TF'
                elif self.icd_mdt_parameters['fvt_tcl'] * 0.512 >= rrint >= self.icd_mdt_parameters[
                    'vf_tcl'] * 0.512 and (self.fvt_combobox.currentText() == "OFF"):
                    self.ecgmarker_label = 'TS'
                else:
                    self.ecgmarker_label = 'VS'
                if not any(label in self.marker_label for label in ['TF', 'TD', 'FD']):
                    pen_col = pg.mkPen('k', width=1)
                else:
                    pen_col = pg.mkPen('r', width=1)


                markerecg_inf_line = pg.InfiniteLine(pos=rpeak, angle=90, movable=False,
                                                  label=self.ecgmarker_label[-1], markers='v',
                                                  pen=pen_col, span=(0.5, 1))

                markerecg_inf_line.label.setPosition(-0.2)
                self.markerecg_vvi_pw.addItem(markerecg_inf_line)


        for key, value in psa_data.items():
            signal = key.split('_')[0]
            if signal in ['ecg', 'ecg3']:
                color = '#0FA00F'
            if signal in ['rvbip', 'rvshock']:
                color = '#732F9B'
            if signal in ['laser1', 'laser2']:
                color = '#03c6fc'

            self['{}_vvi_plt'.format(signal)].setData(x=m_vvi_range, y=value, pen=color, symbol=None,
                                                       antialias=True,
                                                       autoDownsample=True, clipToView=True)

        self.overview_vvi_plt.setData(x=np.arange(samples), y=data, pen='#0FA00F',
                                      symbol=None,
                                      antialise=True, autoDownsample=True, clipToView=True)
        self.overview_infline = pg.InfiniteLine(pos=m_beat_end, angle=90, pen='#fffb1a80', movable=False)
        self.overview_vvi_pw.addItem(self.overview_infline)

        if self.trace_view >= (samples-1):
            self.timer_m.stop()


    def v_detection_gui(self):
        self.labels = []
        self.comboboxes = []

        self.m_vdetection_layout = QtWidgets.QVBoxLayout()
        self.warning_vbox = QtWidgets.QVBoxLayout()

        # Creating labels
        self.vf_lbl = QtWidgets.QLabel()
        self.vf_lbl.setText("VF")
        self.vf_lbl.setStyleSheet("font-weight: bold;")
        self.vf_lbl.setFixedWidth(50)
        self.fvt_lbl = QtWidgets.QLabel()
        self.fvt_lbl.setText("FVT")
        self.fvt_lbl.setFixedWidth(50)
        self.fvt_lbl.setStyleSheet("font-weight: bold;")
        self.vt_lbl = QtWidgets.QLabel()
        self.vt_lbl.setText("VT")
        self.vt_lbl.setStyleSheet("font-weight: bold;")
        self.vt_lbl.setFixedWidth(50)
        self.monitor_lbl = QtWidgets.QLabel()
        self.monitor_lbl.setText("Monitor")
        self.monitor_lbl.setFixedWidth(50)
        self.monitor_lbl.setStyleSheet("font-weight: bold;")
        self.initial_lbl = QtWidgets.QLabel()
        self.initial_lbl.setText("Initial")
        self.initial_lbl.setStyleSheet("font-weight: bold;")
        self.initial_lbl.setFixedWidth(100)
        self.redetect_lbl = QtWidgets.QLabel()
        self.redetect_lbl.setText("Redetect")
        self.redetect_lbl.setStyleSheet("font-weight: bold;")
        self.redetect_lbl.setFixedWidth(100)
        self.v_interval_lbl = QtWidgets.QLabel()
        self.v_interval_lbl.setText("V Interval (ms)")
        self.v_interval_lbl.setFixedWidth(100)
        self.v_interval_lbl.setStyleSheet("font-weight: bold;")

        self.pr_logic_lbl = QtWidgets.QLabel()
        self.pr_logic_lbl.setText("PR Logic")
        self.pr_logic_lbl.setFixedWidth(100)
        self.pr_logic_lbl.setStyleSheet("font-weight: bold;")
        self.af_fl_lbl = QtWidgets.QLabel()
        self.af_fl_lbl.setText("AF/Afl")
        self.af_fl_lbl.setFixedWidth(100)
        self.af_fl_lbl.setStyleSheet("font-weight: bold;")
        self.sinustach_lbl = QtWidgets.QLabel()
        self.sinustach_lbl.setText("Sinus Tach")
        self.sinustach_lbl.setFixedWidth(100)
        self.sinustach_lbl.setStyleSheet("font-weight: bold;")
        self.svt_lbl = QtWidgets.QLabel()
        self.svt_lbl.setText("Other 1:1 SVTs")
        self.svt_lbl.setFixedWidth(100)
        self.svt_lbl.setStyleSheet("font-weight: bold;")
        self.svt_vlimit_lbl = QtWidgets.QLabel()
        self.svt_vlimit_lbl.setText("SVT V Limit (ms)")
        self.svt_vlimit_lbl.setFixedWidth(100)
        self.svt_vlimit_lbl.setStyleSheet("font-weight: bold;")
        self.other_enhance_lbl = QtWidgets.QLabel()
        self.other_enhance_lbl.setText("VVI Discriminators")
        self.other_enhance_lbl.setFixedWidth(120)
        self.other_enhance_lbl.setStyleSheet("font-weight: bold;")
        self.stability_lbl = QtWidgets.QLabel()
        self.stability_lbl.setText("Stability")
        self.stability_lbl.setFixedWidth(75)
        self.stability_lbl.setStyleSheet("font-weight: bold;")
        self.onset_lbl = QtWidgets.QLabel()
        self.onset_lbl.setText("Onset")
        self.onset_lbl.setFixedWidth(75)
        self.onset_lbl.setStyleSheet("font-weight: bold;")

        self.wavelet_lbl_vvi.setStyleSheet("font-weight: bold;")
        self.wavelet_lbl_vvi.setFixedWidth(75)
        self.autocollection_lbl = QtWidgets.QLabel()
        self.autocollection_lbl.setText("Autocollection")
        self.autocollection_lbl.setFixedWidth(100)
        self.match_threshold_lbl = QtWidgets.QLabel()
        self.match_threshold_lbl.setText("Match Threshold")
        self.match_threshold_lbl.setFixedWidth(100)
        self.high_rate_timout_lbl = QtWidgets.QLabel()
        self.high_rate_timout_lbl.setText("High Rate\nTimeout")
        self.high_rate_timout_lbl.setFixedWidth(75)
        self.high_rate_timout_lbl.setStyleSheet("font-weight: bold;")
        self.sensitivity_lbl = QtWidgets.QLabel()
        self.sensitivity_lbl.setText("Sensitivity (mV)")
        self.sensitivity_lbl.setFixedWidth(100)
        self.sensitivity_lbl.setStyleSheet("font-weight: bold;")
        self.rv_lbl = QtWidgets.QLabel()
        self.rv_lbl.setText("RV")
        self.rv_lbl.setFixedWidth(100)
        self.rv_lbl.setStyleSheet("font-weight: bold;")
        self.vf_rate_trigger_lbl = QtWidgets.QLabel()
        self.vf_rate_trigger_lbl.setText("WARNING:\nVF DETECTED!")
        self.vf_rate_trigger_lbl.setStyleSheet("font-weight: bold; color: red; font-size: 42px;")
        self.vf_rate_trigger_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.vf_rate_trigger_lbl.hide()
        self.vt_rate_trigger_lbl = QtWidgets.QLabel()
        self.vt_rate_trigger_lbl.setText("WARNING:\nVT Detected")
        self.vt_rate_trigger_lbl.setStyleSheet("font-weight: bold; color: red; font-size: 42px;")
        self.vt_rate_trigger_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.vt_rate_trigger_lbl.hide()
        self.fvt_rate_vt_trigger_lbl = QtWidgets.QLabel()
        self.fvt_rate_vt_trigger_lbl.setText("WARNING:\nFVT Detected")
        self.fvt_rate_vt_trigger_lbl.setStyleSheet("font-weight: bold; color: red; font-size: 42px;")
        self.fvt_rate_vt_trigger_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.fvt_rate_vt_trigger_lbl.hide()
        self.fvt_rate_vf_trigger_lbl = QtWidgets.QLabel()
        self.fvt_rate_vf_trigger_lbl.setText("WARNING:\nFVT Detected")
        self.fvt_rate_vf_trigger_lbl.setStyleSheet("font-weight: bold; color: red; font-size: 42px;")
        self.fvt_rate_vf_trigger_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.fvt_rate_vf_trigger_lbl.hide()
        self.atp_delivery_lbl = QtWidgets.QLabel()
        self.atp_delivery_lbl.setText("DELIVERING ATP")
        self.atp_delivery_lbl.setStyleSheet("font-weight: bold; color: black; font-size: 20px;")
        self.atp_delivery_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.atp_delivery_lbl.hide()
        self.charging_lbl = QtWidgets.QLabel()
        self.charging_lbl.setText("CHARGING...")
        self.charging_lbl.setStyleSheet("font-weight: bold; color: black; font-size: 20px;")
        self.charging_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.charging_lbl.hide()
        self.shock_lbl = QtWidgets.QLabel()
        self.shock_lbl.setText("SHOCK")
        self.shock_lbl.setStyleSheet("font-weight: bold; color: red; font-size: 30px;")
        self.shock_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.shock_lbl.hide()

        self.onset_met_lbl = QtWidgets.QLabel()
        icon_path = "/Users/amiyazawa/Library/CloudStorage/Dropbox/01 Projects/Medtronic Algorithm/Medtronic Algorithm Python Code/icons/checkbox.png"  # Replace with the path to your icon image
        self.tick_icon = QtGui.QPixmap(icon_path)  # Replace with the path to your icon image
        icon_html = f'<img src="{icon_path}" width="{self.tick_icon.width()}" height="{self.tick_icon.height()}">'
        text = "&nbsp;&nbsp;ONSET"
        label_text = f'<span style="vertical-align:bottom;">{icon_html}</span>{text}'
        self.onset_met_lbl.setText(label_text)
        self.onset_met_lbl.setStyleSheet("font-weight: bold; color: black; font-size: 20px;")
        self.onset_met_lbl.hide()

        self.stability_met_lbl = QtWidgets.QLabel()
        text = "&nbsp;&nbsp;STABILITY"
        label_text = f'<span style="vertical-align:bottom;">{icon_html}</span>{text}'
        self.stability_met_lbl.setText(label_text)
        self.stability_met_lbl.setStyleSheet("font-weight: bold; color: black; font-size: 20px;")
        self.stability_met_lbl.hide()

        self.haem_unstable_lbl = QtWidgets.QLabel()
        text = "&nbsp;&nbsp;HAEMODYNAMIC<br>&nbsp;&nbsp;COMPROMISE"
        label_text = f'<span style="vertical-align:bottom;">{icon_html}</span>{text}'
        self.haem_unstable_lbl.setText(label_text)
        self.haem_unstable_lbl.setStyleSheet("font-weight: bold; color: black; font-size: 20px;")
        self.haem_unstable_lbl.hide()

        self.morphology_met_lbl = QtWidgets.QLabel()
        text = "&nbsp;&nbsp;MORPHOLOGY"
        label_text = f'<span style="vertical-align:bottom;">{icon_html}</span>{text}'
        self.morphology_met_lbl.setText(label_text)
        self.morphology_met_lbl.setStyleSheet("font-weight: bold; color: black; font-size: 20px;")
        self.morphology_met_lbl.hide()
        # Creating Comboboxes
        # DETECTION COMBOBOXES
        self.vf_combobox = QtWidgets.QComboBox()
        self.vf_combobox.addItems("OFF ON".split())
        self.vf_combobox.setCurrentIndex(1)  # Nominal is ON
        self.vf_combobox.setFixedWidth(100)
        self.vf_combobox.currentIndexChanged.connect(self.vf_combobox_changed)

        self.fvt_combobox = QtWidgets.QComboBox()
        self.fvt_combobox.addItems("OFF via-VT via-VF".split())  # Nominal is OFF
        self.fvt_combobox.setCurrentIndex(0)
        self.fvt_combobox.setFixedWidth(100)
        self.fvt_combobox.currentIndexChanged.connect(self.fvt_combobox_changed)

        self.vt_combobox = QtWidgets.QComboBox()
        self.vt_combobox.addItems("OFF ON".split())
        self.vt_combobox.setCurrentIndex(0)  # Nominal is OFF
        self.vt_combobox.setFixedWidth(100)
        self.vt_combobox.currentIndexChanged.connect(self.vt_combobox_changed)

        self.monitor_combobox = QtWidgets.QComboBox()
        self.monitor_combobox.addItems("OFF ON".split())
        self.monitor_combobox.setCurrentIndex(0)  # Nominal is OFF
        self.monitor_combobox.setFixedWidth(100)
        self.monitor_combobox.currentIndexChanged.connect(self.monitor_combobox_changed)

        self.initial_vf_combobox = QtWidgets.QComboBox()
        self.initial_vf_combobox.addItems("12/16 18/24 24/32 30/40 45/60 60/80 75/100 90/120 105/140 120/160".split())
        self.initial_vf_combobox.setCurrentIndex(3)  # Nominal is 30/40 beats
        self.initial_vf_combobox.setFixedWidth(100)
        self.initial_vf_combobox.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.initial_vf_combobox.currentIndexChanged.connect(self.initial_vf_combobox_changed)

        self.redetect_vf_combobox = QtWidgets.QComboBox()
        self.redetect_vf_combobox.addItems("6/8 9/12 12/16 18/24 24/32 27/36 30/40".split())
        self.redetect_vf_combobox.setCurrentIndex(2)  # Nominal is 12/16 beats
        self.redetect_vf_combobox.setFixedWidth(100)
        self.redetect_vf_combobox.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.redetect_vf_combobox.currentIndexChanged.connect(self.redetect_vf_combobox_changed)
        self.vf_redetect_nid = self.redetect_vf_combobox.currentText()
        self.vf_redetect_min_nid = self.vf_redetect_nid.split("/")[0]
        self.vf_redetect_max_nid = self.vf_redetect_nid.split("/")[1]

        self.initial_vt_combobox = QtWidgets.QComboBox()
        self.initial_vt_combobox.addItems("12 16 18 24 30 45 52 76 100".split())
        self.initial_vt_combobox.setCurrentIndex(1)  # Nominal is 16 beats
        self.initial_vt_combobox.setFixedWidth(100)
        self.initial_vt_combobox.currentIndexChanged.connect(self.initial_vt_combobox_changed)
        self.initial_vt_combobox.hide()

        self.redetect_vt_combobox = QtWidgets.QComboBox()
        self.redetect_vt_combobox.addItems("8 12 16 20 24 28 32 36 40 44 48 52".split())
        self.redetect_vt_combobox.setCurrentIndex(1)  # Nominal is 12 beats
        self.redetect_vt_combobox.setFixedWidth(100)
        self.redetect_vt_combobox.currentIndexChanged.connect(self.redetect_vt_combobox_changed)
        self.redetect_vt_combobox.hide()

        self.monitor_vttcl_combobox = QtWidgets.QComboBox()
        self.monitor_vttcl_combobox.addItems(
            "280 290 300 310 320 330 340 350 360 370 380 390 400 410 420 430 440 450 460 470 480 490 500 510 520 530 540 550 560 570 580 590 600 610 620 630 640 650".split())
        self.monitor_vttcl_combobox.setCurrentIndex(17)  # Nominal is 410 ms
        self.monitor_vttcl_combobox.setFixedWidth(100)
        self.monitor_vttcl_combobox.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.monitor_vttcl_combobox.currentIndexChanged.connect(self.monitor_vttcl_combobox_changed)
        self.monitor_vttcl_combobox.hide()

        self.detect_monitor_vt_combobox = QtWidgets.QComboBox()
        self.detect_monitor_vt_combobox.addItems("16 20 24 28 32 36 40 44 48 52 56 80 110 130".split())
        self.detect_monitor_vt_combobox.setCurrentIndex(4)  # Nominal is 32 beats
        self.detect_monitor_vt_combobox.setFixedWidth(100)
        self.detect_monitor_vt_combobox.currentIndexChanged.connect(self.detect_monitor_vt_combobox_changed)
        self.detect_monitor_vt_combobox.hide()

        self.vf_tcl_combobox = QtWidgets.QComboBox()
        self.vf_tcl_combobox.addItems("240 250 260 270 280 290 300 310 320 330 340 350 360 370 380 390 400".split())
        self.vf_tcl_combobox.setCurrentIndex(8)  # 320 is nominal setting
        self.vf_tcl_combobox.setFixedWidth(100)
        self.vf_tcl_combobox.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.vf_tcl_combobox.currentIndexChanged.connect(self.vf_tcl_combobox_changed)

        self.fvt_tcl_combobox = QtWidgets.QComboBox()
        self.fvt_tcl_combobox.addItems(
            "200 210 220 230 240 250 260 270 280 290 300 310 320 330 340 350 360 370 380 390 400 410 420 430 440 450 460 470 480 490 500 510 520 530 540 550 560 570 580 590 600".split())
        self.fvt_tcl_combobox.setCurrentIndex(4)  # 240 is nominal setting
        self.fvt_tcl_combobox.setFixedWidth(100)
        self.fvt_tcl_combobox.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.fvt_tcl_combobox.currentIndexChanged.connect(self.fvt_tcl_combobox_changed)
        self.fvt_tcl_combobox.hide()

        self.vt_tcl_combobox = QtWidgets.QComboBox()
        self.vt_tcl_combobox.addItems(
            "280 290 300 310 320 330 340 350 360 370 380 390 400 410 420 430 440 450 460 470 480 490 500 510 520 530 540 550 560 570 580 590 600 610 620 630 640 650".split())
        self.vt_tcl_combobox.setCurrentIndex(8)  # 360 is nominal setting
        self.vt_tcl_combobox.setFixedWidth(100)
        self.vt_tcl_combobox.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.vt_tcl_combobox.currentIndexChanged.connect(self.vt_tcl_combobox_changed)
        self.vt_tcl_combobox.hide()
        #     PR LOGIC COMBOBOXES
        self.af_fl_combobox = QtWidgets.QComboBox()
        self.af_fl_combobox.addItems("OFF ON".split())
        self.af_fl_combobox.setCurrentIndex(1)  # ON is nominal setting
        self.af_fl_combobox.setFixedWidth(100)
        self.af_fl_combobox.currentIndexChanged.connect(self.af_fl_combobox_changed)

        self.sinustach_combobox = QtWidgets.QComboBox()
        self.sinustach_combobox.addItems("OFF ON".split())
        self.sinustach_combobox.setCurrentIndex(1)  # ON is nominal setting
        self.sinustach_combobox.setFixedWidth(100)
        self.sinustach_combobox.currentIndexChanged.connect(self.sinustach_combobox_changed)

        self.other_svts_combobox = QtWidgets.QComboBox()
        self.other_svts_combobox.addItems("OFF ON".split())
        self.other_svts_combobox.setCurrentIndex(0)  # OFF is nominal setting
        self.other_svts_combobox.setFixedWidth(100)
        self.other_svts_combobox.currentIndexChanged.connect(self.other_svts_combobox_changed)

        self.svt_v_limit_combobox = QtWidgets.QComboBox()
        self.svt_v_limit_combobox.addItems(
            "OFF 240 250 260 270 280 290 300 310 320 330 340 350 360 370 380 390 400 410 420 430 440 450 460 470 480 490 500 510 520 530 540 550 560 570 580 590 600 610 620 630 640 650".split())
        self.svt_v_limit_combobox.setCurrentIndex(0)  # OFF is nominal setting
        self.svt_v_limit_combobox.setFixedWidth(100)
        self.svt_v_limit_combobox.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.svt_v_limit_combobox.currentIndexChanged.connect(self.svt_v_limit_combobox_changed)

        # Other Enhancements (all BELOW - up to wavelet)
        self.stability_combobox = QtWidgets.QComboBox()
        self.stability_combobox.addItems("OFF 30 40 50 60 70 80 90 100".split())
        self.stability_combobox.setCurrentIndex(0)  # OFF is nominal setting
        self.stability_combobox.setFixedWidth(100)
        self.stability_combobox.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.stability_combobox.currentIndexChanged.connect(self.stability_combobox_changed)

        # ONSET Comboboxes
        self.onset_combobox = QtWidgets.QComboBox()
        self.onset_combobox.addItems("OFF ON MONITOR".split())
        self.onset_combobox.setCurrentIndex(0)  # OFF is nominal setting
        self.onset_combobox.setFixedWidth(100)
        self.onset_combobox.currentIndexChanged.connect(self.onset_combobox_changed)

        self.onset_pct_combobox = QtWidgets.QComboBox()
        self.onset_pct_combobox.addItems("72 75 78 81 84 88 91 94 97".split())
        self.onset_pct_combobox.setCurrentIndex(3)  # 81 is nominal setting
        self.onset_pct_combobox.setFixedWidth(100)
        self.onset_pct_combobox.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.onset_pct_combobox.currentIndexChanged.connect(self.onset_pct_combobox_changed)

        # High Rate Timeout Comboboxes
        self.high_rate_vftimeout_combobox = QtWidgets.QComboBox()  # In mins
        self.high_rate_vftimeout_combobox.addItems("OFF 0.25 0.5 0.75 1 1.25 1.5 1.75 2 2.5 3 3.5 4 4.5 5".split())
        self.high_rate_vftimeout_combobox.setCurrentIndex(0)  # OFF is nominal setting
        self.high_rate_vftimeout_combobox.setFixedWidth(100)
        self.high_rate_vftimeout_combobox.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.high_rate_vftimeout_combobox.currentIndexChanged.connect(self.highrate_vftimeout_combobox_changed)

        self.high_rate_allzone_timeout_combobox = QtWidgets.QComboBox()  # In mins
        self.high_rate_allzone_timeout_combobox.addItems(
            "OFF 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 26 28 30".split())
        self.high_rate_allzone_timeout_combobox.setCurrentIndex(0)  # OFF is nominal setting
        self.high_rate_allzone_timeout_combobox.setFixedWidth(100)
        self.high_rate_allzone_timeout_combobox.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.high_rate_allzone_timeout_combobox.currentIndexChanged.connect(
            self.highrate_allzone_timeout_combobox_changed)

        # RV lead Noise and T Wave Comboboxes
        self.twave_combobox = QtWidgets.QComboBox()
        self.twave_combobox.addItems("OFF ON".split())
        self.twave_combobox.setCurrentIndex(1)  # ON is nominal setting
        self.twave_combobox.setFixedWidth(100)
        self.twave_combobox.currentIndexChanged.connect(self.twave_combobox_changed)

        self.rv_lead_noise_combobox = QtWidgets.QComboBox()
        self.rv_lead_noise_combobox.addItems("OFF ON On+Timeout".split())
        self.rv_lead_noise_combobox.setCurrentIndex(1)  # ON is nominal setting
        self.rv_lead_noise_combobox.setFixedWidth(100)
        self.rv_lead_noise_combobox.currentIndexChanged.connect(self.rv_lead_noise_combobox_changed)

        self.rv_lead_noise_timeout_combobox = QtWidgets.QComboBox()  # In mins
        self.rv_lead_noise_timeout_combobox.addItems("0.25 0.5 0.75 1 1.25 1.5 1.75 2".split())
        self.rv_lead_noise_timeout_combobox.setCurrentIndex(2)  # 0.75 is nominal setting
        self.rv_lead_noise_timeout_combobox.setFixedWidth(100)
        self.rv_lead_noise_timeout_combobox.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.rv_lead_noise_timeout_combobox.currentIndexChanged.connect(self.rv_lead_noise_timeout_combobox_changed)

        # Wavelet Comboboxes
        self.wavelet_combobox = QtWidgets.QComboBox()
        self.wavelet_combobox.addItems("OFF ON".split())
        self.wavelet_combobox.setCurrentIndex(1)  # ON is nominal setting
        self.wavelet_combobox.setFixedWidth(100)
        self.wavelet_combobox.currentIndexChanged.connect(self.wavelet_combobox_changed)

        self.wavelet_match_threshold_combobox = QtWidgets.QComboBox()
        self.wavelet_match_threshold_combobox.addItems(
            "40 43 46 49 52 55 58 61 64 67 70 73 76 79 82 85 88 91 94 97".split())
        self.wavelet_match_threshold_combobox.setCurrentIndex(10)  # 70 is nominal setting
        self.wavelet_match_threshold_combobox.setFixedWidth(100)
        self.wavelet_match_threshold_combobox.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.wavelet_match_threshold_combobox.currentIndexChanged.connect(self.wavelet_match_threshold_combobox_changed)

        self.wavelet_autocollection_combobox = QtWidgets.QComboBox()
        self.wavelet_autocollection_combobox.addItems("OFF ON".split())
        self.wavelet_autocollection_combobox.setCurrentIndex(1)  # ON is nominal setting
        self.wavelet_autocollection_combobox.setFixedWidth(100)
        self.wavelet_autocollection_combobox.currentIndexChanged.connect(self.wavelet_autocollection_combobox_changed)

        # self.other_discrim_hbox = QtWidgets.QHBoxLayout()
        self.prlogic_grid = QtWidgets.QGridLayout()
        self.other_enhance_grid = QtWidgets.QGridLayout()

        self.prlogic_grid.addWidget(self.pr_logic_lbl, 0, 0)
        self.prlogic_grid.addWidget(self.af_fl_lbl, 1, 0)
        self.prlogic_grid.addWidget(self.af_fl_combobox, 1, 1)
        self.prlogic_grid.addWidget(self.sinustach_lbl, 2, 0)
        self.prlogic_grid.addWidget(self.sinustach_combobox, 2, 1)
        self.prlogic_grid.addWidget(self.svt_lbl, 3, 0)
        self.prlogic_grid.addWidget(self.other_svts_combobox, 3, 1)
        self.prlogic_grid.addWidget(self.spacer_lbl, 4, 0)
        self.prlogic_grid.addWidget(self.svt_vlimit_lbl, 5, 0)
        self.prlogic_grid.addWidget(self.svt_v_limit_combobox, 5, 1)

        self.other_enhance_grid.addWidget(self.other_enhance_lbl, 0, 1)
        self.other_enhance_grid.addWidget(self.stability_lbl, 1, 0)
        self.other_enhance_grid.addWidget(self.stability_combobox, 1, 1)
        self.other_enhance_grid.addWidget(self.onset_lbl, 2, 0)
        self.other_enhance_grid.addWidget(self.onset_combobox, 2, 1)
        self.other_enhance_grid.addWidget(self.onset_pct_combobox, 3, 1)
        self.onset_pct_combobox.hide()
        self.other_enhance_grid.addWidget(self.wavelet_lbl_vvi, 4, 0)
        self.other_enhance_grid.addWidget(self.wavelet_combobox, 4, 1)
        self.other_enhance_grid.addWidget(self.match_threshold_lbl, 5, 0)
        self.other_enhance_grid.addWidget(self.wavelet_match_threshold_combobox, 5, 1)
        # self.match_threshold_lbl.hide()
        # self.wavelet_match_threshold_combobox.hide()
        self.other_enhance_grid.addWidget(self.autocollection_lbl, 6, 0)
        self.other_enhance_grid.addWidget(self.wavelet_autocollection_combobox, 6, 1)
        # self.autocollection_lbl.hide()
        # self.wavelet_autocollection_combobox.hide()
        self.other_enhance_grid.addWidget(self.high_rate_timout_lbl, 7, 0)
        self.other_enhance_grid.addWidget(self.high_rate_vftimeout_combobox, 7, 1)
        self.other_enhance_grid.addWidget(self.high_rate_allzone_timeout_combobox, 8, 1)
        self.high_rate_allzone_timeout_combobox.hide()
        self.other_enhance_grid.setSpacing(0)
        self.other_enhance_grid.setContentsMargins(0, 20, 0, 0)

        # V Detection into Layout
        self.v_detection_layout = QtWidgets.QHBoxLayout()
        self.tx_zone_layout = QtWidgets.QGridLayout()
        self.tx_zone_layout.addWidget(self.initial_lbl, 0, 2)
        self.tx_zone_layout.addWidget(self.redetect_lbl, 0, 3)
        self.tx_zone_layout.addWidget(self.v_interval_lbl, 0, 4)
        self.tx_zone_layout.addWidget(self.vf_lbl, 1, 0)
        self.tx_zone_layout.addWidget(self.vf_combobox, 1, 1)
        self.tx_zone_layout.addWidget(self.initial_vf_combobox, 1, 2)
        self.tx_zone_layout.addWidget(self.redetect_vf_combobox, 1, 3)
        self.tx_zone_layout.addWidget(self.vf_tcl_combobox, 1, 4)
        self.tx_zone_layout.addWidget(self.fvt_lbl, 2, 0)
        self.tx_zone_layout.addWidget(self.fvt_combobox, 2, 1)
        self.tx_zone_layout.addWidget(self.fvt_tcl_combobox, 2, 4)
        self.tx_zone_layout.addWidget(self.vt_lbl, 3, 0)
        self.tx_zone_layout.addWidget(self.vt_combobox, 3, 1)
        self.tx_zone_layout.addWidget(self.initial_vt_combobox, 3, 2)
        self.tx_zone_layout.addWidget(self.redetect_vt_combobox, 3, 3)
        self.tx_zone_layout.addWidget(self.vt_tcl_combobox, 3, 4)
        self.tx_zone_layout.addWidget(self.monitor_lbl, 4, 0)
        self.tx_zone_layout.addWidget(self.monitor_combobox, 4, 1)
        self.tx_zone_layout.addWidget(self.detect_monitor_vt_combobox, 4, 3)
        self.tx_zone_layout.addWidget(self.monitor_vttcl_combobox, 4, 4)
        self.warning_vbox.addWidget(self.vf_rate_trigger_lbl)
        self.warning_vbox.addWidget(self.fvt_rate_vf_trigger_lbl)
        self.warning_vbox.addWidget(self.fvt_rate_vt_trigger_lbl)
        self.warning_vbox.addWidget(self.vt_rate_trigger_lbl)
        self.warning_vbox.addWidget(self.atp_delivery_lbl)
        self.warning_vbox.addWidget(self.charging_lbl)
        self.warning_vbox.addWidget(self.shock_lbl)
        self.criteria_vbox = QtWidgets.QVBoxLayout()
        self.criteria_vbox.setAlignment(QtCore.Qt.AlignRight)
        self.criteria_vbox.setSpacing(5)
        self.criteria_vbox.setContentsMargins(20, 0, 20, 0)
        self.criteria_vbox.addWidget(self.onset_met_lbl)
        self.criteria_vbox.addWidget(self.stability_met_lbl)
        self.criteria_vbox.addWidget(self.morphology_met_lbl)
        self.criteria_vbox.addWidget(self.haem_unstable_lbl)

        self.warning_vbox.setAlignment(QtCore.Qt.AlignCenter)
        self.warning_vbox.setContentsMargins(0, 0, 0, 0)
        self.tx_zone_layout.setSpacing(10)
        self.tx_zone_layout.setContentsMargins(0, 0, 0, 0)
        self.tx_zone_layout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.v_detection_layout.setSpacing(0)
        self.v_detection_layout.setContentsMargins(0, 0, 0, 0)
        self.v_detection_layout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.v_detection_layout.addLayout(self.prlogic_grid)
        self.v_detection_layout.addLayout(self.tx_zone_layout)
        self.v_detection_layout.addLayout(self.criteria_vbox)
        self.v_detection_layout.addLayout(self.warning_vbox)

    # Media Button Clicks
    def start_btn_clicked_vvi(self):
        self.timer_m.start(1000)

    def pause_btn_clicked_vvi(self):
        self.timer_m.stop()

    def restart_btn_clicked_vvi(self):
        self.index1 = 0
        self.trace_view = 0
        self.icd_memory['active_tachy'] = False
        self.onset_met_lbl.hide()
        self.vf_rate_trigger_lbl.hide()
        self.vf_rate_trigger() == False
        self.stability_met_lbl.hide()
        self.morphology_met_lbl.hide()
        self.fvt_rate_vt_trigger_lbl.hide()
        self.atp_delivery_lbl.hide()
        self.charging_lbl.hide()
        self.overview_vvi_pw.removeItem(self.overview_infline)
        self.update_medtronic_plots()

    # Toggle Changes
    def laser1_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.laser1_vvi_pw.show()
        else:
            self.laser1_vvi_pw.hide()

    def laser2_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.laser2_vvi_pw.show()
        else:
            self.laser2_vvi_pw.hide()

    def ecg_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.ecg_vvi_pw.show()
        else:
            self.ecg_vvi_pw.hide()

    def rvshock_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.rvshock_vvi_pw.show()
        else:
            self.rvshock_vvi_pw.hide()

    def rvbip_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.rvbip_vvi_pw.show()
        else:
            self.rvbip_vvi_pw.hide()

    def marker_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.marker_vvi_pw.show()

        else:
            self.marker_vvi_pw.hide()
    def ecgmarker_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.ecgmarker_vvi_pw.show()
        else:
            self.ecgmarker_vvi_pw.hide()

    def overview_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.overview_vvi_pw.show()
        else:
            self.overview_vvi_pw.hide()

    def rvbip_threshold_line_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.update_rvst()
            self.update_medtronic_plots()
        elif QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Unchecked:
            self.rvbip_vvi_pi.removeItem(self.rvbip_threshold_inf_line)
            self.update_rvst()
            self.update_medtronic_plots()
        else:
            print("error")

    def rv_sens_threshold_changed(self, object):
        btn_id = self.rvst_group.id(object)
        self.value = self.rvst_group.button(btn_id).text()
        print("RV Sensing Threshold: " + str(self.value))
        try:
            self.rvbip_vvi_pi.removeItem(self.rvbip_threshold_inf_line)
        except:
            pass
        print("RV Sensing Threshold: " + str(self.value))
        # self.update_rvst()
        self.update_medtronic_plots()

    def update_rvst(self):
        self.icd_mdt_parameters['rvst_value'] = float(self.rvst_group.checkedButton().text())
        for button in self.rvst_group.buttons():
            if button.isChecked() == True:
                button.setStyleSheet("font-weight: bold")
                print('button checked: ', button.text())
                try:
                    self.rvbip_vvi_pi.addItem(self.rvbip_threshold_inf_line)
                except:
                    pass
            else:
                try:
                    self.rvbip_vvi_pi.removeItem(self.rvbip_threshold_inf_line)
                except:
                    pass
                button.setStyleSheet("font-weight: normal")
        self.update_medtronic_plots()

        print('rvst value: ', self.icd_mdt_parameters['rvst_value'])

    def indication_changed(self):
        # Primary Prevention Guideline Settings (HRS)
        # "primary" or "secondary" or "am_test"
        if self.indication.currentText() == "Primary":
            print("Primary Prevention Settings")

            self.vf_tcl_combobox.setCurrentText('320')  # 188 bpm
            self.icd_mdt_parameters['vf_tcl'] = int(self.vf_tcl_combobox.currentText())
            self.fvt_tcl_combobox.setCurrentIndex(0)  # OFF
            self.vt_tcl_combobox.setCurrentIndex(0)  # OFF
            self.icd_mdt_parameters['vf_min_nid']  = 30
            self.icd_mdt_parameters['vf_max_nid']  = 40
            self.icd_mdt_parameters['vt_nid']  = 0
            self.monitor_combobox.setCurrentText("OFF")
            self.vf_combobox.setCurrentIndex(1)
            self.vt_combobox.setCurrentIndex(0)
            self.fvt_combobox.setCurrentIndex(0)
            self.monitor_combobox.setCurrentIndex(0)

        # Secondary Prevention Guideline Settings (HRS)
        if self.indication.currentText() == "Secondary":
            print("Secondary Prevention Settings")
            self.SecondaryPreventionWindow = SecondaryPreventionWindow()
            self.SecondaryPreventionWindow.show()

            self.output = self.SecondaryPreventionWindow.get_secondary_vt_cycle_length()

            # Secondary Prevention Parameters
            # VF Zone Parameters
            self.icd_mdt_parameters['vf_tcl']  = 319  # 188 bpm
            self.icd_mdt_parameters['vf_min_nid']  = 30
            self.icd_mdt_parameters['vf_max_nid']  = 40

            # VT Zone Parameters
            self.icd_mdt_parameters['fvt_tcl']  = 0  # OFF
            self.icd_mdt_parameters['vt_tcl']  = 0
            self.icd_mdt_parameters['vt_nid']  = 24  # consecutive

            if ((int(self.output) - 20) < 319):
                self.icd_mdt_parameters['vf_tcl']  = 320  # 188 bpm
                self.icd_mdt_parameters['fvt_tcl']  = 0  # OFF
                self.icd_mdt_parameters['vt_tcl']  = 0  # OFF
                self.icd_mdt_parameters['monitor_tcl']  = 0

                self.vf_combobox.setCurrentIndex(1)
                self.vt_combobox.setCurrentIndex(0)
                self.fvt_combobox.setCurrentIndex(0)
                self.monitor_combobox.setCurrentIndex(0)

            else:
                self.icd_mdt_parameters['vf_tcl']  = 320
                self.icd_mdt_parameters['fvt_tcl']  = int(self.icd_mdt_parameters['vt_tcl'] ) - 20
                self.icd_mdt_parameters['vt_tcl']  = 0  # OFF or MONITOR?
                self.icd_mdt_parameters['monitor_tcl']  = int(self.output) + 20

                self.vf_combobox.setCurrentIndex(1)
                self.fvt_combobox.setCurrentIndex(1)
                self.vt_combobox.setCurrentIndex(0)
                self.monitor_combobox.setCurrentIndex(1)
            # Print Settings
            print("Previous Known VT Cycle Length: " + str(self.output))

        if self.indication.currentText() == "AMTest":
            print("AM Test Settings")
            # VF Zone Parameters
            self.vf_tcl_combobox.setCurrentIndex(9)  # 180 bpm
            self.initial_vf_combobox.setCurrentIndex(4)
            self.vf_detect_nid = self.initial_vf_combobox.currentText()
            self.icd_mdt_parameters['vf_min_nid']  = int(self.vf_detect_nid.split("/")[0])
            self.icd_mdt_parameters['vf_max_nid']  = int(self.vf_detect_nid.split("/")[1])

            # VT Zone Parameters
            self.fvt_tcl_combobox.setCurrentIndex(18)  # 160 bpm
            self.vt_tcl_combobox.setCurrentIndex(15)  # 120 bpm - set lower
            self.initial_vt_combobox.setCurrentIndex(4)
            self.icd_mdt_parameters['vt_nid']  = int(self.initial_vt_combobox.currentText())

            self.vf_combobox.setCurrentIndex(1)
            self.vt_combobox.setCurrentIndex(1)
            self.fvt_combobox.setCurrentIndex(1)
            self.monitor_combobox.setCurrentIndex(0)

            self.stability_combobox.setCurrentIndex(2)
            self.onset_combobox.setCurrentIndex(1)
            self.rvst5.setChecked(True)
            # self.rvst2.setChecked(False)

        else:
            print("Unknown Indication")

        # Print ICD Zone Settings
        print("VF Zone: " + str(int(60000 / self.icd_mdt_parameters['vf_tcl'] )))
        print("VF NID: " + str(self.icd_mdt_parameters['vf_min_nid'] ) + "/" + str(self.icd_mdt_parameters['vf_max_nid'] ))

        if self.icd_mdt_parameters['fvt_tcl']  == 0:  # OFF
            print("FVT Zone: OFF")

        else:
            print("FVT Zone: ON")
            print("VT Zone 1: " + str(int(60000 / self.icd_mdt_parameters['fvt_tcl'] )))

            print("FVT Programmed as: " + str(self.fvt_combobox.currentText()))

        if self.fvt_combobox.currentText() == "via-VT":
            print("VT NID: " + str(self.icd_mdt_parameters['vt_nid'] ))
        elif self.fvt_combobox.currentText() == "via-VF":

            print("VF NID: " + str(self.icd_mdt_parameters['vf_min_nid'] ) + "/" + str(self.icd_mdt_parameters['vf_max_nid'] ))

        if self.icd_mdt_parameters['vt_tcl']  == 0:  # OFF
            print("VT Zone 2: OFF")

        else:
            print("VT Zone 2: ON")
            print("VT Zone 2: " + str(int(60000 / self.icd_mdt_parameters['vt_tcl'] )))
            print("VT NID: " + str(self.icd_mdt_parameters['vt_nid'] ))

    def postvsblanking_changed(self, object):
        value = self.pvsb_group.id(object)

        print("Post VS Blanking: " + str(value))

        self.update_pvsb()

    def update_pvsb(self):
        self.icd_mdt_parameters['pvsb_value'] = int(self.pvsb_group.checkedButton().text())

        for button in self.pvsb_group.buttons():
            if button.isChecked() == True:
                button.setStyleSheet("font-weight: bold")
                print('button checked: ', button.text())

            else:
                button.setStyleSheet("font-weight: normal")

        print(str(self.pvsb_group.buttons()))

        print('pvsb value', self.icd_mdt_parameters['pvsb_value'])
        self.update_medtronic_plots()

    def pacing_dep_changed(self):
        print("Pacing Dependent: " + str(self.pacing_dep.currentText()))

    def calc_results_beats(self, begin, end):

        self.laser_exp.begin = int(begin)
        self.laser_exp.end = int(end)

        try:
            self.laser_exp.process()
            print(self.laser_exp.results)
        except Exception as e:
            print("Problem in calculation")
            print(e)
        else:
            self.laser1_val_vt = float(round(self.laser_exp.results['Laser1_Magic'], 4))
            self.laser1_conf_vt = float(round(self.laser_exp.results['Laser1_Conf'], 4))
            self.laser2_val_vt = float(round(self.laser_exp.results['Laser2_Magic'], 4))
            self.laser2_conf_vt = float(round(self.laser_exp.results['Laser2_Conf'], 4))

            self.laser1_value_vvi.setText(
                "Laser1: " + str(round(self.laser_exp.results['Laser1_Magic'], 4)) + "(Laser1 Conf: " + str(
                    round(self.laser_exp.results['Laser1_Conf'], 4)) + ")")
            self.laser2_value_vvi.setText(
                "Laser2: " + str(round(self.laser_exp.results['Laser2_Magic'], 4)) + "(Laser2 Conf: " + str(
                    round(self.laser_exp.results['Laser2_Conf'], 4)) + ")")

            self.rr_value_vvi.setText(
                "RR (RV Bipolar Lead): " + str(int(np.mean(np.diff(self.laser_exp.ecg.peaks_sample)))))
            self.hr_value_vvi.setText("HR: " + str(int(60000 / np.mean(np.diff(self.laser_exp.ecg.peaks_sample)))))

            try:
                self.sbp_value_vvi.setText("SBP: " + str(self.laser_exp.results['SBP_Mean']))
                self.map_value_vvi.setText("MAP: " + str(self.laser_exp.results['MAP_Mean']))

            except Exception as e:
                print(e)

            # dual_laser_window = DualLaserWindow(
            #     laser1_array_ys=100 * (np.exp(self.laser_exp.laser1_magic_data_all) - 1),
            #     laser1_sum=100 * (np.exp(self.laser_exp.laser1_magic_data) - 1),
            #     laser2_array_ys=100 * (np.exp(self.laser_exp.laser2_magic_data_all) - 1),
            #     laser2_sum=100 * (np.exp(self.laser_exp.laser2_magic_data) - 1),
            #     parent=self)
            # dual_laser_window.show()


    def mdtc_onset_trigger(self, a, b):
        mean_a = np.mean(np.array(a))
        mean_b = np.mean(np.array(b))

        onset_trigger = ((mean_a * self.icd_mdt_parameters['onset_pct'] ) > mean_b)
        onset_triggered_true = False
        if (onset_trigger == True) and ((len(a) + len(b)) == 8):
            if onset_triggered_true:
                print("\033[31mVT Onset Criteria Met\033[0m")
                return onset_trigger
                # break
            onset_triggered_true = True
        else:
            onset_trigger = None
            pass
        return onset_trigger

    # V Detection Combobox Connections
    def vf_combobox_changed(self):  # VF THERAPIES ON/OFF
        if self.vf_combobox.currentText() == 'OFF':
            print('VF therapies are OFF')
            self.initial_vf_combobox.hide()
            self.redetect_vf_combobox.hide()
            self.vf_tcl_combobox.hide()
        else:
            print('VF therapies are ON')
            self.initial_vf_combobox.show()
            self.redetect_vf_combobox.show()
            self.vf_tcl_combobox.show()
            self.vf_rate_trigger()

        self.update_medtronic_plots()

    def fvt_combobox_changed(self):  # FVT THERAPIES ON/OFF
        if self.fvt_combobox.currentText() == 'OFF':
            print('FVT therapies are OFF')
            self.fvt_tcl_combobox.hide()
        else:
            print('FVT is ON')
            self.fvt_tcl_combobox.show()
        self.update_medtronic_plots()

    def vt_combobox_changed(self):  # VT THERAPIES ON/OFF
        if self.vt_combobox.currentText() == 'OFF':
            print('VT therapies are OFF')

            self.initial_vt_combobox.hide()
            self.redetect_vt_combobox.hide()
            self.vt_tcl_combobox.hide()

        else:
            print('VT therapies are ON')
            self.initial_vt_combobox.show()
            self.redetect_vt_combobox.show()
            self.vt_tcl_combobox.show()
        self.update_medtronic_plots()

    def monitor_combobox_changed(self):  # VT Monitor Zone ON/OFF
        if self.monitor_combobox.currentText() == 'OFF':
            print('VT Monitor Zone is OFF')
            self.detect_monitor_vt_combobox.hide()
            self.monitor_vttcl_combobox.hide()

        else:
            print('VT Monitor Zone is ON at ', self.icd_mdt_parameters['monitor_tcl'] )

            self.detect_monitor_vt_combobox.show()
            self.monitor_vttcl_combobox.show()

        self.update_medtronic_plots()

    def initial_vf_combobox_changed(self):  # INITIAL VF NID
        self.vf_detect_nid = self.initial_vf_combobox.currentText()

        self.icd_mdt_parameters['vf_min_nid']  = int(self.vf_detect_nid.split("/")[0])
        self.icd_mdt_parameters['vf_max_nid']  = int(self.vf_detect_nid.split("/")[1])

        print("Number of VF beats to detect: ", self.vf_detect_nid)
        print("Min VF NID: ", self.icd_mdt_parameters['vf_min_nid'] )
        print("Max VF NID: ", self.icd_mdt_parameters['vf_max_nid'] )
        self.update_medtronic_plots()

    def redetect_vf_combobox_changed(self):  # REDETECT VF NID
        print(self.redetect_vf_combobox.currentText())
        self.update_medtronic_plots()

    def vf_tcl_combobox_changed(self):
        print(self.vf_tcl_combobox.currentText())
        self.update_medtronic_plots()
        self.icd_mdt_parameters['vf_tcl']  = int(self.vf_tcl_combobox.currentText())

    def fvt_tcl_combobox_changed(self):
        print(self.fvt_tcl_combobox.currentText())
        self.update_medtronic_plots()
        self.icd_mdt_parameters['fvt_tcl']  = int(self.fvt_tcl_combobox.currentText())

    def initial_vt_combobox_changed(self):
        print(self.initial_vt_combobox.currentText())
        self.update_medtronic_plots()

    def redetect_vt_combobox_changed(self):
        print(self.redetect_vt_combobox.currentText())
        self.update_medtronic_plots()

    def vt_tcl_combobox_changed(self):
        print(self.vt_tcl_combobox.currentText())
        self.icd_mdt_parameters['vt_tcl']  = int(self.vt_tcl_combobox.currentText())
        self.update_medtronic_plots()

    def monitor_vttcl_combobox_changed(self):
        print(self.monitor_vttcl_combobox.currentText())
        self.update_medtronic_plots()

    def vf_rate_trigger(self):
        rhythm_deque = self.icd_memory['rhythm_label']
        fs_count = sum(1 for rhythm in self.icd_memory['rhythm_label'] if rhythm == 'FS')
        if len(self.icd_memory['rhythm_label']) == self.icd_mdt_parameters['vf_max_nid'] and fs_count == self.icd_mdt_parameters['vf_min_nid']:
            if self._fd_triggered:
                pass
            else:
                print("VF Rate Triggered")
                self._fd_triggered = True
                self.marker_label = 'FD'
                self.icd_memory['rhythm_label'].append('FD')
                self.vf_rate_trigger_lbl.show()

                if self.indication.currentText() == "AMTest":
                    self.calc_results_beats(begin=self.m_begin, end=self.m_end)
                    self.laser1_vf = float(round(self.laser_exp.results['Laser1_Magic'], 4))
                    self.laser1_conf_vf = float(round(self.laser_exp.results['Laser1_Conf'], 4))
                    self.laser2_vf = float(round(self.laser_exp.results['Laser2_Magic'], 4))
                    self.laser2_conf_vf = float(round(self.laser_exp.results['Laser2_Conf'], 4))
                    print("laser1_val: ", self.laser1_vf)
                    print("laser1_conf: ", self.laser1_conf_vf)
                    print("laser2_val: ", self.laser2_vf)
                    print("laser2_conf: ", self.laser2_conf_vf)

                    if self.laser1_conf_vf >= self.laser2_conf_vf:
                        laser_val_vf = self.laser1_vf
                        laser_conf_vf = self.laser1_conf_vf

                    else:
                        laser_val_vf = self.laser2_vf
                        laser_conf_vf = self.laser2_conf_vf

                    if laser_val_vf > 5.2 and laser_conf_vf > 25:
                        print("Haemodynamically Stable")
                        self.haemStable = HaemStable()

                        self._haemStable = True
                        self.haemStable.show()

                        if self._haem_comp:
                            self.haem_unstable_lbl.hide()

                    if laser_val_vf <= 5.2:
                        print("Haemodynamically Unstable")
                        self._haem_comp = True
                        self.haem_unstable_lbl.show()

                        if self.haem_comp_alert_viewed:
                            pass
                        else:
                            self.haem_unstable_lbl.show()
                            self.haem_comp_alert = HaemCompromise()
                            self.haem_comp_alert.show()
                            self.haem_comp_alert_viewed = True

    def fvt_vfz_rate(self):
        rhythm_deque = self.icd_memory['rhythm_label']

        ts_count = sum(1 for rhythm in self.icd_memory['rhythm_label'] if rhythm == 'TS')
        if len(self.icd_memory['rhythm_label']) == self.icd_mdt_parameters['vf_max_nid'] and ts_count == self.icd_mdt_parameters['vf_min_nid']:
            if self._tf_triggered:
                pass
            else:
                print("FVT Rate Triggered via VF Zone")
                self._tf_triggered = True
                self.marker_label = 'TF'
                self.icd_memory['rhythm_label'].append('FD')
                self.fvt_rate_trigger_lbl.show()

                if self.indication.currentText() == "AMTest":
                    self.calc_results_beats(begin=self.m_begin, end=self.m_end)
                    self.laser1_vf = float(round(self.laser_exp.results['Laser1_Magic'], 4))
                    self.laser1_conf_vf = float(round(self.laser_exp.results['Laser1_Conf'], 4))
                    self.laser2_vf = float(round(self.laser_exp.results['Laser2_Magic'], 4))
                    self.laser2_conf_vf = float(round(self.laser_exp.results['Laser2_Conf'], 4))
                    print("laser1_val: ", self.laser1_vf)
                    print("laser1_conf: ", self.laser1_conf_vf)
                    print("laser2_val: ", self.laser2_vf)
                    print("laser2_conf: ", self.laser2_conf_vf)

                    if self.laser1_conf_vf >= self.laser2_conf_vf:
                        laser_val_vf = self.laser1_vf
                        laser_conf_vf = self.laser1_conf_vf

                    else:
                        laser_val_vf = self.laser2_vf
                        laser_conf_vf = self.laser2_conf_vf

                    if laser_val_vf > 5.2 and laser_conf_vf > 25:
                        print("Haemodynamically Stable")
                        self.haemStable = HaemStable()

                        self._haemStable = True
                        self.haemStable.show()

                        if self._haem_comp:
                            self.haem_unstable_lbl.hide()

                    if laser_val_vf <= 5.2:
                        print("Haemodynamically Unstable")
                        self._haem_comp = True
                        self.haem_unstable_lbl.show()

                        if self.haem_comp_alert_viewed:
                            pass
                        else:
                            self.haem_unstable_lbl.show()
                            self.haem_comp_alert = HaemCompromise()
                            self.haem_comp_alert.show()
                            self.haem_comp_alert_viewed = True


    #
    #
    # def count_item_frequency(self, rhythm_deque):
    #     counter = Counter(rhythm_deque)
    #     mode_frequency = counter.most_common(1)[0]
    #     self.mode_item = mode_frequency[0]
    #     self.mode_frequency = mode_frequency[1]

    def fvt_vtz_trigger(self):
        if self.vt_counter == (int(self.icd_mdt_parameters['vt_nid'] ) + 1) and self.icd_mdt_parameters['vt_nid']  != 0 and self.check_consecutive_vt(self.icd_memory['rhythm_label'], self.icd_mdt_parameters['vt_nid'] ) == True:
            if self._fvt_vtz_triggered:
                pass
            else:
                self.marker_label = 'TF'
                self.icd_memory['rhythm_label'].append(self.marker_label)
                print("FVT Rate Triggered")

                self.fvt_rate_vt_trigger_lbl.show()
                self.atp_delivery_lbl.show()
                self.charging_lbl.show()
                self._fvt_vtz_triggered = True

    def check_consecutive_vt(self, rhythm_deque, vt_nid):
        count = 0
        for string in rhythm_deque:
            if 'TS' in string:
                count += 1
            if count == vt_nid:
                return True
            else:
                count = 0

        return False
    def vt_rate_trigger(self):
        check = self.check_consecutive_vt(self.icd_memory['rhythm_label'], self.icd_mdt_parameters['vt_nid'])
        if self.vt_counter == (int(self.icd_mdt_parameters['vt_nid'] ) + 1) and self.icd_mdt_parameters['vt_nid']  != 0 and check==True:
            if self._vt_rate_triggered:
                pass
            else:
                self.marker_label = 'TD'
                self.icd_memory['rhythm_label'].append(self.marker_label)
                print("FVT Rate Triggered via VT Zone")

                self.vt_rate_trigger_lbl.show()
                self.atp_delivery_lbl.show()
                # self.charging_lbl.show()
                self._vt_rate_triggered = True


    def detect_monitor_vt_combobox_changed(self):
        print(self.detect_monitor_vt_combobox.currentText())

        self.update_medtronic_plots()

    def af_fl_combobox_changed(self):
        print(self.af_fl_combobox.currentText())

    def sinustach_combobox_changed(self):
        print(self.sinustach_combobox.currentText())

    def other_svts_combobox_changed(self):
        print(self.other_svts_combobox.currentText())

    def other_enhanced_svts_combobox_changed(self):
        print(self.other_enhanced_svts_combobox.currentText())

    def stability_combobox_changed(self):
        print(self.stability_combobox.currentText())
        self.icd_mdt_parameters['stability']  = int(self.stability_combobox.currentText())

    def onset_combobox_changed(self):
        print(self.onset_combobox.currentText())
        if self.onset_combobox.currentText() != 'OFF':
            self.onset_pct_combobox.show()
        else:
            self.onset_pct_combobox.hide()

    def svt_v_limit_combobox_changed(self):
        print(self.svt_v_limit_combobox.currentText())

    def wavelet_combobox_changed(self):
        print(self.wavelet_combobox.currentText())
        if self.wavelet_combobox.currentText() == 'ON':
            self.autocollection_lbl.show()
            self.wavelet_autocollection_combobox.show()
            self.match_threshold_lbl.show()
            self.wavelet_match_threshold_combobox.show()
        else:
            self.autocollection_lbl.hide()
            self.match_threshold_lbl.hide()
            self.wavelet_autocollection_combobox.hide()
            self.wavelet_match_threshold_combobox.hide()

    def wavelet_autocollection_combobox_changed(self):
        self.wavelet_autocollection = self.wavelet_autocollection_combobox.currentText()
        print(self.wavelet_autocollection)

    def wavelet_match_threshold_combobox_changed(self):
        print(self.wavelet_match_threshold_combobox.currentText())

        self.match_threshold = int(self.wavelet_match_threshold_combobox.currentText())

    def onset_pct_combobox_changed(self):
        value = int(self.onset_pct_combobox.currentText())
        print(value)
        self.icd_mdt_parameters['onset_pct']  = float(value / 100)

    def highrate_vftimeout_combobox_changed(self):
        print(self.high_rate_vftimeout_combobox.currentText())
        if self.high_rate_vftimeout_combobox.currentText() != 'OFF':
            self.high_rate_allzone_timeout_combobox.show()
            self.update_medtronic_plots()
        else:
            self.high_rate_allzone_timeout_combobox.hide()
            self.update_medtronic_plots()

    def highrate_allzone_timeout_combobox_changed(self):
        print(self.high_rate_allzone_timeout_combobox.currentText())

        self.update_medtronic_plots()

    def twave_combobox_changed(self):
        print(self.twave_combobox.currentText())

    def rv_lead_noise_combobox_changed(self):
        print(self.rv_lead_noise_combobox.currentText())

    def rv_lead_noise_timeout_combobox_changed(self):
        print(self.rv_lead_noise_timeout_combobox.currentText())


class HaemStable(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtCore.Qt.white)
        p.setColor(self.foregroundRole(), QtCore.Qt.black)
        self.setWindowTitle("HAEMODYNAMIC ASSESSMENT")

        self.message = QtWidgets.QLabel()
        icon_path = "/Users/amiyazawa/Library/CloudStorage/Dropbox/01 Projects/Medtronic Algorithm/Medtronic Algorithm Python Code/icons/checkbox.png"
        self.tick_icon = QtGui.QPixmap(icon_path)
        icon_html = f'<img src="{icon_path}" width="{self.tick_icon.width()}" height="{self.tick_icon.height()}">'
        text = "&nbsp;&nbsp;Haemodynamically Stable: <br>Shock Withheld"
        self.message.setText(icon_html + text)
        self.message.setAlignment(QtCore.Qt.AlignCenter)
        self.message.setFont(QtGui.QFont("Arial", 30, QtGui.QFont.Bold))
        self.message.setWordWrap(True)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.message)
        self.setLayout(self.layout)