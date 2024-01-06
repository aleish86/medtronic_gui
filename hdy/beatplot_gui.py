# Beatplot GUI and Plots
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

class BeatPlot_GUI(QtWidgets.QWidget):
    def __init__(self, laser_exp: object, parent: object = None)-> object:
        super().__init__(parent=parent)
        self.laser_exp = laser_exp
        self.icd_mdt_parameters = {}
        self.icd_mdt_parameters['rvst_value'] = 0.3
        self.icd_mdt_parameters['pvsb_value'] = 120
        self.index = 0

        self.beatplot_gui()
        self.setup_beat_plots()

    def beatplot_gui(self):
        pg.setConfigOptions(antialias=True, background='w')

        # Creating Layouts
        self.m_beatplot_layout1 = QtWidgets.QHBoxLayout()
        self.m_beatplot_layout2 = QtWidgets.QHBoxLayout()
        self.m_beatplot_layout3 = QtWidgets.QVBoxLayout()
        self.m_beatplot_overall_layout = QtWidgets.QVBoxLayout()
        self.m_beatplot_btn_layout = QtWidgets.QVBoxLayout()
        self.m_beatplot_main_layout = QtWidgets.QHBoxLayout()

        # Creating Viewing Toggles
        self.start_btn = QtWidgets.QPushButton("Start")
        self.start_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.start_btn.clicked.connect(self.start_btn_clicked)

        self.pause_btn = QtWidgets.QPushButton("Pause")
        self.pause_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.pause_btn.clicked.connect(self.pause_btn_clicked)

        self.ralead_toggle_beatplot = QtWidgets.QCheckBox("View RA Lead")
        self.ralead_toggle_beatplot.setChecked(True)
        self.ralead_toggle_beatplot.stateChanged.connect(self.ralead_toggle_changed)

        self.rvbip_toggle_beatplot = QtWidgets.QCheckBox("View RV Bipolar")
        self.rvbip_toggle_beatplot.setChecked(True)
        self.rvbip_toggle_beatplot.stateChanged.connect(self.rvbip_toggle_changed)

        self.rvshock_toggle_beatplot = QtWidgets.QCheckBox("View RV Shock")
        self.rvshock_toggle_beatplot.setChecked(True)
        self.rvshock_toggle_beatplot.stateChanged.connect(self.rvshock_toggle_changed)

        self.lvlead_toggle_beatplot = QtWidgets.QCheckBox("View LV Lead")
        self.lvlead_toggle_beatplot.setChecked(False)
        self.lvlead_toggle_beatplot.stateChanged.connect(self.lvlead_toggle_changed)

        self.bipecg_toggle_beatplot = QtWidgets.QCheckBox("View Bipolar ECG")
        self.bipecg_toggle_beatplot.setChecked(True)
        self.bipecg_toggle_beatplot.stateChanged.connect(self.bipecg_toggle_changed)

        self.laser1_toggle_beatplot = QtWidgets.QCheckBox("View Laser1")
        self.laser1_toggle_beatplot.setChecked(True)
        self.laser1_toggle_beatplot.stateChanged.connect(self.laser1_toggle_changed)

        self.laser2_toggle_beatplot = QtWidgets.QCheckBox("View Laser2")
        self.laser2_toggle_beatplot.setChecked(False)
        self.laser2_toggle_beatplot.stateChanged.connect(self.laser2_toggle_changed)

        # RV Sensitivity Threshold
        self.rvst1 = QtWidgets.QRadioButton("0.15")
        self.rvst2 = QtWidgets.QRadioButton("0.3")
        self.rvst2.setChecked(True)
        self.rvst3 = QtWidgets.QRadioButton("0.45")
        self.rvst4 = QtWidgets.QRadioButton("0.6")
        self.rvst5 = QtWidgets.QRadioButton("0.9")
        self.rvst6 = QtWidgets.QRadioButton("1.2")

        self.rvst_group = QtWidgets.QButtonGroup()
        self.rvst_group.addButton(self.rvst1, 1)
        self.rvst_group.addButton(self.rvst2, 2)
        self.rvst_group.addButton(self.rvst3, 3)
        self.rvst_group.addButton(self.rvst4, 4)
        self.rvst_group.addButton(self.rvst5, 5)
        self.rvst_group.addButton(self.rvst6, 6)
        self.rvst_group.buttonClicked.connect(self.rv_sens_threshold_changed)

        self.rvst_groupbox = QtWidgets.QGroupBox("RV Sensitivity Threshold")
        self.rvst_groupbox.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.rvst_vbox = QtWidgets.QVBoxLayout()
        self.rvst_vbox.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.rvst_groupbox.setLayout(self.rvst_vbox)
        self.rvst_hbox1 = QtWidgets.QHBoxLayout()
        self.rvst_hbox1.addWidget(self.rvst1)
        self.rvst_hbox1.addWidget(self.rvst2)
        self.rvst_hbox1.addWidget(self.rvst3)
        self.rvst_hbox2 = QtWidgets.QHBoxLayout()
        self.rvst_hbox2.addWidget(self.rvst4)
        self.rvst_hbox2.addWidget(self.rvst5)
        self.rvst_hbox2.addWidget(self.rvst6)
        self.rvst_vbox.addLayout(self.rvst_hbox1)
        self.rvst_vbox.addLayout(self.rvst_hbox2)
        self.m_beatplot_btn_layout.addWidget(self.rvst_groupbox)

        # Adding Toggles to Layout
        self.m_beatplot_btn_layout.addWidget(self.start_btn, 0)
        self.m_beatplot_btn_layout.addWidget(self.pause_btn, 0)
        self.m_beatplot_btn_layout.addWidget(self.ralead_toggle_beatplot, 0)
        self.m_beatplot_btn_layout.addWidget(self.rvbip_toggle_beatplot, 0)
        self.m_beatplot_btn_layout.addWidget(self.rvshock_toggle_beatplot, 0)
        self.m_beatplot_btn_layout.addWidget(self.lvlead_toggle_beatplot, 0)
        self.m_beatplot_btn_layout.addWidget(self.bipecg_toggle_beatplot, 0)
        self.m_beatplot_btn_layout.addWidget(self.laser1_toggle_beatplot, 0)
        self.m_beatplot_btn_layout.addWidget(self.laser2_toggle_beatplot, 0)

        # PLOT LAYOUTS
        self.ralead_beat_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.ralead_beat_pi = self.ralead_beat_pw.getPlotItem()
        self.ralead_beat_pi.setTitle("RA Lead", color="k", size="16pt")
        self.ralead_beat_pi.setLabel(axis='left')
        self.ralead_beat_plt = self.ralead_beat_pi.plot()

        self.bipecg_beat_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.bipecg_beat_pi = self.bipecg_beat_pw.getPlotItem()
        self.bipecg_beat_pi.setTitle("Bipolar ECG", color='k', size='16pt')
        self.bipecg_beat_pi.setLabel(axis='left')
        self.bipecg_beat_plt = self.bipecg_beat_pi.plot()
        self.bipecg_peak_beat_plt = self.bipecg_beat_pi.plot()

        self.rvbip_beat_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.rvbip_beat_pi = self.rvbip_beat_pw.getPlotItem()
        self.rvbip_beat_pi.setTitle("RV Bipolar", color='k', size='16pt')
        self.rvbip_beat_pi.setLabel(axis='left')
        self.rvbip_beat_plt = self.rvbip_beat_pi.plot()

        self.rvshock_beat_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.rvshock_beat_pi = self.rvshock_beat_pw.getPlotItem()
        self.rvshock_beat_pi.setTitle("RV Shock", color='k', size='16pt')
        self.rvshock_beat_pi.setLabel(axis='left')
        self.rvshock_beat_plt = self.rvshock_beat_pi.plot()

        self.lvlead_beat_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.lvlead_beat_pi = self.lvlead_beat_pw.getPlotItem()
        self.lvlead_beat_pi.setTitle("LV Lead", color='k', size='16pt')
        self.lvlead_beat_pi.setLabel(axis='left')
        self.lvlead_beat_plt = self.lvlead_beat_pi.plot()
        self.lvlead_beat_pw.hide()

        self.laser1_beat_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.laser1_beat_pi = self.laser1_beat_pw.getPlotItem()
        self.laser1_beat_pi.setLabel(axis='left')
        self.laser1_beat_pi.setTitle("Laser1", color='k', size='16pt')
        self.laser1_beat_plt = self.laser1_beat_pi.plot()
        self.envelope1_beat_plt = self.laser1_beat_pi.plot()

        self.laser2_beat_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.laser2_beat_pi = self.laser2_beat_pw.getPlotItem()
        self.laser2_beat_pi.setLabel(axis='left')
        self.laser2_beat_pi.setTitle("Laser2", color='k', size='16pt')
        self.laser2_beat_plt = self.laser2_beat_pi.plot()
        self.laser2_beat_pw.hide()

        self.envelope2_beat_plt = self.laser2_beat_pi.plot()

        self.overview_beat_pw = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.overview_beat_pi = self.overview_beat_pw.getPlotItem()
        self.overview_beat_pi.setLabel(axis='left', text="Overview")
        self.overview_beat_plt = self.overview_beat_pi.plot()

        #     Adding plot widgets to layout
        self.m_beatplot_layout1.addWidget(self.ralead_beat_pw, stretch=1)
        self.m_beatplot_layout1.addWidget(self.rvbip_beat_pw, stretch=1)
        self.m_beatplot_layout1.addWidget(self.rvshock_beat_pw, stretch=1)
        self.m_beatplot_layout2.addWidget(self.lvlead_beat_pw, stretch=1)
        self.m_beatplot_layout2.addWidget(self.bipecg_beat_pw, stretch=1)
        self.m_beatplot_layout2.addWidget(self.laser1_beat_pw, stretch=1)
        self.m_beatplot_layout2.addWidget(self.laser2_beat_pw, stretch=1)
        self.m_beatplot_layout3.addWidget(self.overview_beat_pw, stretch=1)
        self.zipfl_lbl1 = QtWidgets.QLabel()
        self.m_beatplot_layout3.addWidget(self.zipfl_lbl1)
        self.zipfl_lbl1.setText(str(self.laser_exp.zip_fl))

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_beatplots)

        # self.timer.start(1000)  # 1000 ms = 1 s
        # QtWidgets.QApplication.processEvents()

        # Combining layouts
        self.m_beatplot_overall_layout.addLayout(self.m_beatplot_layout1, stretch=1)
        self.m_beatplot_overall_layout.addLayout(self.m_beatplot_layout2, stretch=1)
        self.m_beatplot_overall_layout.addLayout(self.m_beatplot_layout3, stretch=1)
        self.m_beatplot_main_layout.addLayout(self.m_beatplot_btn_layout, 0)
        self.m_beatplot_main_layout.addLayout(self.m_beatplot_overall_layout)
        # Fixing Layouts
        self.m_beatplot_btn_layout.setSpacing(0)
        self.m_beatplot_btn_layout.setContentsMargins(0, 0, 0, 750)
        self.m_beatplot_overall_layout.setSpacing(0)
        self.m_beatplot_overall_layout.setContentsMargins(0, 0, 10, 190)  # Left, top, right, bottom
        self.m_beatplot_main_layout.setSpacing(0)
        self.m_beatplot_main_layout.setContentsMargins(2, 2, 2, 2)

        # self.max_x_list, self.rr_list, self.amplitude_list, self.rpeaks = data_list(self.laser_exp.rvbip.data,
        #                                                                self.icd_mdt_parameters['rvst_value'],
        #                                                                self.icd_mdt_parameters['pvsb_value'])
        self.max_x_list, self.rr_list = data_list(self.laser_exp.rvbip.data,
                                                                       self.icd_mdt_parameters['rvst_value'],
                                                                       self.icd_mdt_parameters['pvsb_value'])

        self.setLayout(self.m_beatplot_main_layout)

    def setup_beat_plots(self):

        self.ralead_beat_pi.getAxis('left').setWidth(w=40)
        self.ralead_beat_pi.getAxis('left').setStyle(showValues=False)
        self.rvbip_beat_pi.getAxis('left').setWidth(w=40)
        self.rvbip_beat_pi.getAxis('left').setStyle(showValues=False)
        self.rvshock_beat_pi.getAxis('left').setWidth(w=40)
        self.rvshock_beat_pi.getAxis('left').setStyle(showValues=False)
        self.lvlead_beat_pi.getAxis('left').setWidth(w=40)
        self.lvlead_beat_pi.getAxis('left').setStyle(showValues=False)
        self.bipecg_beat_pi.getAxis('left').setWidth(w=40)
        self.bipecg_beat_pi.getAxis('left').setStyle(showValues=False)
        self.laser1_beat_pi.getAxis('left').setWidth(w=40)
        self.laser2_beat_pi.getAxis('left').setWidth(w=40)
        self.overview_beat_pi.getAxis('left').setWidth(w=50)
        self.overview_beat_pi.getAxis('left').setStyle(showValues=False)


    def update_beatplots(self):
        pg.setConfigOptions(antialias=True, background='w')
        samples = self.laser_exp.pressure.data.shape[0]
        ecg_hint = self.laser_exp.hints['Period']
        print(ecg_hint)
        print(len(self.max_x_list))
        print(len(self.rr_list))

        max_pt_beat = self.max_x_list[self.index]

        print(max_pt_beat)

        # print('max_x_list: ', self.max_x_list)
        # Perform the necessary updates with the item
        rr_interval = self.rr_list[self.index]
        # rr_interval = self.rr_list[self.index]
        print(rr_interval)
        half_range = int(rr_interval / 2)
        beat_start = max_pt_beat - half_range
        beat_end = max_pt_beat + half_range
        beat_range = np.arange(beat_start, beat_end)

        if beat_start < 0:
            beat_start = 0
        if beat_end > samples:
            beat_end = samples

        self.ralead_data = self.laser_exp.ralead.data[beat_start:beat_end]
        self.rvbip_data = self.laser_exp.rvbip.data[beat_start:beat_end]
        self.rvshock_data = self.laser_exp.rvshock.data[beat_start:beat_end]
        self.lvlead_data = self.laser_exp.lvlead.data[beat_start:beat_end]
        self.bipecg_data = self.laser_exp.bipecg.data[beat_start:beat_end]
        self.laser1_data = self.laser_exp.laser1.data[beat_start:beat_end]
        self.laser2_data = self.laser_exp.laser2.data[beat_start:beat_end]

        self.ralead_beat_plt.setData(x=beat_range, y=self.ralead_data, pen='#732F9B', symbol=None, antialise=True,
                                     autoDownsample=True, clipToView=True)
        self.rvbip_beat_plt.setData(x=beat_range, y=self.rvbip_data, pen='#732F9B', symbol=None, antialise=True,
                                    autoDownsample=True, clipToView=True)
        self.rvshock_beat_plt.setData(x=beat_range, y=self.rvshock_data, pen='#732F9B', symbol=None, antialise=True,
                                      autoDownsample=True, clipToView=True)
        self.lvlead_beat_plt.setData(x=beat_range, y=self.lvlead_data, pen='#732F9B', symbol=None, antialise=True,
                                     autoDownsample=True, clipToView=True)
        self.bipecg_beat_plt.setData(x=beat_range, y=self.bipecg_data, pen='#0FA00F', symbol=None, antialise=True,
                                     autoDownsample=True, clipToView=True)
        self.laser1_beat_plt.setData(x=beat_range, y=self.laser1_data, pen='#03c6fc', symbol=None, antialise=True,
                                     autoDownsample=True, clipToView=True)
        self.laser2_beat_plt.setData(x=beat_range, y=self.laser2_data, pen='#03c6fc', symbol=None, antialise=True,
                                     autoDownsample=True, clipToView=True)
        self.overview_beat_plt.setData(x=np.arange(samples), y=self.laser_exp.rvbip.data, pen='#0FA00F', symbol=None,
                                       antialise=True, autoDownsample=True, clipToView=True)
        self.overview_beat_pw.addItem(pg.InfiniteLine(pos=max_pt_beat, angle=90, pen='r', markers='o'))
        # self.overview_beat_pw.addItem(pg.InfiniteLine.addMarker(position=max_pt_beat, marker= 'o'))

        if self.index >= len(self.max_x_list):
            self.timer.stop()

        self.index += 1

    # Media Buttons
    def start_btn_clicked(self):
        self.timer.start(1000)

    def pause_btn_clicked(self):
        self.timer.stop()


    def rv_sens_threshold_changed(self, object):
        self.value = self.rvst_group.id(object)

        print("RV Sensing Threshold: " + str(self.value))
        self.max_x_list, self.rr_list, self.amplitude_list = data_list(self.laser_exp.rvbip.data,
                                                                           self.icd_mdt_parameters['rvst_value'],
                                                                           self.icd_mdt_parameters['pvsb_value'])

        self.update_beatplots()

    def update_rvst(self):

        self.icd_mdt_parameters['rvst_value'] = float(self.rvst_group.checkedButton().text())

        for button in self.rvst_group.buttons():
            if button.isChecked() == True:
                button.setStyleSheet("font-weight: bold")
                print('button checked: ', button.text())

            else:

                button.setStyleSheet("font-weight: normal")
        self.update_beatplots()

        print('rvst value: ', self.icd_mdt_parameters['rvst_value'])

    # Toggle Changes
    def laser1_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.laser1_beat_pw.show()

        else:
            self.laser1_beat_pw.hide()


    def laser2_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.laser2_beat_pw.show()
        else:
            self.laser2_beat_pw.hide()

    def bipecg_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.bipecg_beat_pw.show()

        else:
            self.bipecg_beat_pw.hide()


    def rvshock_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.rvshock_beat_pw.show()

        else:
            self.rvshock_beat_pw.hide()

    def rvbip_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.rvbip_beat_pw.show()

        else:
            self.rvbip_beat_pw.hide()

    def lvlead_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.lvlead_beat_pw.show()

        else:
            self.lvlead_beat_pw.hide()

    def ralead_toggle_changed(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.ralead_beat_pw.show()

        else:
            self.ralead_beat_pw.hide()

    def overview_region_update(self, window, viewRange):
        rgn = viewRange[0]
        self.overview_region.setRegion(rgn)