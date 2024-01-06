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
import pywt

class vtherapies_GUI(QtWidgets.QWidget):
    def __init__(self, laser_exp: object, parent: object = None)-> object:
        super().__init__(parent=parent)
        self.laser_exp = laser_exp

        self.icd_mdt_parameters = {}
        self.icd_mdt_parameters['vf_rx1_status'] = "ON"

        self.vtherapies_gui()


    def vtherapies_gui(self):

        vf_therapies_lbl = QtWidgets.QLabel("VF Therapies")
        vf_therapies_lbl.setFont(QtGui.QFont("Arial", 20, QtGui.QFont.Bold))
        vf_therapies_lbl.setFixedWidth(200)
        vf_therapies_lbl.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        self.vf_labels()
        self.vf_comboboxes()
        self.vf_tx_gridlayout()
        self.vf_atp_lbls()
        self.vf_atp_comboboxes()
        self.vf_atp_gridlayout()

        vf_vbox = QtWidgets.QVBoxLayout()
        vf_vbox.addWidget(vf_therapies_lbl)
        vf_vbox.addLayout(self.vf_tx_grid)
        vf_vbox.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        vf_vbox.setSpacing(0)
        vf_vbox.setContentsMargins(0, 0, 0, 0)
        atp_vbox = QtWidgets.QVBoxLayout()
        atp_vbox.addWidget(self.vf_atp_lbl)
        atp_vbox.addLayout(self.vf_atp_grid)
        atp_vbox.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        atp_vbox.setSpacing(0)
        atp_vbox.setContentsMargins(0, 0, 0, 0)
        self.vf_tx_grid.setSpacing(0)
        self.vf_tx_grid.setContentsMargins(0, 0, 0, 0)
        self.vf_atp_grid.setSpacing(0)
        self.vf_atp_grid.setContentsMargins(0, 0, 0, 0)

        tx_vbox = QtWidgets.QVBoxLayout()
        tx_vbox.setSpacing(0)
        tx_vbox.setContentsMargins(0, 0, 0, 0)
        tx_vbox.addLayout(vf_vbox)
        tx_vbox.addLayout(atp_vbox)

        self.setLayout(tx_vbox)

    def vf_tx_gridlayout(self):
        self.vf_tx_grid = QtWidgets.QGridLayout()
        self.vf_tx_grid.addWidget(self.rx1_lbl, 0, 1)
        self.vf_tx_grid.addWidget(self.rx2_lbl, 0, 2)
        self.vf_tx_grid.addWidget(self.rx3_lbl, 0, 3)
        self.vf_tx_grid.addWidget(self.rx4_lbl, 0, 4)
        self.vf_tx_grid.addWidget(self.rx5_lbl, 0, 5)
        self.vf_tx_grid.addWidget(self.rx6_lbl, 0, 6)
        self.vf_tx_grid.addWidget(self.energy_lbl, 1, 0)
        self.vf_tx_grid.addWidget(self.vf_rx1_energy, 1, 1)
        self.vf_tx_grid.addWidget(self.vf_rx2_energy, 1, 2)
        self.vf_tx_grid.addWidget(self.vf_rx3_energy, 1, 3)
        self.vf_tx_grid.addWidget(self.vf_rx4_energy, 1, 4)
        self.vf_tx_grid.addWidget(self.vf_rx5_energy, 1, 5)
        self.vf_tx_grid.addWidget(self.vf_rx6_energy, 1, 6)
        self.vf_tx_grid.addWidget(self.pathway_lbl, 2, 0)
        self.vf_tx_grid.addWidget(self.vf_rx1_pathway, 2, 1)
        self.vf_tx_grid.addWidget(self.vf_rx2_pathway, 2, 2)
        self.vf_tx_grid.addWidget(self.vf_rx3_pathway, 2, 3)
        self.vf_tx_grid.addWidget(self.vf_rx4_pathway, 2, 4)
        self.vf_tx_grid.addWidget(self.vf_rx5_pathway, 2, 5)
        self.vf_tx_grid.addWidget(self.vf_rx6_pathway, 2, 6)

    def vf_atp_gridlayout(self):
        self.vf_atp_grid = QtWidgets.QGridLayout()

        self.vf_atp_grid.addWidget(self.vf_atp_lbl, 0, 0)
        self.vf_atp_grid.addWidget(self.vf_atp_rx_status_lbl, 1, 0)
        self.vf_atp_grid.addWidget(self.vf_atp_combobox, 1, 1)
        self.vf_atp_grid.addWidget(self.vf_atp_type_lbl, 2, 0)
        self.vf_atp_grid.addWidget(self.vf_atp_type, 2, 1)
        self.vf_atp_grid.addWidget(self.vf_atp_del_8rr_lbl, 3, 0)
        self.vf_atp_grid.addWidget(self.vf_atp_del_8rr, 3, 1)
        self.vf_atp_grid.addWidget(self.vf_atp_num_seq_before_lbl, 4, 0)
        self.vf_atp_grid.addWidget(self.vf_atp_num_seq_before, 4, 1)
        self.vf_atp_grid.addWidget(self.vf_atp_num_seq_during_lbl, 5, 0)
        self.vf_atp_grid.addWidget(self.vf_atp_num_seq_during, 5, 1)
        self.vf_atp_grid.addWidget(self.vf_atp_num_pulses_lbl, 6, 0)
        self.vf_atp_grid.addWidget(self.vf_atp_num_pulses, 6, 1)
        self.vf_atp_grid.addWidget(self.vf_atp_rs1_lbl, 7, 0)
        self.vf_atp_grid.addWidget(self.vf_atp_rs1, 7, 1)

    def vf_labels(self):

        self.rx1_lbl = QtWidgets.QLabel("Rx1")
        self.rx1_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))
        self.rx1_lbl.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self.rx1_lbl.setFixedWidth(150)

        self.rx2_lbl = QtWidgets.QLabel("Rx2")
        self.rx2_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))
        self.rx2_lbl.setFixedWidth(150)
        self.rx2_lbl.setAlignment(QtCore.Qt.AlignLeft| QtCore.Qt.AlignTop)

        self.rx3_lbl = QtWidgets.QLabel("Rx3")
        self.rx3_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))
        self.rx3_lbl.setFixedWidth(150)
        self.rx3_lbl.setAlignment(QtCore.Qt.AlignLeft| QtCore.Qt.AlignTop)

        self.rx4_lbl = QtWidgets.QLabel("Rx4")
        self.rx4_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))
        self.rx4_lbl.setFixedWidth(150)
        self.rx4_lbl.setAlignment(QtCore.Qt.AlignLeft| QtCore.Qt.AlignTop)

        self.rx5_lbl = QtWidgets.QLabel("Rx5")
        self.rx5_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))
        self.rx5_lbl.setFixedWidth(150)
        self.rx5_lbl.setAlignment(QtCore.Qt.AlignLeft)

        self.rx6_lbl = QtWidgets.QLabel("Rx6")
        self.rx6_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))
        self.rx6_lbl.setFixedWidth(150)
        self.rx6_lbl.setAlignment(QtCore.Qt.AlignLeft| QtCore.Qt.AlignTop)

        self.vf_rx_status_lbl = QtWidgets.QLabel("VF Therapy Status")
        self.vf_rx_status_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))
        self.vf_rx_status_lbl.setFixedWidth(150)
        self.vf_rx_status_lbl.setAlignment(QtCore.Qt.AlignLeft| QtCore.Qt.AlignTop)

        self.energy_lbl = QtWidgets.QLabel("Energy")
        self.energy_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))
        self.energy_lbl.setFixedWidth(150)
        self.energy_lbl.setAlignment(QtCore.Qt.AlignLeft| QtCore.Qt.AlignTop)

        self.pathway_lbl = QtWidgets.QLabel("Pathway")
        self.pathway_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))
        self.pathway_lbl.setFixedWidth(150)
        self.pathway_lbl.setMinimumSize(QtCore.QSize(0, 0))
        self.pathway_lbl.setAlignment(QtCore.Qt.AlignLeft| QtCore.Qt.AlignTop)

        self.atp_lbl = QtWidgets.QLabel("ATP...")
        self.atp_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))
        self.atp_lbl.setFixedWidth(200)
        self.atp_lbl.setAlignment(QtCore.Qt.AlignLeft)

    def vf_comboboxes(self):
        # VF Rx Comboboxes
        self.vf_rx1_status = QtWidgets.QComboBox()
        self.vf_rx1_status.addItems(["ON", "OFF"])
        self.vf_rx1_status.setCurrentIndex(0)
        self.vf_rx1_status.setFixedWidth(100)
        self.vf_rx1_status.currentIndexChanged.connect(self.vf_rx1_status_changed)

        self.vf_rx2_status = QtWidgets.QComboBox()
        self.vf_rx2_status.addItems(["ON", "OFF"])
        self.vf_rx2_status.setCurrentIndex(0)
        self.vf_rx2_status.setFixedWidth(100)
        self.vf_rx2_status.currentIndexChanged.connect(self.vf_rx2_status_changed)

        self.vf_rx3_status = QtWidgets.QComboBox()
        self.vf_rx3_status.addItems(["ON", "OFF"])
        self.vf_rx3_status.setCurrentIndex(0)
        self.vf_rx3_status.setFixedWidth(100)
        self.vf_rx3_status.currentIndexChanged.connect(self.vf_rx3_status_changed)

        self.vf_rx4_status = QtWidgets.QComboBox()
        self.vf_rx4_status.addItems(["ON", "OFF"])
        self.vf_rx4_status.setCurrentIndex(0)
        self.vf_rx4_status.setFixedWidth(100)
        self.vf_rx4_status.currentIndexChanged.connect(self.vf_rx4_status_changed)

        self.vf_rx5_status = QtWidgets.QComboBox()
        self.vf_rx5_status.addItems(["ON", "OFF"])
        self.vf_rx5_status.setCurrentIndex(0)
        self.vf_rx5_status.setFixedWidth(100)
        self.vf_rx5_status.currentIndexChanged.connect(self.vf_rx5_status_changed)

        self.vf_rx6_status = QtWidgets.QComboBox()
        self.vf_rx6_status.addItems(["ON", "OFF"])
        self.vf_rx6_status.setCurrentIndex(0)
        self.vf_rx6_status.setFixedWidth(100)
        self.vf_rx6_status.currentIndexChanged.connect(self.vf_rx6_status_changed)

        self.vf_rx1_energy = QtWidgets.QComboBox()
        self.vf_rx1_energy.addItems("0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 18 20 22 24 25 26 28 30 32 35 40".split())
        self.vf_rx1_energy.setCurrentIndex(33)
        self.vf_rx1_energy.setFixedWidth(100)
        self.vf_rx1_energy.currentIndexChanged.connect(self.vf_rx1_energy_changed)

        self.vf_rx2_energy = QtWidgets.QComboBox()
        self.vf_rx2_energy.addItems("0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 18 20 22 24 25 26 28 30 32 35 40".split())
        self.vf_rx2_energy.setCurrentIndex(33)
        self.vf_rx2_energy.setFixedWidth(100)
        self.vf_rx2_energy.currentIndexChanged.connect(self.vf_rx2_energy_changed)

        self.vf_rx3_energy = QtWidgets.QComboBox()
        self.vf_rx3_energy.addItems("10 11 12 13 14 15 16 18 20 22 24 25 26 28 30 32 35 40".split())
        self.vf_rx3_energy.setCurrentIndex(17)
        self.vf_rx3_energy.setFixedWidth(100)
        self.vf_rx3_energy.currentIndexChanged.connect(self.vf_rx3_energy_changed)

        self.vf_rx4_energy = QtWidgets.QComboBox()
        self.vf_rx4_energy.addItems("10 11 12 13 14 15 16 18 20 22 24 25 26 28 30 32 35 40".split())
        self.vf_rx4_energy.setCurrentIndex(17)
        self.vf_rx4_energy.setFixedWidth(100)
        self.vf_rx4_energy.currentIndexChanged.connect(self.vf_rx4_energy_changed)

        self.vf_rx5_energy = QtWidgets.QComboBox()
        self.vf_rx5_energy.addItems("10 11 12 13 14 15 16 18 20 22 24 25 26 28 30 32 35 40".split())
        self.vf_rx5_energy.setCurrentIndex(17)
        self.vf_rx5_energy.setFixedWidth(100)
        self.vf_rx5_energy.currentIndexChanged.connect(self.vf_rx5_energy_changed)

        self.vf_rx6_energy = QtWidgets.QComboBox()
        self.vf_rx6_energy.addItems("10 11 12 13 14 15 16 18 20 22 24 25 26 28 30 32 35 40".split())
        self.vf_rx6_energy.setCurrentIndex(17)
        self.vf_rx6_energy.setFixedWidth(100)
        self.vf_rx6_energy.currentIndexChanged.connect(self.vf_rx6_energy_changed)

        self.vf_rx1_pathway = QtWidgets.QComboBox()
        self.vf_rx1_pathway.addItems(["AX>B", "B>AX"])
        self.vf_rx1_pathway.setCurrentIndex(1)
        self.vf_rx1_pathway.setFixedWidth(100)
        self.vf_rx1_pathway.currentIndexChanged.connect(self.vf_rx1_pathway_changed)

        self.vf_rx2_pathway = QtWidgets.QComboBox()
        self.vf_rx2_pathway.addItems(["AX>B", "B>AX"])
        self.vf_rx2_pathway.setCurrentIndex(1)
        self.vf_rx2_pathway.setFixedWidth(100)
        self.vf_rx2_pathway.currentIndexChanged.connect(self.vf_rx2_pathway_changed)

        self.vf_rx3_pathway = QtWidgets.QComboBox()
        self.vf_rx3_pathway.addItems(["AX>B", "B>AX"])
        self.vf_rx3_pathway.setCurrentIndex(1)
        self.vf_rx3_pathway.setFixedWidth(100)
        self.vf_rx3_pathway.currentIndexChanged.connect(self.vf_rx3_pathway_changed)

        self.vf_rx4_pathway = QtWidgets.QComboBox()
        self.vf_rx4_pathway.addItems(["AX>B", "B>AX"])
        self.vf_rx4_pathway.setFixedWidth(100)
        self.vf_rx4_pathway.setCurrentIndex(1)
        self.vf_rx4_pathway.currentIndexChanged.connect(self.vf_rx4_pathway_changed)

        self.vf_rx5_pathway = QtWidgets.QComboBox()
        self.vf_rx5_pathway.addItems(["AX>B", "B>AX"])
        self.vf_rx5_pathway.setFixedWidth(100)
        self.vf_rx5_pathway.setCurrentIndex(0)
        self.vf_rx5_pathway.currentIndexChanged.connect(self.vf_rx5_pathway_changed)

        self.vf_rx6_pathway = QtWidgets.QComboBox()
        self.vf_rx6_pathway.addItems(["AX>B", "B>AX"])
        self.vf_rx6_pathway.setFixedWidth(100)
        self.vf_rx6_pathway.setCurrentIndex(0)
        self.vf_rx6_pathway.currentIndexChanged.connect(self.vf_rx6_pathway_changed)

    def vf_atp_lbls(self):

        # VF ATP
        self.vf_atp_lbl = QtWidgets.QLabel("VF ATP")
        self.vf_atp_lbl.setFont(QtGui.QFont("Arial", 20, QtGui.QFont.Bold))
        self.vf_atp_lbl.setFixedWidth(150)
        self.vf_atp_lbl.setAlignment(QtCore.Qt.AlignLeft| QtCore.Qt.AlignTop)

        self.vf_atp_rx_status_lbl = QtWidgets.QLabel("Therapy Status")
        self.vf_atp_rx_status_lbl.setAlignment(QtCore.Qt.AlignLeft)
        self.vf_atp_rx_status_lbl.setFixedWidth(150)
        self.vf_atp_rx_status_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))

        self.vf_atp_type_lbl = QtWidgets.QLabel("Therapy Type")
        self.vf_atp_type_lbl.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.vf_atp_type_lbl.setFixedWidth(150)
        self.vf_atp_type_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))

        self.vf_atp_del_8rr_lbl = QtWidgets.QLabel("Deliver ATP if last 8 R-R >=")
        self.vf_atp_del_8rr_lbl.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.vf_atp_del_8rr_lbl.setFixedWidth(200)
        self.vf_atp_del_8rr_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))

        self.vf_atp_num_seq_before_lbl = QtWidgets.QLabel("# Sequences before Charging")
        self.vf_atp_num_seq_before_lbl.setAlignment(QtCore.Qt.AlignLeft)
        self.vf_atp_num_seq_before_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))
        self.vf_atp_num_seq_before_lbl.hide()

        self.vf_atp_num_seq_during_lbl = QtWidgets.QLabel("# Sequences during Charging")
        self.vf_atp_num_seq_during_lbl.setAlignment(QtCore.Qt.AlignLeft)
        self.vf_atp_num_seq_during_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))

        self.vf_atp_num_pulses_lbl = QtWidgets.QLabel("Initial # Pulses")
        self.vf_atp_num_pulses_lbl.setAlignment(QtCore.Qt.AlignLeft)
        self.vf_atp_num_pulses_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))

        self.vf_atp_rs1_lbl = QtWidgets.QLabel("R-S1 Interval=(%RR)")
        self.vf_atp_rs1_lbl.setAlignment(QtCore.Qt.AlignLeft)
        self.vf_atp_rs1_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))

        self.vf_atp_int_dec_lbl = QtWidgets.QLabel("Interval Decrement (ms)")
        self.vf_atp_int_dec_lbl.setAlignment(QtCore.Qt.AlignLeft)
        self.vf_atp_int_dec_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))

        self.vf_atp_chargesaver_lbl = QtWidgets.QLabel("Charge Saver")
        self.vf_atp_chargesaver_lbl.setAlignment(QtCore.Qt.AlignLeft)
        self.vf_atp_chargesaver_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))

        self.vf_atp_smartmode_lbl = QtWidgets.QLabel("Smart Mode")
        self.vf_atp_smartmode_lbl.setAlignment(QtCore.Qt.AlignLeft)
        self.vf_atp_smartmode_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))

        self.vf_atp_s1s2_lbl = QtWidgets.QLabel("S1-S2(Ramp+)=(%RR)")
        self.vf_atp_s1s2_lbl.setAlignment(QtCore.Qt.AlignLeft)
        self.vf_atp_s1s2_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))

        self.vf_atp_s2sn_lbl = QtWidgets.QLabel("S2-SN(Ramp+)=(%RR)")
        self.vf_atp_s2sn_lbl.setAlignment(QtCore.Qt.AlignLeft)
        self.vf_atp_s2sn_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))

    def vf_atp_comboboxes(self):
        self.vf_atp_combobox = QtWidgets.QComboBox()
        self.vf_atp_combobox.addItems("During-Charging Before-Charging OFF".split())
        self.vf_atp_combobox.setCurrentIndex(0)
        self.vf_atp_combobox.setFixedWidth(150)
        self.vf_atp_combobox.currentIndexChanged.connect(self.vf_atp_combobox_changed)

        self.vf_atp_type = QtWidgets.QComboBox()
        self.vf_atp_type.addItems("Ramp Burst Ramp+".split())
        self.vf_atp_type.setCurrentIndex(1)
        self.vf_atp_type.setFixedWidth(150)
        self.vf_atp_type.currentIndexChanged.connect(self.vf_atp_type_changed)

        self.vf_atp_del_8rr = QtWidgets.QComboBox()
        self.vf_atp_del_8rr.addItems("200 210 220 230 240 250 260 270 280 290 300".split())
        self.vf_atp_del_8rr.setCurrentIndex(4)
        self.vf_atp_del_8rr.setFixedWidth(150)
        self.vf_atp_del_8rr.currentIndexChanged.connect(self.vf_atp_del_8rr_changed)

        self.vf_atp_num_seq_before = QtWidgets.QComboBox()
        self.vf_atp_num_seq_before.addItems("0 1".split())
        self.vf_atp_num_seq_before.setCurrentIndex(0)
        self.vf_atp_num_seq_before.currentIndexChanged.connect(self.vf_atp_num_seq_before_changed)
        self.vf_atp_num_seq_before.hide()

        self.vf_atp_num_seq_during = QtWidgets.QComboBox()
        self.vf_atp_num_seq_during.addItems("1")
        self.vf_atp_num_seq_during.setCurrentIndex(0)

        self.vf_atp_num_pulses = QtWidgets.QComboBox()
        self.vf_atp_num_pulses.addItems("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15".split())
        # If Ramp - nominal 6 (5) pulses, Ramp+ 3 (2) pulses, Burst 8 (7) pulses
        self.vf_atp_num_pulses.setCurrentIndex(5)
        self.vf_atp_num_pulses.currentIndexChanged.connect(self.vf_atp_num_pulses_changed)

        self.vf_atp_rs1 = QtWidgets.QComboBox()
        self.vf_atp_rs1.addItems("50 53 56 59 63 66 69 72 75 78 81 84 88 91 94 97".split())
        self.vf_atp_rs1.setCurrentIndex(13) # 91
        self.vf_atp_rs1.setFixedWidth(150)
        self.vf_atp_rs1.currentIndexChanged.connect(self.vf_atp_rs1_changed)
        # Ramp: default 91 (13), Burst: default 88 (12), Ramp+: default 75 (8)

        self.vf_atp_int_dec = QtWidgets.QComboBox() # Only for ramp and burst
        self.vf_atp_int_dec.addItems("0 10 20 30 40".split())
        self.vf_atp_int_dec.setCurrentIndex(1)
        self.vf_atp_int_dec.currentIndexChanged.connect(self.vf_atp_int_dec_changed)

        self.vf_atp_chargesaver = QtWidgets.QComboBox()
        self.vf_atp_chargesaver.addItems(["ON", "OFF"])
        self.vf_atp_chargesaver.setCurrentIndex(0)
        self.vf_atp_chargesaver.currentIndexChanged.connect(self.vf_atp_chargesaver_changed)

        self.vf_atp_smartmode = QtWidgets.QComboBox()
        self.vf_atp_smartmode.addItems(["ON", "OFF"])
        self.vf_atp_smartmode.setFixedWidth(75)
        self.vf_atp_smartmode.setCurrentIndex(0)
        self.vf_atp_smartmode.currentIndexChanged.connect(self.vf_atp_smartmode_changed)

        self.vf_atp_s1s2 = QtWidgets.QComboBox()
        self.vf_atp_s1s2.addItems("50 53 56 59 63 66 69 72 75 78 81 84 88 91 94 97".split())
        self.vf_atp_s1s2.setCurrentIndex(6) # 69
        self.vf_atp_s1s2.currentIndexChanged.connect(self.vf_atp_s1s2_changed)

        self.vf_atp_s2sn = QtWidgets.QComboBox()
        self.vf_atp_s2sn.addItems("50 53 56 59 63 66 69 72 75 78 81 84 88 91 94 97".split())
        self.vf_atp_s2sn.setCurrentIndex(5) # 66
        self.vf_atp_s2sn.currentIndexChanged.connect(self.vf_atp_s2sn_changed)

    def vt_labels(self):

        self.vt_rx1_lbl = QtWidgets.QLabel("Rx1")
        self.vt_vt_rx1_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))
        self.vt_rx1_lbl.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self.vt_rx1_lbl.setFixedWidth(150)

        self.vt_rx2_lbl = QtWidgets.QLabel("Rx2")
        self.vt_rx2_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))
        self.vt_rx2_lbl.setFixedWidth(150)
        self.vt_rx2_lbl.setAlignment(QtCore.Qt.AlignLeft| QtCore.Qt.AlignTop)

        self.vt_rx3_lbl = QtWidgets.QLabel("Rx3")
        self.vt_rx3_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))
        self.vt_rx3_lbl.setFixedWidth(150)
        self.vt_rx3_lbl.setAlignment(QtCore.Qt.AlignLeft| QtCore.Qt.AlignTop)

        self.vt_rx4_lbl = QtWidgets.QLabel("Rx4")
        self.vt_rx4_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))
        self.vt_rx4_lbl.setFixedWidth(150)
        self.vt_rx4_lbl.setAlignment(QtCore.Qt.AlignLeft| QtCore.Qt.AlignTop)

        self.vt_rx5_lbl = QtWidgets.QLabel("Rx5")
        self.vt_rx5_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))
        self.vt_rx5_lbl.setFixedWidth(150)
        self.vt_rx5_lbl.setAlignment(QtCore.Qt.AlignLeft)

        self.vt_rx6_lbl = QtWidgets.QLabel("Rx6")
        self.vt_rx6_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))
        self.vt_rx6_lbl.setFixedWidth(150)
        self.vt_rx6_lbl.setAlignment(QtCore.Qt.AlignLeft| QtCore.Qt.AlignTop)

        self.vt_rx_status_lbl = QtWidgets.QLabel("VT Therapy Status")
        self.vt_rx_status_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))
        self.vt_rx_status_lbl.setFixedWidth(150)
        self.vt_rx_status_lbl.setAlignment(QtCore.Qt.AlignLeft| QtCore.Qt.AlignTop)

        self.vt_atp_type_lbl = QtWidgets.QLabel("Therapy Type")
        self.vt_atp_type_lbl.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.vt_atp_type_lbl.setFixedWidth(150)
        self.vt_atp_type_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))

        self.vt_energy_lbl = QtWidgets.QLabel("Energy")
        self.vt_energy_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))
        self.vt_energy_lbl.setFixedWidth(150)
        self.vt_energy_lbl.setAlignment(QtCore.Qt.AlignLeft| QtCore.Qt.AlignTop)

        self.vt_pathway_lbl = QtWidgets.QLabel("Pathway")
        self.vt_pathway_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))
        self.vt_pathway_lbl.setFixedWidth(150)
        self.vt_pathway_lbl.setMinimumSize(QtCore.QSize(0, 0))
        self.vt_pathway_lbl.setAlignment(QtCore.Qt.AlignLeft| QtCore.Qt.AlignTop)

        self.vt_atp_num_pulses_lbl = QtWidgets.QLabel("Initial # Pulses")
        self.vt_atp_num_pulses_lbl.setAlignment(QtCore.Qt.AlignLeft)
        self.vt_atp_num_pulses_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))

        self.vt_atp_rs1_lbl = QtWidgets.QLabel("R-S1 Interval=(%RR)")
        self.vt_atp_rs1_lbl.setAlignment(QtCore.Qt.AlignLeft)
        self.vt_atp_rs1_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))

        self.vt_atp_s1s2_lbl = QtWidgets.QLabel("S1-S2(Ramp+)=(%RR)")
        self.vt_atp_s1s2_lbl.setAlignment(QtCore.Qt.AlignLeft)
        self.vt_atp_s1s2_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))

        self.vt_atp_s2sn_lbl = QtWidgets.QLabel("S2-SN(Ramp+)=(%RR)")
        self.vt_atp_s2sn_lbl.setAlignment(QtCore.Qt.AlignLeft)
        self.vt_atp_s2sn_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))

        self.vt_atp_int_dec_lbl = QtWidgets.QLabel("Interval Decrement (ms)")
        self.vt_atp_int_dec_lbl.setAlignment(QtCore.Qt.AlignLeft)
        self.vt_atp_int_dec_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))

        self.vt_atp_num_seq_during_lbl = QtWidgets.QLabel("# Sequences")
        self.vt_atp_num_seq_during_lbl.setAlignment(QtCore.Qt.AlignLeft)
        self.vt_atp_num_seq_during_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))

        self.vt_atp_smartmode_lbl = QtWidgets.QLabel("Smart Mode")
        self.vt_atp_smartmode_lbl.setAlignment(QtCore.Qt.AlignLeft)
        self.vt_atp_smartmode_lbl.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))

    def vt_comboboxes(self):
        # VT Rx Comboboxes
        self.vt_rx1_status = QtWidgets.QComboBox()
        self.vt_rx1_status.addItems(["ON", "OFF"])
        self.vt_rx1_status.setCurrentIndex(0)
        self.vt_rx1_status.setFixedWidth(100)
        self.vt_rx1_status.currentIndexChanged.connect(self.vt_rx1_status_changed)

        self.vt_rx2_status = QtWidgets.QComboBox()
        self.vt_rx2_status.addItems(["ON", "OFF"])
        self.vt_rx2_status.setCurrentIndex(0)
        self.vt_rx2_status.setFixedWidth(100)
        self.vt_rx2_status.currentIndexChanged.connect(self.vt_rx2_status_changed)

        self.vt_rx3_status = QtWidgets.QComboBox()
        self.vt_rx3_status.addItems(["ON", "OFF"])
        self.vt_rx3_status.setCurrentIndex(0)
        self.vt_rx3_status.setFixedWidth(100)
        self.vt_rx3_status.currentIndexChanged.connect(self.vt_rx3_status_changed)

        self.vt_rx4_status = QtWidgets.QComboBox()
        self.vt_rx4_status.addItems(["ON", "OFF"])
        self.vt_rx4_status.setCurrentIndex(0)
        self.vt_rx4_status.setFixedWidth(100)
        self.vt_rx4_status.currentIndexChanged.connect(self.vt_rx4_status_changed)

        self.vt_rx5_status = QtWidgets.QComboBox()
        self.vt_rx5_status.addItems(["ON", "OFF"])
        self.vt_rx5_status.setCurrentIndex(0)
        self.vt_rx5_status.setFixedWidth(100)
        self.vt_rx5_status.currentIndexChanged.connect(self.vt_rx5_status_changed)

        self.vt_rx6_status = QtWidgets.QComboBox()
        self.vt_rx6_status.addItems(["ON", "OFF"])
        self.vt_rx6_status.setCurrentIndex(0)
        self.vt_rx6_status.setFixedWidth(100)
        self.vt_rx6_status.currentIndexChanged.connect(self.vt_rx6_status_changed)

        self.vt_rx1_energy = QtWidgets.QComboBox()
        self.vt_rx1_energy.addItems("0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 18 20 22 24 25 26 28 30 32 35 40".split())
        self.vt_rx1_energy.setCurrentIndex(33)
        self.vt_rx1_energy.setFixedWidth(100)
        self.vt_rx1_energy.currentIndexChanged.connect(self.vt_rx1_energy_changed)

        self.vt_rx2_energy = QtWidgets.QComboBox()
        self.vt_rx2_energy.addItems("0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 18 20 22 24 25 26 28 30 32 35 40".split())
        self.vt_rx2_energy.setCurrentIndex(33)
        self.vt_rx2_energy.setFixedWidth(100)
        self.vt_rx2_energy.currentIndexChanged.connect(self.vt_rx2_energy_changed)

        self.vt_rx3_energy = QtWidgets.QComboBox()
        self.vt_rx3_energy.addItems("10 11 12 13 14 15 16 18 20 22 24 25 26 28 30 32 35 40".split())
        self.vt_rx3_energy.setCurrentIndex(17)
        self.vt_rx3_energy.setFixedWidth(100)
        self.vt_rx3_energy.currentIndexChanged.connect(self.vt_rx3_energy_changed)

        self.vt_rx4_energy = QtWidgets.QComboBox()
        self.vt_rx4_energy.addItems("10 11 12 13 14 15 16 18 20 22 24 25 26 28 30 32 35 40".split())
        self.vt_rx4_energy.setCurrentIndex(17)
        self.vt_rx4_energy.setFixedWidth(100)
        self.vt_rx4_energy.currentIndexChanged.connect(self.vt_rx4_energy_changed)

        self.vt_rx5_energy = QtWidgets.QComboBox()
        self.vt_rx5_energy.addItems("10 11 12 13 14 15 16 18 20 22 24 25 26 28 30 32 35 40".split())
        self.vt_rx5_energy.setCurrentIndex(17)
        self.vt_rx5_energy.setFixedWidth(100)
        self.vt_rx5_energy.currentIndexChanged.connect(self.vt_rx5_energy_changed)

        self.vt_rx6_energy = QtWidgets.QComboBox()
        self.vt_rx6_energy.addItems("10 11 12 13 14 15 16 18 20 22 24 25 26 28 30 32 35 40".split())
        self.vt_rx6_energy.setCurrentIndex(17)
        self.vt_rx6_energy.setFixedWidth(100)
        self.vt_rx6_energy.currentIndexChanged.connect(self.vt_rx6_energy_changed)

        self.vt_rx1_pathway = QtWidgets.QComboBox()
        self.vt_rx1_pathway.addItems(["AX>B", "B>AX"])
        self.vt_rx1_pathway.setCurrentIndex(1)
        self.vt_rx1_pathway.setFixedWidth(100)
        self.vt_rx1_pathway.currentIndexChanged.connect(self.vt_rx1_pathway_changed)

        self.vt_rx2_pathway = QtWidgets.QComboBox()
        self.vt_rx2_pathway.addItems(["AX>B", "B>AX"])
        self.vt_rx2_pathway.setCurrentIndex(1)
        self.vt_rx2_pathway.setFixedWidth(100)
        self.vt_rx2_pathway.currentIndexChanged.connect(self.vt_rx2_pathway_changed)

        self.vt_rx3_pathway = QtWidgets.QComboBox()
        self.vt_rx3_pathway.addItems(["AX>B", "B>AX"])
        self.vt_rx3_pathway.setCurrentIndex(1)
        self.vt_rx3_pathway.setFixedWidth(100)
        self.vt_rx3_pathway.currentIndexChanged.connect(self.vt_rx3_pathway_changed)

        self.vt_rx4_pathway = QtWidgets.QComboBox()
        self.vt_rx4_pathway.addItems(["AX>B", "B>AX"])
        self.vt_rx4_pathway.setFixedWidth(100)
        self.vt_rx4_pathway.setCurrentIndex(1)
        self.vt_rx4_pathway.currentIndexChanged.connect(self.vt_rx4_pathway_changed)

        self.vt_rx5_pathway = QtWidgets.QComboBox()
        self.vt_rx5_pathway.addItems(["AX>B", "B>AX"])
        self.vt_rx5_pathway.setFixedWidth(100)
        self.vt_rx5_pathway.setCurrentIndex(0)
        self.vt_rx5_pathway.currentIndexChanged.connect(self.vt_rx5_pathway_changed)

        self.vt_rx6_pathway = QtWidgets.QComboBox()
        self.vt_rx6_pathway.addItems(["AX>B", "B>AX"])
        self.vt_rx6_pathway.setFixedWidth(100)
        self.vt_rx6_pathway.setCurrentIndex(0)
        self.vt_rx6_pathway.currentIndexChanged.connect(self.vt_rx6_pathway_changed)

    def vf_rx1_status_changed(self, i):
        if i == 0:
            self.vf_rx1_status = "ON"
        else:
            self.vf_rx1_status = "OFF"

    def vf_rx2_status_changed(self, i):
        if i == 0:
            self.vf_rx2_status = "ON"
        else:
            self.vf_rx2_status = "OFF"

    def vf_rx3_status_changed(self, i):
        if i == 0:
            self.vf_rx3_status = "ON"
        else:
            self.vf_rx3_status = "OFF"

    def vf_rx4_status_changed(self, i):
        if i == 0:
            self.vf_rx4_status = "ON"
        else:
            self.vf_rx4_status = "OFF"

    def vf_rx5_status_changed(self, i):
        if i == 0:
            self.vf_rx5_status = "ON"
        else:
            self.vf_rx5_status = "OFF"

    def vf_rx6_status_changed(self, i):
        if i == 0:
            self.vf_rx6_status = "ON"
        else:
            self.vf_rx6_status = "OFF"

    def vf_rx1_energy_changed(self, i):
        self.vf_rx1_energy = i

    def vf_rx2_energy_changed(self, i):
        self.vf_rx2_energy = i

    def vf_rx3_energy_changed(self, i):
        self.vf_rx3_energy = i

    def vf_rx4_energy_changed(self, i):
        self.vf_rx4_energy = i

    def vf_rx5_energy_changed(self, i):
        self.vf_rx5_energy = i

    def vf_rx6_energy_changed(self, i):
        self.vf_rx6_energy = i

    def vf_rx1_pathway_changed(self, i):
        self.vf_rx1_pathway = i

    def vf_rx2_pathway_changed(self, i):
        self.vf_rx2_pathway = i

    def vf_rx3_pathway_changed(self, i):
        self.vf_rx3_pathway = i

    def vf_rx4_pathway_changed(self, i):
        self.vf_rx4_pathway = i

    def vf_rx5_pathway_changed(self, i):
        self.vf_rx5_pathway = i

    def vf_rx6_pathway_changed(self, i):
        self.vf_rx6_pathway = i

    def vf_atp_combobox_changed(self, i):
        self.vf_atp_tx = i

    def vf_atp_type_changed(self, i):
        self.vf_atp_type_tx = i

    def vf_atp_del_8rr_changed(self, i):
        self.vf_atp_del_8rr_tx = i

    def vf_atp_num_seq_before_changed(self, i):
        self.vf_atp_num_seq_before_tx = i

    def vf_atp_num_seq_during_changed(self, i):
        self.vf_atp_num_seq_during_tx = i

    def vf_atp_num_pulses_changed(self, i):
        self.vf_atp_num_pulses_tx = i

    def vf_atp_rs1_changed(self, i):
        self.vf_atp_rs1_tx = i

    def vf_atp_s1s2_changed(self, i):
        self.vf_atp_s2sn_tx = i

    def vf_atp_s2sn_changed(self, i):
        self.vf_atp_s2sn_tx = i

    def vf_atp_int_dec_changed(self, i):
        self.vf_atp_int_dec_tx = i

    def vf_atp_chargesaver_changed(self, i):
        self.vf_atp_chargesaver_tx = i

    def vf_atp_smartmode_changed(self, i):
        self.vf_atp_smartmode_tx = i

