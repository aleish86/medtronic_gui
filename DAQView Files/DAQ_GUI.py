# import libraries
import os
from PySide6 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from .DAQ_File import DAQ_File
# Need to create tab for Moving plot graphs
class DAQ_GUI(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        # self.setStyleSheet("background-color: white;")

        self.daq_exp_widget = None

        self.statusBar().showMessage('Ready')

        self.open_file_btn = QtWidgets.QPushButton("Open DAQ File")
        self.open_file_btn.clicked.connect(self.showFileOpen)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')

        openFile = QtGui.QAction(QtGui.QIcon('open.png'), 'Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.showFileOpen)

        fileMenu.addAction(openFile)
        menubar.addMenu(fileMenu)

        self.setWindowTitle('DAQ View')
        self.setCentralWidget(self.open_file_btn)
        self.resize(1000, 800)
        self.show()

    def showFileOpen(self):
        daq_exp_fl = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/Users/amiyazawa/Library/CloudStorage/Dropbox/01 Research/02 Current Projects/')

        if daq_exp_fl is None:
            return

        if True:
            daq_exp_dir, daq_exp_fn = os.path.split(daq_exp_fl[0])
            daq_exp = DAQ_File(daq_exp_dir, daq_exp_fn, search_for_files=False)
            self.daq_exp_widget = DAQ_Exp_Widget(daq_exp)
            self.setCentralWidget(self.daq_exp_widget)

class DAQ_Exp_Widget(QtWidgets.QWidget):
    def __init__(self, daq_exp, parent = None):
        super().__init__(parent=parent)

        self.plot_w_list = []
        self.lbl_plot_list = {}
        self.source_lbl_list = {}
        self.ecg_plot_num = None

        self.daq_exp = daq_exp
        self.setup()

        self.show()

    def setup(self):
        pg.setConfigOptions(antialias=True, background='w')

        self.control_layout = QtWidgets.QVBoxLayout()

        self.pos_lbl = QtWidgets.QLabel()
        self.zipfl_lbl = QtWidgets.QLabel()

        # self.control_layout.addWidget(self.pos_lbl)
        # self.control_layout.setSpacing(0)
        # self.control_layout.setContentsMargins(0,0,0,0)

        # Buttons for Time Stamp Data
        self.spacer_lbl = QtWidgets.QLabel("\n")

# n        self.csv_input = QtWidgets.QLineEdit()
#         self.csv_input.setText("Enter CSV Filepath here")
#         self.csv_input.setFixedWidth(250)
#         self.csv_input.textChanged.connect(self.csv_input_changed)
#
#         self.timestamp_lbl = QtWidgets.QLineEdit()
#         self.timestamp_lbl.setText("Baseline")
#         self.timestamp_lbl.setFixedWidth(250)
#         self.timestamp_lbl.textChanged.connect(self.time_stamp_lbl_changed)
#
#         self.notes_lbl = QtWidgets.QLineEdit()
#         self.notes_lbl.setText("Comment")
#         self.notes_lbl.setFixedWidth(250)
#         self.notes_lbl.textChanged.connect(self.notes_lbl_changed)
#
#         self.group_lbl = QtWidgets.QLineEdit()
#         self.group_lbl.setText("SimVT")
#         self.group_lbl.setFixedWidth(250)
#         self.group_lbl.textChanged.connect(self.group_lbl_changed)

        self.calc_move_btn = QtWidgets.QPushButton("Move Calc ROI")
        self.calc_move_btn.setFixedWidth(150)
        self.calc_move_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.calc_move_btn.clicked.connect(self.calc_region_move)

        # self.timestamp_btn = QtWidgets.QPushButton("Time Stamp Period")
        # self.timestamp_btn.setFixedWidth(150)
        # self.timestamp_btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        # self.timestamp_btn.clicked.connect(self.time_stamp_period)

        self.ecg_lbl = QtWidgets.QLabel("ecg")
        self.bp_lbl = QtWidgets.QLabel("bp")
        self.bpao_lbl = QtWidgets.QLabel("bpao")
        self.plethg_lbl = QtWidgets.QLabel("plethg")
        self.qfin_lbl = QtWidgets.QLabel("qfin")
        self.plethh_lbl = QtWidgets.QLabel("plethh")
        self.boxa_lbl = QtWidgets.QLabel("boxa")
        self.boxb_lbl = QtWidgets.QLabel("boxb")
        self.plethi_lbl = QtWidgets.QLabel("plethi")
        self.plethr_lbl = QtWidgets.QLabel("plethr")

        # self.ecg_plot_lbl = QtWidgets.QComboBox()
        # self.ecg_plot_lbl.addItems("None BipECG ECG3 BP BPAO dBP Laser1 Laser2 CBF RALead RVBip RVShock LVLead".split())
        # self.ecg_plot_lbl.currentIndexChanged.connect(self.ecg_plot_lbl_changed)
        #
        # self.boxa_plot_lbl = QtWidgets.QComboBox()
        # self.boxa_plot_lbl.addItems("None BipECG ECG3 BP BPAO dBP Laser1 Laser2 CBF RALead RVBip RVShock LVLead".split())
        # self.boxa_plot_lbl.currentIndexChanged.connect(self.boxa_plot_lbl_changed)
        #
        # self.boxb_plot_lbl = QtWidgets.QComboBox()
        # self.boxb_plot_lbl.addItems("None BipECG ECG3 BP BPAO dBP Laser1 Laser2 CBF RALead RVBip RVShock LVLead".split())
        # self.boxb_plot_lbl.currentIndexChanged.connect(self.boxb_plot_lbl_changed)
        #
        # self.bp_plot_lbl = QtWidgets.QComboBox()
        # self.bp_plot_lbl.addItems("None BipECG ECG3 BP BPAO dBP Laser1 Laser2 CBF RALead RVBip RVShock LVLead".split())
        # self.bp_plot_lbl.currentIndexChanged.connect(self.bp_plot_lbl_changed)
        #
        # self.bpao_plot_lbl = QtWidgets.QComboBox()
        # self.bpao_plot_lbl.addItems("None BipECG ECG3 BP BPAO dBP Laser1 Laser2 CBF RALead RVBip RVShock LVLead".split())
        # self.bpao_plot_lbl.currentIndexChanged.connect(self.bpao_plot_lbl_changed)
        #
        # self.plethg_plot_lbl = QtWidgets.QComboBox()
        # self.plethg_plot_lbl.addItems("None BipECG ECG3 BP BPAO dBP Laser1 Laser2 CBF RALead RVBip RVShock LVLead".split())
        # self.plethg_plot_lbl.currentIndexChanged.connect(self.plethg_plot_lbl_changed)
        #
        # self.plethh_plot_lbl = QtWidgets.QComboBox()
        # self.plethh_plot_lbl.addItems("None BipECG ECG3 BP BPAO dBP Laser1 Laser2 CBF RALead RVBip RVShock LVLead".split())
        # self.plethh_plot_lbl.currentIndexChanged.connect(self.plethh_plot_lbl_changed)
        #
        # self.qfin_plot_lbl = QtWidgets.QComboBox()
        # self.qfin_plot_lbl.addItems("None BipECG ECG3 BP BPAO dBP Laser1 Laser2 CBF RALead RVBip RVShock LVLead".split())
        # self.qfin_plot_lbl.currentIndexChanged.connect(self.qfin_plot_lbl_changed)
        #
        # self.plethi_plot_lbl = QtWidgets.QComboBox()
        # self.plethi_plot_lbl.addItems("None BipECG ECG3 BP BPAO dBP Laser1 Laser2 CBF RALead RVBip RVShock LVLead".split())
        # self.plethi_plot_lbl.currentIndexChanged.connect(self.plethi_plot_lbl_changed)
        #
        # self.plethr_plot_lbl = QtWidgets.QComboBox()
        # self.plethr_plot_lbl.addItems("None BipECG ECG3 BP BPAO dBP Laser1 Laser2 CBF RALead RVBip RVShock LVLead".split())
        # self.plethr_plot_lbl.currentIndexChanged.connect(self.plethr_plot_lbl_changed)

        self.vlayout = QtWidgets.QVBoxLayout()
        self.vlayout.setSpacing(0)
        self.vlayout.setContentsMargins(0,0,0,150)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.addWidget(self.pos_lbl)

        for i, source in enumerate(self.daq_exp.sources):
            if source == 'ecg':
                self.ecg_plot_num = i

            data = getattr(self.daq_exp, source)
            plot_label_le = QtWidgets.QLineEdit(source)
            plot_w = pg.PlotWidget()
            plot_pi = plot_w.getPlotItem()
            plot_pi.plot(data, pen="k", antialise=True, autoDownsample=True, downsampleMethod='peak', clipToView=False)
            plot_pi.setClipToView(True)
            plot_pi.showAxis('bottom', False)
            plot_pi.setLabel('left', plot_label_le)
            plot_pi.sigRangeChanged.connect(self.update_region)
            y_axis = plot_pi.getAxis('left')
            y_axis.setWidth(75)
            y_axis.setTextPen('k')

            hide_plot = QtWidgets.QCheckBox(source)
            hide_plot.setChecked(True)  # Initial state: plot is visible
            hide_plot.stateChanged.connect(self.toggle_plot_visibility)

            self.plot_w_list.append(plot_w)
            self.layout.addWidget(hide_plot)
            self.vlayout.addWidget(plot_label_le)
            self.layout.addWidget(plot_w)


        if True:
            source = 'ecg'
            data = getattr(self.daq_exp, source)

            plot_w = pg.PlotWidget()
            plot_pi = plot_w.getPlotItem()
            plot_pi.plot(data, pen="k", antialise=True, autoDownsample=True, downsampleMethod='peak', clipToView=False)
            plot_pi.setClipToView(True)
            plot_pi.showAxis('bottom', False)
            plot_pi.setLabel('left', source)
            plot_pi.getAxis('left').setWidth(75)

            self.overview_region = pg.LinearRegionItem()
            self.overview_region.setZValue(10)
            self.overview_region.sigRegionChanged.connect(self.region_updated)
            plot_pi.addItem(self.overview_region)
            self.overview_region.sigRegionChanged.connect(self.region_updated)
            self.overview_region.setRegion((0, data.shape[0]))

            self.layout.addWidget(plot_w)

        if self.ecg_plot_num is not None:
            i = self.ecg_plot_num
            self.vLine = pg.InfiniteLine(angle=90, movable=False, pen='r')
            self.hLine = pg.InfiniteLine(angle=0, movable=False, pen = 'r')
            self.plot_w_list[i].addItem(self.vLine, ignoreBounds=True)
            self.plot_w_list[i].addItem(self.hLine, ignoreBounds=True)
            self.ecg_vb = self.plot_w_list[i].getPlotItem().vb
            self.plot_w_list[i].scene().sigMouseMoved.connect(self.mouse_moved)

            self.calc_region = pg.LinearRegionItem()
            self.calc_region.setZValue(-10)
            self.plot_w_list[i].addItem(self.calc_region)
            self.calc_region.sigRegionChanged.connect(self.calc_region_changed)

        self.vlayout.addWidget(self.calc_move_btn)

        self.main_layout = QtWidgets.QHBoxLayout()
        self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(0, 0, 0, 0)  # Adjust the margin values as needed

        self.main_layout.addLayout(self.vlayout)
        self.main_layout.addLayout(self.layout)
        self.layout.addWidget(self.zipfl_lbl)
        self.zipfl_lbl.setText(str(self.daq_exp.zip_fl))
        self.layout.addWidget(self.spacer_lbl)
        for item in (self.layout.itemAt(i).widget() for i in range(self.layout.count())):
            item.setStyleSheet("background-color: white;")

        self.setLayout(self.main_layout)

    def update_region(self, window, viewRange):
        rgn = viewRange[0]
        self.overview_region.setRegion(rgn)

    def region_updated(self):
        self.overview_region.setZValue(10)
        minX, maxX = self.overview_region.getRegion()
        for plot_w in self.plot_w_list:
            plot_w.setXRange(minX, maxX, padding=0)

    def mouse_moved(self, evt):
        if self.ecg_plot_num is None:
            return

        i = self.ecg_plot_num

        pos = evt.toPoint()

        if self.plot_w_list[i].sceneBoundingRect().contains(pos):
            mousePoint = self.ecg_vb.mapSceneToView(pos)
            index = int(mousePoint.x())
            if index > 0:
                self.pos_lbl.setText("X: {pos}".format(pos=int(mousePoint.x())))
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())

    def toggle_plot_visibility(self, state):
        checkbox = self.sender()
        index = self.layout.indexOf(checkbox)  # Get the index of the checkbox in the layout
        plot_widget = self.layout.itemAt(index + 1).widget()  # Get the plot widget next to the checkbox
        print(self.sender().text())

        if state == QtCore.Qt.Checked:
            plot_widget.show()
            if self.sender().text() == self.ecg_lbl.text():
                self.ecg_lbl.show()
            if self.sender().text() == self.bp_lbl.text():
                self.bp_lbl.show()
            if self.sender().text() == self.bpao_lbl.text():
                self.bpao_lbl.show()
            if self.sender().text() == self.plethg_lbl.text():
                self.plethg_lbl.show()
            if self.sender().text() == self.qfin_lbl.text():
                self.qfin_lbl.show()
            if self.sender().text() == self.plethh_lbl.text():
                self.plethh_lbl.show()
            if self.sender().text() == self.plethi_lbl.text():
                self.plethi_lbl.show()
            if self.sender().text() == self.plethr_lbl.text():
                self.plethr_lbl.show()
            if self.sender().text() == self.boxa_lbl.text():
                self.boxa_lbl.show()
            if self.sender().text() == self.boxb_lbl.text():
                self.boxb_lbl.show()

        else:
            plot_widget.hide()
            if checkbox.text() == self.ecg_lbl.text():
                self.ecg_lbl.hide()
            if self.sender().text() == self.bp_lbl.text():
                self.bp_lbl.hide()
            if self.sender().text() == self.bpao_lbl.text():
                self.bpao_lbl.hide()
            if self.sender().text() == self.plethg_lbl.text():
                self.plethg_lbl.hide()
            if self.sender().text() == self.qfin_lbl.text():
                self.qfin_lbl.hide()
            if self.sender().text() == self.plethh_lbl.text():
                self.plethh_lbl.hide()
            if self.sender().text() == self.plethi_lbl.text():
                self.plethi_lbl.hide()
            if self.sender().text() == self.plethr_lbl.text():
                self.plethr_lbl.hide()
            if self.sender().text() == self.boxa_lbl.text():
                self.boxa_lbl.hide()
            if self.sender().text() == self.boxb_lbl.text():
                self.boxb_lbl.hide()

    # Calc Region Changes
    def calc_region_changed(self):
        roi_begin, roi_end = self.calc_region.getRegion()
        print(roi_begin, roi_end)

    def calc_region_move(self):
        x, y = self.overview_region.getRegion()
        print(x, y)

        rgn = (x + (y - x) / 4, y - (y - x) / 4)
        self.calc_region.setRegion(rgn)
