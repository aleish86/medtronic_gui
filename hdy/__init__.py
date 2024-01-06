from .DAQ_File import DAQ_File
#from .EGM_Analysis_Functions import *
from .BaseClasses import DAQSignal, DAQContainerBP, DAQContainerBPAO, DAQContainerBoxA, DAQContainerCranial, DAQContainerECG, DAQContainerLaser
from .sensing_class import sensing
# from .LaserClasses import LaserAnalysis
# from .Laser_GUI import LaserGui
from .laser import calc_magic_laser
# from .EGM_Analysis_Classes import EGMAnalysis
from .AAD_LaserClasses import LaserAnalysis1
from .Medtronic_GUI import MedtronicGui
from .Medtronic_GUI import TimeAxisItem
from .Medtronic_GUI import data_list
from .Medtronic_Analysis import MedtronicAnalysis
from .wavelet_gui import Wavelet_GUI
from .beatplot_gui import BeatPlot_GUI
from .vvi_gui import MedtronicVVI_GUI
from .therapies_gui import vtherapies_GUI
# from .DAQ_GUI import DAQ_GUI
# from .am_algorithm import am_egm



def remove_outliers(data):
    data_mean = data.mean()
    data_std = data.std()
    data[(data > (data_mean + 5 * data_std)) | (data < (data_mean - 5 * data_std))] = 0
    return data
