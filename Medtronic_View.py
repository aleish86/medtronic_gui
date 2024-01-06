# Medtronic Viewer
import os
import sys
import hdy
from PySide6 import QtWidgets
os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'

SOURCE="rvnoise"

def main():
    if SOURCE=="aad":
        database_dir = "/Users/amiyazawa/Library/CloudStorage/Dropbox/01 Research/02 Current Projects/02 AAD/Data"
        database_fn = 'AAD_all.csv'
        patient = 'SR01'#'A01'#'B09' #'VTAbl03'#'VTAbl07'#'A01'#'B08'
        exp ='Exp14' #'Exp1' #'Exp185' #"Exp20" #297 (RVP) "Exp1" vs 282 (HBP)
        mode = "normal"
    if SOURCE=="rvnoise":
        database_dir = "/Users/amiyazawa/Library/CloudStorage/Dropbox/01 Research/02 Current Projects/02 AAD/Data"
        database_fn = 'SimRVNoise.csv'
        patient = 'A01'#'A01'
        exp = 'Exp3' #'Exp3'
        mode = "normal"

    else:
        raise Exception

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Medtronic Viewer")

    pt = hdy.LaserAnalysis1(database_dir=database_dir, database_fn=database_fn, patient=patient, exp=exp, mode=mode)
    m = hdy.MedtronicGui(pt)

    sys.exit(app.exec())

if __name__ == '__main__':
    main()