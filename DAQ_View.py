import sys
import os
import hdy
from DAQ_GUI import DAQ_GUI
from DAQ_File import DAQ_File

from PySide6 import QtWidgets
os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = DAQ_GUI()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
