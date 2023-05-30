from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self,parent)
        self.canvas = FigureCanvas(Figure(tight_layout=True))

        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)

        self.canvas.axis = self.canvas.figure.add_subplot()
        self.setLayout(vertical_layout)