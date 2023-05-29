from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QThread, pyqtSignal
from FK import *
from UI import Ui_Dialog
from time import time
import scipy.interpolate as interp
import numpy as np
import matplotlib.pyplot as plt

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.pushButton_start.clicked.connect(self.buttonClicked)

    def buttonClicked(self):
        print('Start')

        # read user inputs from lineEdit
        self.r_ca = float(self.ui.lineEdit_r_ca.text())
        self.nx = int(self.ui.lineEdit_nx.text())
        self.ny = int(self.ui.lineEdit_ny.text())
        self.nz = int(self.ui.lineEdit_nz.text())
        self.n_bins = int(self.ui.lineEdit_n_bins.text())
        self.filepath = (self.ui.lineEdit_filepath.text())
        
        # refresh outputs according to user inputs
        self.ui.label_r_ca.setText('r_ca = {}'.format(self.r_ca))
        self.ui.label_n_xyz.setText('[nx, ny, nz] = [{}, {}, {}]'.format(self.nx,self.ny,self.nz))
        self.ui.label_n_bins.setText('n_bins = {}'.format(self.n_bins))

        self.n_particles = self.nx*self.ny*self.nz*30
        self.ui.label_n_particles.setText('n_particles = {}'.format(self.n_particles))

        print('r_ca = {}'.format(self.r_ca))
        print('[nx, ny, nz] = [{}, {}, {}]'.format(self.nx,self.ny,self.nz))
        print('n_bins = {}'.format(self.n_bins))

        self.start_progress()
        self.coordinates_inputs()
        self.eval_Debye_SQ()

    def start_progress(self):
        print('reset progress bar')
        self.ui.progressBar.setMaximum(100)
        self.ui.progressBar.setValue(0)

    def progress_changed(self, value):        
        self.ui.progressBar.setValue(value)

    def coordinates_inputs(self):
        # c/a ratio
        ratio_ca=self.r_ca

        # 6*6*11 unit cells
        n_x = self.nx
        n_y = self.ny
        n_layers = self.nz

        l_a = np.sqrt((1+np.sqrt(3)/2)**2+0.5**2)
        l_c = l_a*ratio_ca

        c_layer_sigma = c_sigma(n_x,n_y,Ratio_ca=(1/ratio_ca))
        c_rod = stack_coords([shift_coords(c_layer_sigma, np.array([0,0,l_c])*s) for s in range(n_layers)])
        self.c_all = np.vstack(c_rod)
        self.V = n_x*l_a*n_y*l_a*n_layers*l_c

        self.bounds = np.array([[0,n_x*l_a],[0,n_y*l_a],[0,n_layers*l_c]])

        # interplanar spacings in FK sigma phase
        l_c_s = 2*np.pi/l_c
        l_a_s = 2*np.pi/l_a
        
        d_c_s = l_c_s*np.array([0,0,1])
        d_a1_s = l_a_s*np.array([1,0,0])
        d_a2_s = l_a_s*np.array([0,1,0])
        d_s = np.vstack([d_a1_s,d_a2_s,d_c_s])

        self.d_410 = 2*np.pi/np.linalg.norm(np.array([4,1,0])@d_s)
        self.Q_002 = np.linalg.norm(np.array([0,0,2])@d_s)
        self.Q_410 = np.linalg.norm(np.array([4,1,0])@d_s)

    def eval_Debye_SQ(self):
        self.qq = np.linspace(0.1,50,500)
        N = self.c_all.shape[0]
        rho = N/self.V

        self.qthread = ThreadTask()
        self.qthread.c = self.c_all.T
        self.qthread.qq = self.qq
        self.qthread.p_sub = 1.0
        self.qthread.n_bins = self.n_bins

        self.qthread.start()
        self.qthread.qthread_signal.connect(self.progress_changed)
        self.qthread.finished.connect(self.scattering_finished)
        self.qthread.start()

    def scattering_finished(self):
        self.S_q = self.qthread.S_q
        self.plot_results()
        self.ui.progressBar.setValue(100)

    def plot_results(self):
        """
        Plot the scattering function on MplWidget canvas
        self.ui.widget.canvas behaved line the normal matplotlib figure
        """
        ## postprocessing
        Qd410 = self.qq*self.d_410
        # Interpolation
        Q_scaled = np.linspace(0.04,16,400)
        f = interp.interp1d(Qd410,self.S_q,fill_value="extrapolate")
        SQ_scaled = f(Q_scaled)
        
        self.output_SQ = np.vstack([Q_scaled,SQ_scaled]).T

        np.savetxt(self.filepath+'SQ_{}.csv'.format(self.r_ca), self.output_SQ, delimiter=',')

        ## plot results
        ax = self.ui.widget.canvas.axis
        hold = self.ui.checkBox_hold.isChecked()
        if not hold:
            print('clear current plot')
            ax.cla()
        ax.plot(self.qq*self.d_410,self.S_q,'-',color='r')
        ax.plot([self.Q_410*self.d_410,self.Q_410*self.d_410],[-40,40],'-',color='b')
        ax.plot([self.Q_002*self.d_410,self.Q_002*self.d_410],[-40,40],'-',color='g')
        index_qq = (self.qq*self.d_410>4)*(self.qq*self.d_410<8)
        max_S_q = np.max(self.S_q[index_qq])
        ax.set_xlim(2,15)
        ax.set_ylim(-0.1,max_S_q*1.1)
        ax.set_xlabel('$Qd_{410}$',fontsize=12)
        ax.set_ylabel('$S(Qd_{410})$',fontsize=12)
        # ax.set_yscale('log')
        ax.tick_params(direction='in', axis='both', which='both', labelsize=10)

        self.ui.widget.canvas.draw()

class ThreadTask(QThread):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._c = None
        self._qq = None
        self._p_sub = None
        self._n_bins = None

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, value):
        self._c = value

    @property
    def qq(self):
        return self._qq

    @qq.setter
    def qq(self, value):
        self._qq = value

    @property
    def p_sub(self):
        return self._p_sub

    @p_sub.setter
    def p_sub(self, value):
        self._p_sub = value

    @property
    def n_bins(self):
        return self._n_bins

    @n_bins.setter
    def n_bins(self, value):
        self._n_bins = value

    qthread_signal = pyqtSignal(int)
    def run(self):
        """
        Calculate scattering function.

        Args:
            c: N by 3 particle trajectory
            qq: array
                wave vectors
            p_sub: amount of particles used to calculate S(Q)
        """
        c = self.c
        qq = self.qq
        p_sub = self.p_sub
        n_bins = self.n_bins

        N = c.shape[1]

        # two-point correlation
        n_list = int(N*p_sub)
        # np.random.seed(0)
        i_list = np.random.choice(np.arange(N), size=n_list, replace=False)
        r_jk = c[:,i_list].T.reshape(n_list,1,3) - c[:,i_list].T.reshape(1,n_list,3)
        d_jk = np.sqrt(np.sum(r_jk**2,axis=2))
        r_jk = None

        # RDF
        d_max = np.max(d_jk)
        rr = np.linspace(d_max/n_bins,d_max,n_bins)
        rho_r = np.zeros(n_bins)
        
        index_r_jk = np.floor(d_jk/d_max*n_bins) -1
        d_jk = None
        np.fill_diagonal(index_r_jk,n_bins*2) # we are not calculating these pairs
        index_f = index_r_jk.flatten()
        index_r_jk = None
        index_f = index_f[index_f!=(n_bins*2)]
        
        for i_r in range(len(index_f)):
            rho_r[int(index_f[i_r])] += 1
            if i_r%int(len(index_f)/100) == 0:
                self.qthread_signal.emit(int((i_r+1)/len(index_f)*100))

        S_q = np.array([np.sum(rho_r*np.sin(rr*q)/(rr*q)) for q in qq])/N + 1
        self.S_q = S_q
        # return S_q



