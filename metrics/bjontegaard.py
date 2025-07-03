import numpy as np
import scipy
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class Bjontegaard_Delta:
    def compute_BD_PSNR(self, model1, model2):
        log_bitrates1 = np.log10(model1.bitrates)
        log_bitrates2 = np.log10(model2.bitrates)

        # Get bounds for integration
        rL = np.max([np.min(log_bitrates1), np.min(log_bitrates2)])
        rH = np.min([np.max(log_bitrates1), np.max(log_bitrates2)])
        if rH <= rL:
            return np.nan 

        # Integrals of the polynomial in log space
        P1 = np.poly1d(np.polyint(model1.parameters_PSNR))
        P2 = np.poly1d(np.polyint(model2.parameters_PSNR))

        bd_delta_PSNR = 1 / (rH-rL) * ( (P2(rH)-P1(rH)) - (P2(rL)-P1(rL)) )
        return bd_delta_PSNR

    def compute_BD_Rate(self, model1, model2):
        D1 = model1.psnr_values
        D2 = model2.psnr_values
        
        # Get bounds for integration
        DL = np.max([np.min(D1), np.min(D2)])
        DH = np.min([np.max(D1), np.max(D2)])

        if DH <= DL:
            return np.nan 

        # Integrals of the polynomial
        P1 = np.poly1d(np.polyint(model1.parameters_Rate))
        P2 = np.poly1d(np.polyint(model2.parameters_Rate))

        exponent = 1 / (DH-DL) * ( (P2(DH)-P1(DH)) - (P2(DL)-P1(DL)) )
        bd_delta_rate = 10**(exponent) - 1
        return bd_delta_rate




class Bjontegaard_Model:
    def __init__(self, bitrates, psnr_values):
        self.bitrates = bitrates
        self.psnr_values = psnr_values

        self.parameters_PSNR = [0, 0, 0, 0]
        self.parameters_Rate = [0, 0, 0, 0]
        self.__update_model()

    def __update_model(self):
        logR = np.log10(self.bitrates) 
        self.parameters_PSNR = np.polyfit(logR, self.psnr_values, 3, rcond=1e-8)
        self.parameters_Rate = np.polyfit(self.psnr_values, logR, 3, rcond=1e-8)

    def evaluate(self, R):
        logR = np.log10(R)
        p = np.poly1d(self.parameters_PSNR)
        value = p(logR)
        return value

    def evaluate_rate(self, R):
        p = np.poly1d(self.parameters_Rate)
        value = p(R)
        value = 10**value
        return value

    def plot(self, ax):
        xdata = np.linspace(np.min(self.bitrates), np.max(self.bitrates), 100)
        p = np.poly1d(self.parameters_PSNR)

        ax.scatter(self.bitrates, self.psnr_values)
        ax.plot(xdata, p(np.log10(xdata)))

    def get_plot_data(self):
        xdata = np.linspace(np.min(self.bitrates), np.max(self.bitrates), 100)
        p = np.poly1d(self.parameters_PSNR)
        ydata = p(np.log10(xdata))
        return self.bitrates, self.psnr_values, xdata, ydata


if __name__ == "__main__":
    # Test of the BD model
    bitrates1 = [0.01, 0.2, 0.6, 1.52]
    bitrates2 = [0.0270371, 0.151195, 0.615206, 1.64363]
    d1 = [0.02, 0.003, 0.0015, 0.001]# 0.0003]
    d2 = [0.00946959, 0.00181347, 0.0012, 0.001]#0.000378549, 0.000165418]
    metric1 = Bjontegaard_Model(bitrates1, d1)
    metric2 = Bjontegaard_Model(bitrates2, d1)
    fig, ax = plt.subplots()
    metric1.plot(ax)
    metric2.plot(ax)
    BD_Delta = Bjontegaard_Delta()
    bd_psnr = BD_Delta.compute_BD_PSNR(metric1, metric2)
    bd_rate = BD_Delta.compute_BD_Rate(metric1, metric2)
    print(bd_psnr, bd_rate)
    bd_psnr = BD_Delta.compute_BD_PSNR(metric2, metric1)
    bd_rate = BD_Delta.compute_BD_Rate(metric2, metric1)
    print(bd_psnr, bd_rate)
    plt.show()
