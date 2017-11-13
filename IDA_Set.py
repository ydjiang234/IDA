#An IDA set class to store and operate the IDA data
import numpy as np
from Spline_fit import Spline_fit
from scipy.interpolate import interp1d

class IDA_Stripe():

    def __init__(self, data):
        self.data = data
        self.num = len(data)
        self.separate_inf()
        self.median = self.get_median()
        self.fractile_16 = self.get_fractile(0.16)
        self.fractile_84 = self.get_fractile(0.84)
        if self.fractile_84 != 'inf':
            self.norm_mean = np.log(self.median)
            self.norm_CDF_16 = np.log(self.fractile_16)
            self.norm_CDF_84 = np.log(self.fractile_84)
            self.norm_sigma = (self.norm_CDF_84 - self.norm_CDF_16) / 2.0



    def separate_inf(self):
        self.data1 = []
        self.data_inf = []
        for i in range(self.num):
            cur_data = self.data[i]
            if cur_data != 'inf':
                self.data1.append(cur_data)
            else:
                self.data_inf.append(cur_data)

        self.data1.sort()
        self.num1 = len(self.data1)
        self.num_inf = len(self.data_inf)
    
    def median_index(self):
        if self.num%2 == 0:
            n_median1, n_median2 = self.num / 2 - 1, self.num / 2
        elif self.num%2 == 1:
             n_median1, n_median2 = (self.num - 1) / 2, (self.num - 1) / 2
        return  n_median1, n_median2

    def get_median(self):
        n_median1, n_median2 = self.median_index()
        if n_median2 > self.num1 - 1:
            return 'inf'
        else:
            return (self.data1[n_median1] + self.data1[n_median2]) / 2.0
    
    def fractile_index(self, fractile):
        
        for i in range(self.num):
            #print i*1.0/self.num
            if i*1.0/self.num < fractile and (i+1)*1.0/self.num >= fractile:
                break
        return i
    
    def get_fractile(self, fractile):
        ind = self.fractile_index(fractile)
        if ind > self.num1 - 1:
            return 'inf'
        else:
            return self.data1[ind]
        
    
class IDA_Single():

    def __init__(self, IM, DM, num=100, limit=0.1):
        self.limit=limit
        self.IM, self.DM = self.remove_collapse(IM, DM)
        self.f_DM = Spline_fit(self.IM, self.DM, extend=False)
        self.IM_min, self.IM_max = min(self.IM), max(self.IM)
        self.IM_interp, self.DM_interp = self.data_plot(num=num)

    def remove_collapse(self, x_data, y_data):
        is_outside = False
        x_new = np.array([])
        y_new = np.array([])
        for x, y in zip(x_data, y_data):
            if y!=-1:
                x_new = np.append(x_new, x)
                y_new = np.append(y_new, y)
            if y>=self.limit:
                is_outside = True
                break
        if is_outside:
            return x_new[:-1], y_new[:-1]
        else:
            return x_new, y_new
    
    def data_plot(self, num = 100):
        IM_data = np.linspace(self.IM_min, self.IM_max, num)
        DM_data = self.f_DM(IM_data)
        IM_data = np.append(IM_data, IM_data[-1])
        DM_data = np.append(DM_data, DM_data[-1]*100)
        return IM_data, DM_data

    def LS_DM(self, DM_target):
        for i in range(len(self.DM_interp)):
            DM_cur = self.DM_interp[i]
            if DM_cur > DM_target:
                break
        return self.IM_interp[i-1]

    def LS_tangent(self, tangent_target):
        K_init = (self.IM_interp[1] - self.IM_interp[0]) / (self.DM_interp[1] - self.DM_interp[0])
        for i in range(1, len(self.DM_interp)):
            K_cur = (self.IM_interp[i] - self.IM_interp[i-1]) / (self.DM_interp[i] - self.DM_interp[i-1])
            if abs(K_cur) <= tangent_target * K_init:
                break
        return self.IM_interp[i-1]


class IDA_Set():

    def __init__(self):
        self.num = 0
        self.IDA_list = []
        self.IM_max = 0.0
    
    def append(self, IDA_Single):
        self.IDA_list.append(IDA_Single)
        self.num += 1
        if self.IM_max <= IDA_Single.IM_max:
            self.IM_max = IDA_Single.IM_max

    def Stripe_Analysis(self, IM_level):
        data = []
        for IDA in self.IDA_list:
            data.append(IDA.f_DM(IM_level))
        cur_stripe = IDA_Stripe(data)
        return cur_stripe.median, cur_stripe.fractile_16, cur_stripe.fractile_84

    def Stats(self, IM_step = 0.1):
        DM_median, DM_16, DM_84 = np.array([]), np.array([]), np.array([])
        IM_levels = np.arange(0.0, self.IM_max, IM_step)
        for IM_level in IM_levels:
            median, f_16, f_84 = self.Stripe_Analysis(IM_level)
            DM_median = np.append(DM_median, median)
            DM_16 = np.append(DM_16, f_16)
            DM_84 = np.append(DM_84, f_84)

        cur_IM, cur_DM = self.remove_inf(IM_levels, DM_median)
        self.IDA_median = IDA_Single(cur_IM, cur_DM)
        cur_IM, cur_DM = self.remove_inf(IM_levels, DM_16)
        self.IDA_16 = IDA_Single(cur_IM, cur_DM)
        cur_IM, cur_DM = self.remove_inf(IM_levels, DM_84)
        self.IDA_84 = IDA_Single(cur_IM, cur_DM)
            

    def remove_inf(self, x_data, y_data):
        x_new = np.array([])
        y_new = np.array([])
        for x, y in zip(x_data, y_data):
            if y!='inf':
                x_new = np.append(x_new, x)
                y_new = np.append(y_new, y)
        return x_new, y_new.astype(x_new.dtype)
    
    def LS_DM(self, DM_target):
        data = []
        for IDA in self.IDA_list:
            data.append(IDA.LS_DM(DM_target))
        stripe_cur = IDA_Stripe(data)
        median, norm_sigma = stripe_cur.median, stripe_cur.norm_sigma
        return median, norm_sigma, data

    def LS_tangent(self, tangent_target):
        data = []
        for IDA in self.IDA_list:
            data.append(IDA.LS_tangent(tangent_target))
        stripe_cur = IDA_Stripe(data)
        median, norm_sigma = stripe_cur.median, stripe_cur.norm_sigma
        return median, norm_sigma, data
