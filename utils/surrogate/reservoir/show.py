import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import model.multipolygon as pg


class ShowResults:
    def read_table(data_file_name: str, names = [], feather = False, delimiter='\t'):
        if feather:
            df = pd.read_feather(data_file_name)
        else:
            df = pd.read_csv(data_file_name, delimiter=delimiter) 
        
    
        data = {}
        for name in names:
            data[name] = []
    
        times = None
    
        for var_id, var_df in df.groupby('variant_id'):
            var_data = {}
            
            for name in names:
                var_data[name] = []
    
            for well_id, well_df in var_df.groupby('well_id'):
                for name in names:
                    vals = np.array(well_df[name].values)
                    var_data[name].append(vals)
                
                times = np.array(well_df['date'].values)
    
            for name in names:
                var_data[name] = np.array(var_data[name])
    
            for name in names:
                data[name].append(var_data[name])
    
        for name in names:
            vals = np.array(data[name])
            vals = np.transpose(vals, (2, 0, 1))
            data[name] = vals
    
        return data, times
    
    def read_results(data_file_name):
        names = [
            'bhp',
            'sat',
            'prod',
            'pressure',
            'saturation',
        ]
    
        return ShowResults.read_table(data_file_name, names, True)
    
    def read_input(data_file_name, delimiter='\t'):
        names = [
            'bhp',
            'sat',
            'prod',
            'wf'
        ]
    
        return ShowResults.read_table(data_file_name, names, False, delimiter)
    
    def show_data(self, bhp_o, bhp_p, pres_p, prod_o, prod_p, prod_o_o, prod_o_p, sufix):
        fig, ax = plt.subplots(3, figsize=(19,10))

        sat_o = np.where(prod_o > 1e-6, 1.0 - prod_o_o / prod_o, 1.0)
        sat_p = np.where(prod_p > 1e-6, 1.0 - prod_o_p / prod_p, 1.0)

        ax[0].plot(self.times, bhp_o, label = 'bhp_orig_' + sufix)
        ax[0].plot(self.times, bhp_p, label = 'bhp_pred_' + sufix)
        ax[0].plot(self.times, pres_p, label = 'pres_pred_' + sufix)
        ax[0].legend()

        ax[1].plot(self.times, prod_o, label = 'prod_orig_' + sufix)
        ax[1].plot(self.times, prod_p, label = 'prod_pred_' + sufix)
        ax[1].plot(self.times, prod_o_o, label = 'prod_o_orig_' + sufix)
        ax[1].plot(self.times, prod_o_p, label = 'prod_o_pred_' + sufix)
        ax[1].legend()
        
        ax[2].plot(self.times, sat_o, label = 'sat_orig_' + sufix)
        ax[2].plot(self.times, sat_p, label = 'sat_pred_' + sufix)
        ax[2].legend()
    
    def show_well(self, var_id, well_id):
        bhp_o = self.bhp_orig[:, var_id, well_id]
        bhp_p = self.bhp_pred[:, var_id, well_id]
        pres_p = self.pres_pred[:, var_id, well_id]

        prod_o = self.prod_orig[:, var_id, well_id]
        prod_p = self.prod_pred[:, var_id, well_id]
        
        prod_o_o = self.prod_o_orig[:, var_id, well_id]
        prod_o_p = self.prod_o_pred[:, var_id, well_id]

        self.show_data(bhp_o, bhp_p, pres_p, prod_o, prod_p, prod_o_o, prod_o_p, str(well_id))
        
        os.makedirs(self.result_folder + f'/imgs/var_{var_id}', exist_ok=True)
        plt.savefig(self.result_folder + f'/imgs/var_{var_id}/well_{well_id + 1000}.png')
        print('saved: ' + self.result_folder + f'/imgs/var_{var_id}/well_{well_id + 1000}.png')
        #plt.show()
        plt.clf()
        plt.cla()
        plt.close()
    
    def show_well_mean(self, well_id):
        bhp_o = self.bhp_orig[:, :, well_id].mean(axis=1)
        bhp_p = self.bhp_pred[:, :, well_id].mean(axis=1)
        pres_p = self.pres_pred[:, :, well_id].mean(axis=1)
        
        prod_o = self.prod_orig[:, :, well_id].mean(axis=1)
        prod_p = self.prod_pred[:, :, well_id].mean(axis=1)
        
        prod_o_o = self.prod_o_orig[:, :, well_id].mean(axis=1)
        prod_o_p = self.prod_o_pred[:, :, well_id].mean(axis=1)

        self.show_data(bhp_o, bhp_p, pres_p, prod_o, prod_p, prod_o_o, prod_o_p, str(well_id))
        
        os.makedirs(self.result_folder + f'/imgs', exist_ok=True)
        plt.savefig(self.result_folder + f'/imgs/well_{well_id}.png')
        print('saved: ' + self.result_folder + f'/imgs/well_{well_id}.png')
        #plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    def show_var_mean(self, var_id):
        bhp_o = self.bhp_orig[:, var_id, :].mean(axis=1)
        bhp_p = self.bhp_pred[:, var_id, :].mean(axis=1)
        pres_p = self.pres_pred[:, var_id, :].mean(axis=1)

        prod_o = self.prod_orig[:, var_id, :].mean(axis=1)
        prod_p = self.prod_pred[:, var_id, :].mean(axis=1)
        
        prod_o_o = self.prod_o_orig[:, var_id, :].mean(axis=1)
        prod_o_p = self.prod_o_pred[:, var_id, :].mean(axis=1)

        fig, ax = plt.subplots(3, figsize=(19,10))

        dob_o = np.cumsum(self.prod_orig[:, var_id, :] * self.wf_hist[:, var_id, :], axis=0).mean(axis=1)
        dob_p = np.cumsum(self.prod_pred[:, var_id, :] * self.wf_hist[:, var_id, :], axis=0).mean(axis=1)
        dob_o_o = np.cumsum(self.prod_o_orig[:, var_id, :] * self.wf_hist[:, var_id, :], axis=0).mean(axis=1)
        dob_o_p = np.cumsum(self.prod_o_pred[:, var_id, :] * self.wf_hist[:, var_id, :], axis=0).mean(axis=1)

        sufix = 'var'+ str(var_id)

        ax[0].plot(self.times, bhp_o, label = 'bhp_orig_' + sufix)
        ax[0].plot(self.times, bhp_p, label = 'bhp_pred_' + sufix)
        ax[0].plot(self.times, pres_p, label = 'pres_pred_' + sufix)
        ax[0].legend()

        ax[1].plot(self.times, prod_o, label = 'prod_orig_' + sufix)
        ax[1].plot(self.times, prod_p, label = 'prod_pred_' + sufix)
        ax[1].plot(self.times, prod_o_o, label = 'prod_o_orig_' + sufix)
        ax[1].plot(self.times, prod_o_p, label = 'prod_o_pred_' + sufix)
        ax[1].legend()
        
        ax[2].plot(self.times, dob_o, label = 'dob_orig_' + sufix)
        ax[2].plot(self.times, dob_p, label = 'dob_pred_' + sufix)
        ax[2].plot(self.times, dob_o_o, label = 'dob_o_orig_' + sufix)
        ax[2].plot(self.times, dob_o_p, label = 'dob_o_pred_' + sufix)
        ax[2].legend()

        os.makedirs(self.result_folder + f'/imgs/var_{var_id}', exist_ok=True)
        plt.savefig(self.result_folder + f'/imgs/var_{var_id}/total.png')
        print('saved: ' + self.result_folder + f'/imgs/var_{var_id}/total.png')
        #plt.show()
        plt.clf()
        plt.cla()
        plt.close()
    
    def show_total(self):
        bhp_o = self.bhp_orig.mean(axis=(1,2))
        bhp_p = self.bhp_pred.mean(axis=(1,2))
        pres_p = self.pres_pred.mean(axis=(1,2))
        
        prod_o = self.prod_orig.mean(axis=1).sum(axis=1)
        prod_p = self.prod_pred.mean(axis=1).sum(axis=1)
        
        prod_o_o = self.prod_o_orig.mean(axis=1).sum(axis=1)
        prod_o_p = self.prod_o_pred.mean(axis=1).sum(axis=1)

        self.show_data(bhp_o, bhp_p, pres_p, prod_o, prod_p, prod_o_o, prod_o_p, "tot")
                
        os.makedirs(self.result_folder + f'/imgs', exist_ok=True)
        plt.savefig(self.result_folder + f'/imgs/total.png')
        print('saved: ' + self.result_folder + f'/imgs/total.png')
        #plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    def save_images(self, test_data_file, result_folder, delimiter='\t'):
        self.result_folder = result_folder
        data_pred, self.times = ShowResults.read_results(result_folder + '/result_valid.feth')
        data_orig, _ = ShowResults.read_input(test_data_file, delimiter)

        self.wf_hist = data_orig['wf']
        self.bhp_orig = data_orig['bhp']
        self.bhp_pred = data_pred['bhp']
        self.pres_pred = data_pred['pressure'] #/ 0.02284190467700601 + 143.34223179628646        

        self.prod_orig = data_orig['prod']
        self.prod_pred = data_pred['prod']

        self.sat_orig = data_orig['sat']
        self.sat_pred = data_pred['sat']

        self.bhp_orig = np.where(np.isnan(self.bhp_orig), self.bhp_pred, self.bhp_orig)
        self.wf_hist = np.where(np.abs(self.prod_orig) < 1e-6, 0.0, self.wf_hist)

        print(self.prod_orig.shape)
        self.prod_o_orig = self.prod_orig * (1.0 - self.sat_orig)
        self.prod_o_pred = self.prod_pred * (1.0 - self.sat_pred)

        self.show_total()
        #exit()

        var_count = self.bhp_orig.shape[1]
        well_count = self.bhp_orig.shape[2]

        for well_id in range(well_count):
            self.show_well_mean(well_id)
            
            for var_id in range(var_count):    
                self.show_well(var_id, well_id)
        for var_id in range(var_count): 
            self.show_var_mean(var_id)

def save_images(test_data_file, result_folder, delimiter='\t'):
    ShowResults().save_images(test_data_file, result_folder, delimiter)
    