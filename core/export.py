import numpy as np
import pandas as pd
import pickle
import pathlib

class result_exporter(object):
    '''
    Class to export results
    '''

    def __init__(self):
        
        self.it_results = self.init_dataframes()
        
    def init_dataframes(self):
        
        iterative_results = dict()
        iterative_results['general'] = pd.DataFrame()
        iterative_results['run_times'] = pd.DataFrame()
        iterative_results['performance'] = pd.DataFrame()
        iterative_results['model_size'] = pd.DataFrame()
        
        return iterative_results
    
    def create_writer(self, Ab, N):
        
        # Save case-specific data in Excel
        output_file = Ab.setup.directories['outputFcase'] + \
            Ab.setup.time['datetime'] + '_N='+str(N)+'_data_export.xlsx'
        
        # Create a Pandas Excel writer using XlsxWriter as the engine
        self.writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

        return self.writer
        
    def add_results(self, Ab, optimal_policy, model_size, case_id):
        # Write model size results to Excel
        model_size_df = pd.DataFrame(model_size, index=[case_id])
        model_size_df.to_excel(self.writer, sheet_name='Model size')
        
        # Load data into dataframes
        policy_df   = pd.DataFrame( optimal_policy,
        columns=range(Ab.partition['nr_regions']), index=range(1 if Ab.N == np.inf else  Ab.N)).T

        reward_df   = pd.Series( Ab.results['optimal_reward'], 
         index=range(Ab.partition['nr_regions'])).T
        
        # Write dataframes to a different worksheet
        policy_df.to_excel(self.writer, sheet_name='Optimal policy')
        reward_df.to_excel(self.writer, sheet_name='Optimal reward')
    
    def add_to_df(self, df, key):
        
        self.it_results[key] = pd.concat([self.it_results[key], df], axis=0)
        
    def save_to_excel(self, output_file):
        
        # Create a Pandas Excel writer using XlsxWriter as the engine
        writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
        
        for key,df in self.it_results.items():
            df.to_excel(writer, sheet_name=str(key))
        
        # Close the Pandas Excel writer and output the Excel file.
        writer.close()



def pickle_results(Ab):
    '''
    Export the raw output data to a pickle file
    '''

    out_folder = Ab.setup.directories['outputF']
    out_file   = 'data_dump.p'
    out_path   = pathlib.Path(out_folder, out_file) 

    data = {
        'model': Ab.model,
        'setup': Ab.setup,
        'spec': Ab.spec,
        'results': Ab.results,
        'regions': Ab.partition['R'],
        'goal_regions': Ab.partition['goal'],
        'critical_regions': Ab.partition['critical'],
        'args': Ab.args
    }

    if hasattr(Ab, 'mc'):
        data['mc'] = Ab.mc

    pickle.dump(data, open( out_path, "wb" ) )