import pandas as pd             # Import Pandas to store data in frames
import pickle
import pathlib

class result_exporter(object):
    
    def __init__(self):
        
        self.it_results = self.init_dataframes()
        
    def init_dataframes(self):
        
        iterative_results = dict()
        iterative_results['general'] = pd.DataFrame()
        iterative_results['run_times'] = pd.DataFrame()
        iterative_results['performance'] = pd.DataFrame()
        iterative_results['model_size'] = pd.DataFrame()
        
        return iterative_results
    
    def create_writer(self, Ab, model_size, case_id, N):
        
        # Save case-specific data in Excel
        output_file = Ab.setup.directories['outputFcase'] + \
            Ab.setup.time['datetime'] + '_N='+str(N)+'_data_export.xlsx'
        
        # Create a Pandas Excel writer using XlsxWriter as the engine
        writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
        
        # Write model size results to Excel
        model_size_df = pd.DataFrame(model_size, index=[case_id])
        model_size_df.to_excel(writer, sheet_name='Model size')
        
        # Load data into dataframes
        policy_df   = pd.DataFrame( Ab.results['optimal_policy'], 
         columns=range(Ab.partition['nr_regions']), index=range(Ab.N)).T
        reward_df   = pd.Series( Ab.results['optimal_reward'], 
         index=range(Ab.partition['nr_regions'])).T
        
        # Write dataframes to a different worksheet
        policy_df.to_excel(writer, sheet_name='Optimal policy')
        reward_df.to_excel(writer, sheet_name='Optimal reward')
        
        return writer
    
    def add_to_df(self, df, key):
        
        self.it_results[key] = pd.concat([self.it_results[key], df], axis=0)
        
    def save_to_excel(self, output_file):
        
        # Create a Pandas Excel writer using XlsxWriter as the engine
        writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
        
        for key,df in self.it_results.items():
            df.to_excel(writer, sheet_name=str(key))
        
        # Close the Pandas Excel writer and output the Excel file.
        writer.save()

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
        'args': Ab.args
    }

    if hasattr(Ab, 'mc'):
        data['mc'] = Ab.mc

    pickle.dump(data, open( out_path, "wb" ) )