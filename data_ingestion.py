import os
import sys
from exceptions import CustomException
from logger import logging
import pandas as pd
from dataclasses import dataclass
from Module1 import DataTransformation
from module_2 import Module2_model_DataTransformation
import config
import pyarrow as pa
import warnings
warnings.filterwarnings("ignore")

requirement_dict=config.requirement_dict

def import_data():
    population_table = pd.read_csv("population_table.csv")
    segment_table = pd.read_csv("segment_table.csv")
    treatment_table = pd.read_csv("treatment_table.csv")
    return population_table, segment_table, treatment_table

@dataclass
class DataIngestionConfig:
    population_table_path: str=os.path.join('artifacts',"population_table.csv")
    segment_table_path: str=os.path.join('artifacts',"segemnt_table.csv")
    treatment_table_path: str=os.path.join('artifacts',"treatment_table.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            population_table, segment_table, treatment_table =import_data()

            logging.info('Project Name is:{}'.format(requirement_dict['input_dict']['project_name']))
            logging.info('Read  Successfully the dataset as dataframe')
            population_table = population_table.rename(columns=requirement_dict['population_table_column_mapping'])   
            treatment_table = treatment_table.rename(columns=requirement_dict['treatment_table_columns_mapping'])
            segment_table = segment_table.rename(columns=requirement_dict['segment_table_column_mapping'])
            segment_table["emp_id"]=segment_table["emp_id"].astype(str)
            Required_Col = ['emp_id'] +list(requirement_dict['segment_variable_columns'].values())
            segment_table = segment_table[Required_Col]
            population_table['emp_id']=population_table['emp_id'].astype(str)
            treatment_table['emp_id']=treatment_table['emp_id'].astype(str)
            treatment_table['treatment_id']=treatment_table['treatment_id'].astype(str)
            population_table['start_date']=pd.to_datetime(population_table['start_date'])
            population_table['end_date']=pd.to_datetime(population_table['end_date'])
            treatment_table['treatment_allocated_date']=pd.to_datetime(treatment_table['treatment_allocated_date'])
            treatment_table['treatment_allocated_end_date']=pd.to_datetime(treatment_table['treatment_allocated_end_date'])
            pp=population_table[['emp_id',"treatment_id"]].drop_duplicates()
            pp['new']=1
            treatment_table=treatment_table.merge(pp,on=['emp_id',"treatment_id"],how='outer')
            del treatment_table['treatment_allocated_date']
            del treatment_table['treatment_allocated_end_date']
            del treatment_table['new']
            tt=pd.DataFrame()
            tt['treatment_allocated_date'] = requirement_dict['input_dict']['treatment_start_date']
            tt['treatment_allocated_end_date'] = requirement_dict['input_dict']['treatment_end_date']
            tt['treatment_id'] = requirement_dict['input_dict']['treatment_name'] 
            treatment_table=treatment_table.merge(tt, on='treatment_id',how='left')

            logging.info('Inital Data Transformation done ')

            os.makedirs(os.path.dirname(self.ingestion_config.population_table_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.segment_table_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.treatment_table_path),exist_ok=True)

            population_table.to_csv(self.ingestion_config.population_table_path,index=False,header=True)
            segment_table.to_csv(self.ingestion_config.segment_table_path,index=False,header=True)
            treatment_table.to_csv(self.ingestion_config.treatment_table_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.population_table_path,
                self.ingestion_config.segment_table_path,
                self.ingestion_config.treatment_table_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    population_table,segment_table,treatment_table=obj.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    module1_population_table,module1_segment_table,module1_treatment_table,\
    module1_outcome_table,module1_eligibility_table=\
    data_transformation.initiate_data_transformation(population_table,segment_table,treatment_table)


    model_transformation=Module2_model_DataTransformation()
    model_transformation.initiate_classification_model_data_transformation(module1_population_table,
                          module1_segment_table,module1_treatment_table,module1_outcome_table,
                        module1_eligibility_table)

    requirement_dict=config.requirement_dict
    model_transformation.initiate_regression_model_data_transformation(module1_population_table,
                        module1_segment_table,module1_treatment_table,module1_eligibility_table,module1_outcome_table,requirement_dict)

