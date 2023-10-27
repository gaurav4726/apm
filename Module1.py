import os
import pandas as pd
from logger import logging
from exceptions import CustomException
import sys
from marshmallow import Schema, fields
import sys
from dataclasses import dataclass
from helper import usecase1,usecase2,usecase3,usecase4,usecase5,usecase6,usecase7,usecase8,usecase9,usecase10,label_encoder_segment
import config
import pyarrow as pa
from sklearn.preprocessing import LabelEncoder

requirement_dict=config.requirement_dict

# Import Data

@dataclass
class DataTransformationConfig:
    module1_population_table_path: str=os.path.join('artifacts',"population_table_module1.csv")
    module1_treatment_table_path: str=os.path.join('artifacts',"treatment_table_module1.csv")
    module1_segment_table_path: str=os.path.join('artifacts',"segement_table_module1.csv")
    module1_outcome_table_path: str=os.path.join('artifacts',"outcome_table_module1.csv")
    module1_eligibility_table_path: str=os.path.join('artifacts',"eligibility_table_module1.csv")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def initiate_data_transformation(self,population_table_path,segment_table_path,treatment_table_path):    
        try:
            error=[]
            logging.info("Logging has started")
            #population_table, segment_table, treatment_table = import_data()

            population_table = pd.read_csv(population_table_path)
            segment_table = pd.read_csv(segment_table_path)
            treatment_table = pd.read_csv(treatment_table_path)

            popu_table_format=usecase1(requirement_dict,population_table)
            usecase2(requirement_dict)
            usecase3(requirement_dict,population_table)
            treatment_table,population_table=usecase4(requirement_dict,treatment_table,population_table)
            treatment_table,population_table=usecase5(requirement_dict,treatment_table,population_table,popu_table_format)
            treatment_table=usecase6(requirement_dict,treatment_table)
            treatment_table,population_table=usecase8(requirement_dict,treatment_table,population_table,popu_table_format)

            treatment_table,population_table=usecase7(requirement_dict,treatment_table,population_table,popu_table_format)
            treatment_table,population_table=usecase9(requirement_dict,treatment_table,population_table)
            outcome_table,population_table=usecase10(requirement_dict,treatment_table,population_table,popu_table_format)
            id_mapping=treatment_table[['treatment_id',"treatment_label_taken"]].drop_duplicates()
            id_mapping=id_mapping.dropna()
            id_mapping.to_csv("artifacts/treatmnent_id_mapping_with_names.csv",index=False)
            treatment_table['Control/Test_group'] = 1
            treatment_table.loc[treatment_table['end_date_of_treatment_taken'].isnull(), 'Control/Test_group'] = 0 
            eligibility_table=treatment_table[['emp_id','treatment_id','treatment_allocated_date','treatment_allocated_end_date','Control/Test_group']]  
            col=["emp_id", 'treatment_label_taken',
                'start_date_of_treatment_taken', 'end_date_of_treatment_taken']
            treatment_table=treatment_table[col].dropna()
            os.makedirs(os.path.dirname(self.data_transformation_config.module1_population_table_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.module1_segment_table_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.module1_treatment_table_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.module1_outcome_table_path),exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.module1_eligibility_table_path),exist_ok=True)

            segment_table,segment_labels=label_encoder_segment(segment_table)
            segment_labels.to_csv("artifacts/segment_labels.csv",index=False)


            label_encoders = {}
            label_encoder = LabelEncoder()
            eligibility_table["treatment_id"] = label_encoder.fit_transform(eligibility_table["treatment_id"])
            label_encoders["treatment_id"] = {'treatment_id': label_encoder, 'mapping': \
                                    dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}
            
            
            treatment_id_mapping = pd.DataFrame(label_encoders["treatment_id"]['mapping'],index=['mapping']).T.reset_index()
            treatment_id_mapping.columns=['treatment_id','mapped_value']
          
            treatment_id_mapping.to_csv("artifacts/treatment_id_mapping.csv")

            population_table=population_table.merge(treatment_id_mapping,on='treatment_id',how='left')
            del population_table['treatment_id']
            population_table.rename(columns={'mapped_value':'treatment_id'},inplace=True)

            outcome_table=outcome_table.merge(treatment_id_mapping,on='treatment_id',how='left')
            del outcome_table['treatment_id']
            outcome_table.rename(columns={'mapped_value':'treatment_id'},inplace=True)




            population_table.to_csv(self.data_transformation_config.module1_population_table_path,index=False,header=True)
            segment_table.to_csv(self.data_transformation_config.module1_segment_table_path,index=False,header=True)
            treatment_table.to_csv(self.data_transformation_config.module1_treatment_table_path,index=False,header=True)
            outcome_table.to_csv(self.data_transformation_config.module1_outcome_table_path,index=False,header=True)
            eligibility_table.to_csv(self.data_transformation_config.module1_eligibility_table_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                    self.data_transformation_config.module1_population_table_path,
                    self.data_transformation_config.module1_segment_table_path,
                    self.data_transformation_config.module1_treatment_table_path,
                    self.data_transformation_config.module1_outcome_table_path,
                    self.data_transformation_config.module1_eligibility_table_path)
        
        except Exception as e:
                raise CustomException(e,sys)
            

    

