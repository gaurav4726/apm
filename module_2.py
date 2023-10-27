import os
import pandas as pd
from logger import logging
from exceptions import CustomException
from marshmallow import Schema, fields
import sys
import config
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
import warnings
import pyarrow as pa
import pyarrow.parquet as pq
warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score, r2_score
import matplotlib.pyplot as plt
from helper import Module_2_Classification_Model,Module_2_Reg_Model

requirement_dict=config.requirement_dict

@dataclass
class Module2_TransformationConfig:
    output_table_module2_path: str=os.path.join('artifacts',"classification_model_ouput.csv")
    output_table_module2_path_parquet: str=os.path.join('artifacts',"classification_model_ouput.parquet")
    F1_score_path: str=os.path.join('artifacts',"F1_score.csv")

    regression_output_module2_path: str=os.path.join('artifacts',"regression_output_data.csv")
    regression_output_module2_path_parquet: str=os.path.join('artifacts',"regression_output_data.parquet")
    R2_score_path: str=os.path.join('artifacts',"R2_score.csv")

class Module2_model_DataTransformation:
    def __init__(self):
        self.data_transformation_config=Module2_TransformationConfig()  
    
    def initiate_classification_model_data_transformation(self,population_table_path,segment_table_path,
                                           treatment_table_path,outcome_table_path,eligibility_table_path):
        logging.info("Classification Model has started")
        population_table = pd.read_csv(population_table_path)
        segment_table = pd.read_csv(segment_table_path)
        treatment_table = pd.read_csv(treatment_table_path)  
        eligibility_table= pd.read_csv(eligibility_table_path)  
        outcome_table= pd.read_csv(outcome_table_path)  
        population_table['emp_id']=population_table['emp_id'].astype(str)
        segment_table['emp_id']=segment_table['emp_id'].astype(str)
        treatment_table['emp_id']=treatment_table['emp_id'].astype(str)
        eligibility_table['emp_id']=eligibility_table['emp_id'].astype(str)
        outcome_table['emp_id']=outcome_table['emp_id'].astype(str)
        

        output_table,F1_score,FIP=Module_2_Classification_Model(eligibility_table,treatment_table,population_table,segment_table)
        os.makedirs(os.path.dirname(self.data_transformation_config.output_table_module2_path),exist_ok=True)
        output_table.to_csv(self.data_transformation_config.output_table_module2_path,index=False,header=True)

        output_table_pa=pa.Table.from_pandas(output_table)
        os.makedirs(os.path.dirname(self.data_transformation_config.output_table_module2_path_parquet),exist_ok=True)
        pq.write_table(output_table_pa,self.data_transformation_config.output_table_module2_path_parquet)


        os.makedirs(os.path.dirname(self.data_transformation_config.F1_score_path),exist_ok=True)
        F1_score_df=pd.DataFrame([F1_score],columns=['F1_Score'])
        F1_score_df.to_csv(self.data_transformation_config.F1_score_path,index=False)
       
        F1_score_pa=pa.Table.from_pandas(F1_score_df)
        pq.write_table(F1_score_pa, 'artifacts/F1_score.parquet')

        FIP.plot(kind='bar', figsize=(10, 6), fontsize=14)
        plt.savefig('artifacts/Classifier_Feat_Imp.png',bbox_inches = 'tight')

        
        logging.info("Classification Model has Done") 
        return(self.data_transformation_config.output_table_module2_path,self.data_transformation_config.F1_score_path)
    

    
    def initiate_regression_model_data_transformation(self,population_table_path,segment_table_path,
                                           treatment_table_path,eligibility_table_path,outcome_table_path,requirement_dict):
         logging.info("Regression Model has started")
         population_table = pd.read_csv(population_table_path)
         segment_table = pd.read_csv(segment_table_path)
         treatment_table = pd.read_csv(treatment_table_path)  
         eligibility_table= pd.read_csv(eligibility_table_path)
         outcome_table= pd.read_csv(outcome_table_path)

         population_table['emp_id']=population_table['emp_id'].astype(str)
         segment_table['emp_id']=segment_table['emp_id'].astype(str)
         treatment_table['emp_id']=treatment_table['emp_id'].astype(str)
         eligibility_table['emp_id']=eligibility_table['emp_id'].astype(str)
         outcome_table['emp_id']=outcome_table['emp_id'].astype(str)


         regression_running_periods=requirement_dict['Performace_days']['Performace_Post_day']

         #####################

         for time_period in range(0,len((regression_running_periods))):
            outcome_table_filtered=outcome_table[outcome_table['date_diff']==regression_running_periods[time_period]]
            if outcome_table_filtered.shape[0]>10:
                output_table_pred=pd.DataFrame()
                R2_score=[]
                Feature_Importance=pd.DataFrame()
                #feature_labels={}
                performance_matrix=requirement_dict['input_dict']['performance_matrix']
                for performance_KPI in performance_matrix:
                    a,b,c,d=Module_2_Reg_Model(eligibility_table,treatment_table,population_table,segment_table,
                                                 outcome_table_filtered,performance_KPI)
                
                    output_table_pred=pd.concat([output_table_pred,b],axis=1)
                    R2_score.append(c)
                    FI_DF = pd.DataFrame(d, columns= ['Feat_Imp '+performance_KPI])
                    FI_DF.plot(kind='bar',  figsize=(10, 6), fontsize=14)
            
                    plt.savefig("artifacts/"+performance_KPI+str(regression_running_periods[time_period])+' Feat_Imp.png',bbox_inches = 'tight')
                    plt.close()
                    Feature_Importance = pd.concat([Feature_Importance,round(FI_DF,2)],axis=1)
                    #feature_labels[performance_KPI]=e
                output_table_pred=pd.concat([a,output_table_pred],axis=1)
                
                null_cols = output_table_pred.columns[output_table_pred.isnull().all()]
                output_table_pred.drop(columns=null_cols, inplace=True)
                null_col_list = null_cols.tolist()
                null_col_list_predictions = [s.replace(' predictions', '') for s in null_col_list]
                output_table_pred.drop(columns=null_col_list_predictions, inplace=True)
                null_col_list_pre = [s.replace(' predictions', '_pre') for s in null_col_list]
                output_table_pred.drop(columns=null_col_list_pre, inplace=True)
                
                path=self.data_transformation_config.regression_output_module2_path.split(".")[0]+\
                                                        "_"+str(regression_running_periods[time_period]) +"days.csv"
                path1=self.data_transformation_config.regression_output_module2_path_parquet.split(".")\
                                                        [0]+"_"+str(regression_running_periods[time_period])+"days.parquet"
                
                path2=self.data_transformation_config.R2_score_path.split(".")[0]+"_"+\
                                                    str(regression_running_periods[time_period])+"days.csv"
                path3=self.data_transformation_config.R2_score_path.split(".")[0]+"_"+\
                                                    str(regression_running_periods[time_period])+"days.parquet"

                os.makedirs(os.path.dirname(path),exist_ok=True)
                os.makedirs(os.path.dirname(path1),exist_ok=True)
                os.makedirs(os.path.dirname(path2),exist_ok=True)  


                output_table_pred.to_csv(path,index=False,header=True)

                output_table_pred_pa=pa.Table.from_pandas(output_table_pred)

                R2_score_df = pd.DataFrame(R2_score).T
                

                R2_score_df.columns=performance_matrix
                
                #null_col_list = [s.replace(' predictions', '') for s in null_col_list]
                R2_score_df.drop(columns=null_col_list_predictions, inplace=True)
                
                R2_score_df.to_csv(path2,index=False)
                R2_score_pa=pa.Table.from_pandas(R2_score_df)
                pq.write_table(output_table_pred_pa, path1)
                pq.write_table(R2_score_pa, path3)

                logging.info("Regression Model has Done of "+str(regression_running_periods[time_period])+"days")   


                
        #         # Change Regression output in Pyarrow
        #         output_table_pred_pyarrow=pa.Table.from_pandas(output_table_pred)
            



      
        # #  for time_period in range(0,len((regression_running_periods))):
        # #      outcome_table_filtered=outcome_table[outcome_table['date_diff']==regression_running_periods[time_period]]
        # #      if outcome_table_filtered.shape[0]>10:
        # #         output_table_pred=pd.DataFrame()
        # #         R2_score=[]
        # #         FIP=[]
        # #         performace_matrix=requirement_dict['input_dict']['performance_matrix']
        # #         for KPI in performace_matrix:
        # #             a,b,c,d=Module_2_Reg_Model(eligibility_table,treatment_table,population_table,segment_table,outcome_table_filtered,KPI)
        # #             output_table_pred=pd.concat([output_table_pred,b],axis=1)
        # #             R2_score.append(c)
        # #             FIP.append(d)
        # #         output_table_pred=pd.concat([a,output_table_pred],axis=1)

        #         path=self.data_transformation_config.regression_output_module2_path.split(".")[0]+"_"+str(regression_running_periods[time_period])+"days.csv"
        #         path1=self.data_transformation_config.regression_output_module2_path_parquet.split(".")\
        #                                                                              [0]+"_"+str(regression_running_periods[time_period])+"days.parquet"
        #         path2=self.data_transformation_config.R2_score_path.split(".")[0]+"_"+str(regression_running_periods[time_period])+"days.csv"
              
        #         os.makedirs(os.path.dirname(path),exist_ok=True)
        #         os.makedirs(os.path.dirname(path1),exist_ok=True)
        #         os.makedirs(os.path.dirname(path2),exist_ok=True)  

        #         R2_score_df = pd.DataFrame(R2_score).T
        #         R2_score_df.columns=performace_matrix
        #         R2_score_df.to_csv(path2,index=False)
        #         output_table_pred.to_csv(path,index=False,header=True)
        #         # Change Regression output in Pyarrow
        #         output_table_pred_pyarrow=pa.Table.from_pandas(output_table_pred)
        #         pq.write_table(output_table_pred_pyarrow,path1)
        #         logging.info("Regression Model has Done of "+str(regression_running_periods[time_period])+"days")   
            #  else:
            #     logging.info("Not enough data of "+str(regression_running_periods[time_period])+"days for running regression model") 
                 
                                
         return(self.data_transformation_config.regression_output_module2_path,self.data_transformation_config.R2_score_path)
    
    


    
