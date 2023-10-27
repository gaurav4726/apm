import os
import pandas as pd
from logger import logging
from exceptions import CustomException
import sys
from marshmallow import Schema, fields
import sys
import config
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from xgboost import XGBClassifier,XGBRegressor
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import pyarrow as pa
from xgboost import XGBRFRegressor
import requests, json
from sklearn.metrics import f1_score, r2_score
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
#!pip install bayesian-optimization
from skopt import BayesSearchCV

def custom_mapping(value):
    if 85 <= value <= 95:
        return 90
    elif 25 <= value <= 35:
        return 30
    elif 55 <= value <= 65:
        return 60
    else:
        return value


def label_encoder_segment (segment_table):
    # Create a dictionary to store the mappings
    label_encoders = {}

    # Apply label encoding to the original DataFrame
    try:
        columns_to_encode = segment_table.select_dtypes(include=['object']).columns.drop('emp_id')
    except:
        columns_to_encode = segment_table.select_dtypes(include=['object'])

    for column in columns_to_encode:
        label_encoder = LabelEncoder()
        segment_table[column] = label_encoder.fit_transform(segment_table[column])
        # Store the label encoder and its mapping
        label_encoders[column] = {'encoder': label_encoder, 'mapping': dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}

    segment_labels=pd.DataFrame()
    for col in label_encoders.keys():
        df = pd.DataFrame(label_encoders[col]['mapping'],index=['Segment']).T.reset_index()
        df['Segment_Name']=col
        segment_labels=pd.concat([segment_labels,df],axis=0)
    segment_labels.columns=['Segments',"Segment","Segment_Name"]

    return segment_table,segment_labels

requirement_dict=config.requirement_dict
error=[]

def agg_kpi(table):
    population_table=table
    agg=pd.DataFrame()
    for item1, item2 in zip(requirement_dict['input_dict']['performance_matrix'],
                            requirement_dict['input_dict']['agg_level_performance_Matrix']):

        pp=population_table.groupby(['emp_id']).agg(item1=(str(item1),str(item2))).reset_index()
        pp[item1]=item1
        agg=pd.concat([agg,pp],axis=0)
    column=requirement_dict['input_dict']['performance_matrix']
    agg['kpi'] = agg.apply(lambda row: next((row[col] for col in column\
                                                if not pd.isnull(row[col])), np.nan), axis=1)  
    agg=agg[['emp_id','item1','kpi']]
    agg_pivot = agg.pivot(index='emp_id', columns='kpi', values='item1').reset_index()
    population_table['date']=pd.to_datetime(population_table['date'])   
    popu_date=population_table.groupby(["emp_id"]).agg(start_date=("date","min"),
                                            end_date=("date","max")).reset_index()
    table=agg_pivot.merge(popu_date,on='emp_id',how='left')

    return table

def usecase1(requirement_dict,population_table):
# Change Name of Population Table
# First need to change emp_id and date column name , so we can check data is daily or aggreagted 


    popu_table_format="agg"
    popu=requirement_dict["population_table_column_mapping"]

    try:
        desired_value = ['emp_id', 'date']
        result_dict = {}
        for key, value in popu.items():
            if value in desired_value or key == 'date':
                result_dict[key] = value
        population_table = population_table.rename(columns=result_dict)        
        ## Now Check Do we have daily data or aggregated data and if daily wise convert it into aggregated 
        popu_agg=population_table.groupby(['emp_id',"treatment_id"]).agg(count=("emp_id","count")).reset_index()
        lent =len(requirement_dict['Performace_days']['Performace_Pre_day']+\
                requirement_dict['Performace_days']['Performace_Post_day'])
        #if popu_agg[popu_agg['count']>2].shape[0]>20: # >2 because pre and post data for 1 employess 
        lent=20
        if popu_agg[popu_agg['count']>lent].shape[0]>=1: # >2 because pre and post data for 1 employess 
            popu_table_format="daily"
            logging.info("Usecase:1 Pass, Provided Data is daily format ")
        logging.info("Usecase:1 Pass, Provided Data is Aggragated format ")    
    except Exception as e:
        logging.info("Usecase 1 : Failed")
        raise CustomException(e,sys)

    return popu_table_format

def usecase2(requirement_dict): 
    try:
        if len(requirement_dict["input_dict"]['performance_matrix'])==0:
            logging.info("BASIC SANITY CHECK - Use case 2 failed: Missing performance matrix")
            raise Exception("BASIC SANITY CHECK - Use case 2 failed: Missing performance matrix")
        else:
            logging.info("Usecase 2 Pass , All Performance Matrix Found")
        
    except Exception as e:
        logging.info("Usecase 2 : Failed",f"Error occurred: {str(e)}")
        error.append(f"Error occurred: {str(e)}")  

def usecase3(requirement_dict,population_table):
    elements_to_check = requirement_dict["input_dict"]['performance_matrix']
    for i in [population_table]:
        target_list = list(i.columns)
        try:
            missing_attribute = [element for element in elements_to_check if element not in target_list]
            if len(missing_attribute) > 0:
                logging.info("BASIC SANITY CHECK - Use case 3 failed: Missing performance matrix data")
                raise Exception("BASIC SANITY CHECK - Use case 3 failed: Missing performance matrix data")
            else:
                logging.info("Usecase 3 Pass , All Performance Matrix Data Found")
        except Exception as e:
            logging.info("Usecase 3 : Failed",f"Error occurred: {str(e)}")
            error.append(f"Error occurred: {str(e)}")

def usecase4(requirement_dict,treatment_table,population_table):
    try:
        treatment_table = treatment_table.rename(columns=requirement_dict["treatment_table_columns_mapping"])
        logging.info("Usecase 4 : Pass , All treatment_table column renamed")
    except Exception as e:
        logging.info("Usecase 4 : Failed due to treatment_table"+ str(e))
        error.append("Correct Column Mapping Not Provided in Treatment _Table: " + str(e))

    try:
        population_table = population_table.rename(columns=requirement_dict["population_table_column_mapping"])
        logging.info("Usecase 4 : Pass, All population_table column renamed")
    except Exception as e:
        logging.info("Usecase 4 : Failed due to population_table"+ str(e))
        error.append("Correct Column Mapping Not Provided in Population_Table: " + str(e))

    return treatment_table,population_table

def usecase5(requirement_dict,treatment_table,population_table,popu_table_format):
    try:
        Required_Columns = ['emp_id', 'treatment_allocated_date', 'treatment_allocated_end_date', 
                    'treatment_id', 'treatment_label_taken',
                    'start_date_of_treatment_taken', 'end_date_of_treatment_taken']
        treatment_table = treatment_table[Required_Columns]
        logging.info("Usecase 5 : Pass, Unwanted Columns Removed from treatment_table")

        Required_Col = ['emp_id', 'start_date', 'end_date',"treatment_id"] +\
                    requirement_dict["input_dict"]['performance_matrix']+\
                        requirement_dict['input_dict']['population_table_variable_for_model']
        
        if popu_table_format=="daily":
            Required_Col = ['emp_id', 'date',"treatment_id"] +\
                requirement_dict["input_dict"]['performance_matrix']+\
                requirement_dict['input_dict']['population_table_variable_for_model']
        
        Required_Col = [item for item in Required_Col if item.strip()]
        population_table = population_table[Required_Col]
        logging.info("Usecase 5 : Pass, Unwanted Columns Removed from population_table")


    except Exception as e:
        logging.info("Usecase 5 : Failed,Columns not Removed"+ str(e))
        error.append("Unwanted Columns not Removed: " + str(e))

    return treatment_table,population_table

def usecase6(requirement_dict,treatment_table):
    try:
        treatment_name = requirement_dict["input_dict"]['treatment_name']
        data = pd.DataFrame()
        error = []
        for i in treatment_name:
            dt = treatment_table[treatment_table['treatment_id'] == i]
            if dt.empty or dt.shape[0] == 0 or dt.shape[1] == 0:
                    raise Exception("Please check your treatment ID or data.")
            data = pd.concat([data, dt], axis=0).drop_duplicates().reset_index(drop=True)  
        logging.info("Usecase 6 :Pass, No Any Redundancy Found in treatment_id") 
    except Exception as e:
        logging.info("Usecase 6: Failed,Redundant treatment id Found in treatment_id_taken"+ str(e))
        error.append("Failed,Redundancy Found in treatment_id_taken: " + str(e))

    return data

def usecase7(requirement_dict,treatment_table,population_table,popu_table_format):
    try:
        # Create a sample DataFrame
        Required_Columns = ['emp_id', 'treatment_allocated_date', 'treatment_allocated_end_date', 
                        'treatment_id', 'treatment_label_taken',
                        'start_date_of_treatment_taken', 'end_date_of_treatment_taken']
        df = treatment_table[Required_Columns]

        # Define a schema class using Marshmallow
        class MySchema(Schema):
            emp_id = fields.String()
            treatment_allocated_date = fields.DateTime()
            treatment_allocated_end_date = fields.DateTime(data_key="treatment_allocated_end_date")
            treatment_id = fields.String()
            treatment_label_taken = fields.String()
            start_date_of_treatment_taken = fields.DateTime(data_key="start_date_of_treatment_taken")
            end_date_of_treatment_taken = fields.DateTime(data_key="end_date_of_treatment_taken")

        schema = MySchema()# Instantiate the schema
        row_index = 1# Define the row index to check
        row = df.iloc[row_index]# Get the specific row from the DataFrame
        mismatched_columns = []# Initialize a list to store mismatched column names
        # Validate each column value against the corresponding field in the schema
        for column_name, value in row.items():
            try:
                schema.fields[column_name].deserialize(value)
            except Exception:
                mismatched_columns.append(column_name)      
                
        #### Correct the Schema
        dates_column = ['treatment_allocated_date', 'treatment_allocated_end_date',
                        'start_date_of_treatment_taken', 'end_date_of_treatment_taken']

        for column in mismatched_columns:
            if column in dates_column:
                treatment_table[column] = pd.to_datetime(treatment_table[column])  
                
        str_column = ['treatment_id','treatment_label_taken','emp_id' ]

        for column in mismatched_columns:
            if column in str_column:treatment_table[column] = treatment_table[column].astype(str)         
        
        logging.info("Usecase 7 : Pass,Data Type changed for population_Table ")    

        # Performance Table check 
        elements_to_check = requirement_dict["input_dict"]['performance_matrix']
        for perfor_column in elements_to_check:
            population_table[perfor_column] = population_table[perfor_column].astype(int)
            
        if popu_table_format=="daily":  
            population_table['date']=pd.to_datetime(population_table['date'])
        else:     
            population_table['start_date']=pd.to_datetime(population_table['start_date'])
            population_table['end_date']=pd.to_datetime(population_table['end_date'])
            population_table['emp_id']=population_table['emp_id'].astype(str)   

        logging.info("Usecase 7 : Pass,Data Type changed for population_Table ")    

    except Exception as e:
        logging.info("Usecase 7 : Failed,Data Type not changed "+ str(e))
        error.append("Usecase 7: Failed, Data Type not changed : " + str(e))      

    return treatment_table,population_table

def usecase8(requirement_dict,treatment_table,population_table,popu_table_format):
    try:
        elements_to_check = requirement_dict["input_dict"]['performance_matrix']
        population_table_1 = population_table.dropna(subset=elements_to_check) # Drop Missing rows of Performance Matrix columns
        diff=population_table.shape[0]-population_table_1.shape[0]
        if diff>0:
            error.append(f"Missing rows found in Performance Table . Diff: {diff}")
            logging.info(f"UseCase 8 :Missing rows found in Performance Table . Diff: {diff}")

        if popu_table_format=="daily":population_table_2 = population_table_1.dropna(subset=['emp_id','date'])  # Drop Missing row of Employee Columns
        else:population_table_2 = population_table_1.dropna(subset=['emp_id','start_date','end_date'])  # Drop Missing row of Employee Columns
        diff=population_table_1.shape[0]-population_table_2.shape[0]
        if diff>0:
            error.append(f"Missing rows found in Employess/date . Diff: {diff}") 
            logging.info(f"Missing rows found in Employess/date . Diff: {diff}")

        population_table=population_table_2
        logging.info("UseCase 8 : Pass,population table missing values treatment done")

        #Treatment_Table Missing Value Check   
        columns_to_fill = ['treatment_allocated_date','treatment_allocated_end_date', 'treatment_id'] # Fill Na with Mode
        treatment_table=treatment_table.dropna(subset=columns_to_fill)
        
        
        logging.info("UseCase 8 : Pass,Treatament table missing values treatment done")

    except Exception as e:
        logging.info("Usecase 8 : Failed, Missing Value treatment not done"+ str(e))
        error.append("UUsecase 8 : Failed, Missing Value treatment not done" + str(e))   

    return treatment_table,population_table  

def usecase9(requirement_dict,treatment_table,population_table):
    try:
        elements_to_check = requirement_dict["input_dict"]['performance_matrix']
        columns_to_clip = elements_to_check
        for column in elements_to_check:
            lower_bound = population_table[column].quantile(.99)  # Specify the lower bound for clipping
            upper_bound = population_table[column].quantile(.01) # Specify the upper bound for clipping
            population_table[column] = population_table[column].\
                                                    clip(lower=lower_bound, upper=upper_bound)
        logging.info("Usecase 9 : Pass ,Population table outlier treatment done")

    except Exception as e:
        logging.info("Usecase 9 : Failed, Outlier treatment not done"+ str(e))
        error.append("Usecase 9 : Failed, Outlier treatment not done" + str(e))   
            
    return treatment_table,population_table   

def usecase10(requirement_dict,treatment_table,population_table,popu_table_format):
    try:
        tt=treatment_table[['emp_id',"treatment_allocated_date","treatment_allocated_end_date","treatment_id"]].drop_duplicates()
        tt1=tt.merge(population_table, on=['emp_id',"treatment_id"],how='left')
        ###################################
        if popu_table_format=="daily": tt1['date_diff'] = (tt1['treatment_allocated_date'] - tt1['date']).dt.days    
        else: tt1['date_diff'] = (tt1['treatment_allocated_date'] - tt1['start_date']).dt.days
        tt1['date_diff'] = tt1['date_diff'].apply(custom_mapping)    
                                        
        tt1['Status'] = tt1['date_diff'].apply(lambda x: 'pre' if x > 0 else ('post' if x <= 0 else 'NA'))  
        outcome_table=tt1[tt1['Status']=="post"].drop(columns=["Status","treatment_allocated_date"],axis=1).reset_index(drop=True)
        outcome_table['date_diff']=(outcome_table['end_date'] - outcome_table['treatment_allocated_end_date']).dt.days
        outcome_table=outcome_table[outcome_table['date_diff']>=0].drop(columns=["treatment_allocated_end_date"],axis=1).reset_index(drop=True)
        
        outcome_table['date_diff'] = outcome_table['date_diff'].apply(custom_mapping)
        
        population_table=tt1[tt1['Status']=="pre"].drop(columns=["Status","treatment_allocated_date",
                                                                "treatment_allocated_end_date"],axis=1).reset_index(drop=True)
        
        if popu_table_format=="daily": 
            # Aggregate Popultation daily data 
            outcome_table=tt1[tt1['Status']=="post"].drop(columns=['date_diff',"Status"
                                                                ],axis=1).reset_index(drop=True)
            outcome_table['diff']=(outcome_table['treatment_allocated_end_date'] - outcome_table['date']).dt.days
            outcome_table=outcome_table[outcome_table['diff']<=0].drop(columns=['diff',"treatment_allocated_date",
                                                                "treatment_allocated_end_date","treatment_id"
                                                                ],axis=1).reset_index(drop=True)        
            pd1=pd.DataFrame()
            for days in requirement_dict ["Performace_days"]['Performace_Pre_day']:
                t1=population_table[population_table['date_diff']<=days].drop(columns=['date_diff'],axis=1).reset_index(drop=True)
                t2=agg_kpi(t1)
                t2['date_diff']=days
                pd1=pd.conact([pd1,t2],axis=0)
                
            ot=pd.DataFrame()    
            for days in requirement_dict ["Performace_days"]['Performace_Post_day']:
                t1=outcome_table[outcome_table['date_diff']>=days].drop(columns=\
                                            ['date_diff'],axis=1).reset_index(drop=True)
                t2=agg_kpi(t1)
                t2['date_diff']=days
                ot=pd.conact([ot,t2],axis=0)        
            
            population_table=pd1 
            outcome_table=ot
            logging.info("Usecase 10: Pass, Aggregation done")

        #####
        # Check sufficient data 
        t_info=treatment_table[['treatment_allocated_date','treatment_allocated_end_date',"treatment_id"]].drop_duplicates()
        t_name=list(t_info['treatment_id'])
        new_data=pd.DataFrame()
        for i in t_name:
            p=population_table[population_table['treatment_id']==i]
            info=t_info[t_info['treatment_id']==str(i)][['treatment_allocated_date',"treatment_allocated_end_date"]]
            timestamp = pd.Timestamp(info['treatment_allocated_date'].iloc[0])
            p = p.assign(NewColumn=timestamp)
            p['pre_date_check']=(p['NewColumn']-p['start_date']).dt.days 
            for j in requirement_dict['Performace_days']['Performace_Pre_day']:    
                pre_short_data=p[p['pre_date_check']<=j].shape[0] 
                if pre_short_data<10: 
                    error.append(f"treatment_id :{i}  does not have {j} Pre Days sufficient data for APM in population table")
            o=outcome_table[outcome_table['treatment_id']==i]
            timestamp = pd.Timestamp(info['treatment_allocated_end_date'].iloc[0])
            o = o.assign(NewColumn=timestamp)
            o['post_date_check']=(o['end_date']-o['NewColumn']).dt.days
            for j in requirement_dict['Performace_days']['Performace_Post_day']: 
                post_short_data=o[o['post_date_check']==j].shape[0]     
                if post_short_data<1: 
                    error.append(f"treatment_id :{i}  does not have {j} Post Days sufficient data for APM in population table")

        logging.info("Usecase 10: Pass, Population and Outcome Table Data Checked ")

    except Exception as e:
        logging.info("Usecase 10 : Failed, Sufficient data check "+ str(e))
        error.append("Usecase 10 : Failed, Sufficient data check" + str(e)) 
        
    return outcome_table,population_table  


def get_response (prompt, max_tokens=200, temperature=0, model="text-davinci-003", 
                 top_p=1, frequency_penalty=0, presence_penalty=0):
    body = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty
    }
    headers = {
        "X-Org-Id": "00Dx0000000ALskEAG",
        "X-LLM-Provider": "OpenAI",
        "Content-Type": "application/json",
        "Authorization": "API_KEY c6024ef6-99d3-462c-b8ee-7cb930e26a5e"
    }    
    LLM_gateway_url = "https://bot-svc-llm.sfproxy.einstein.aws-dev4-uswest2.aws.sfdc.cl/v1.0/generations"
    resp = requests.post(
        LLM_gateway_url,
        json=body,
        headers=headers,
        verify=False
    )    
    result_dict = json.loads(resp.text)
    message = result_dict["generations"][0]["text"]
    return message


def data_quality_insights(population_table,segment_table,treatment_table):
    population_table = population_table[population_table.columns.drop(list(population_table.filter(regex='Unnamed:|#')))]
    treatment_table = treatment_table[treatment_table.columns.drop(list(treatment_table.filter(regex='Unnamed:|#')))]

    segment_missing_percentage = segment_table.isnull().sum()/segment_table.shape[0]*100
    missing_values_segment = get_response("""One sentence summary of quantitative insights related to percentage missing 
    values in the following table columns  - null percentages: {}. All figures should be in percentage. 
    If no missing values do not display output"""
    .format(segment_missing_percentage.sort_values(ascending=False)[segment_missing_percentage>10]))
    
    population_missing_percentage = population_table.isnull().sum()/population_table.shape[0]*100
    missing_values_population = get_response("""One sentence summary of quantitative insights related to percentage 
    missing values in the following table columns  - null percentages: {}. All figures should be in percentage. 
    If no missing values do not display output"""
    .format(population_missing_percentage.sort_values(ascending=False)[population_missing_percentage>10]))
    
    treatment_missing_percentage = treatment_table.isnull().sum()/treatment_table.shape[0]*100
    missing_values_treatment = get_response("""One sentence summary of quantitative insights related to percentage 
    missing values in the following table columns  - null percentages: {}. All figures should be in percentage. 
    Don't throw any output if no missing value found."""
    .format(treatment_missing_percentage.sort_values(ascending=False)[treatment_missing_percentage>10]))
    
    duplicate_records_population = get_response("""Give me important figures in sentence related to percentage 
    duplicate records from this table - {} All figures should be in percentage. If no duplicates display the 
    output - 'No duplicates found'""".format((population_table.shape[0]-population_table.drop_duplicates().
                                              shape[0])/population_table.shape[0]*100))
    duplicate_records_segment = get_response("""Give me important figures in sentence related to percentage 
    duplicate records from this table - {} All figures should be in percentage. If no duplicates display the 
    output - 'No duplicates found'""".format((segment_table.shape[0]-segment_table.drop_duplicates().
                                              shape[0])/segment_table.shape[0]*100))
    duplicate_records_treatment = get_response("""Give me important figures in sentence related to percentage 
    duplicate records from this table - {} All figures should be in percentage. If no duplicates display the 
    output - 'No duplicates found'""".format((treatment_table.shape[0]-treatment_table.drop_duplicates().
                                              shape[0])/treatment_table.shape[0]*100))
    
    
    def find_outliers_IQR(table):
       q1=table.quantile(0.25)
       q3=table.quantile(0.75)
       IQR=q3-q1
       No_of_outliers = table[((table<(q1-1.5*IQR)) | (table>(q3+1.5*IQR)))]
       return No_of_outliers

    outliers=[]
    col_list =[]
    for col in population_table.columns:
                        if (population_table[col].dtype == 'float'):
                            outlier = find_outliers_IQR(population_table[col])
                            col_list.append(col)
                            outliers.append(round(len(outlier)/population_table.shape[0]*100,2))
    outlier_df = pd.DataFrame(outliers,col_list,columns=['percentage outliers'])
    outlier_df = outlier_df[outlier_df["percentage outliers"]>=5]
    outliers = get_response("Generate summary of percentage outliers in each numeric column from the following table - {}"
                       .format(outlier_df))

    return (missing_values_treatment, duplicate_records_treatment, missing_values_segment, duplicate_records_segment, 
            missing_values_population, duplicate_records_population, outliers
           )


def ml_model_insights(eligibility_table_module1,treatment_table_module1,population_table_module1,outcome_table_module1,segment_table_module1,
                      classification_model_ouput,F1_score,regression_output_data_30days,regression_output_data_60days,
                      regression_output_data_90days,R2_score_30days,R2_score_60days,R2_score_90days):
    
    Feature_Importance = get_response("""Top 3 important features to generate 'predictions' column a predictor of 
    column 'Control/Test_group' from this table - {}. Do not consider columns such as emp_id & probability"""
                                      .format(classification_model_ouput))
    
    crosstab_df =pd.crosstab(classification_model_ouput['Control/Test_group'],classification_model_ouput['predictions'])
    crosstab_df = round(crosstab_df/crosstab_df.values.sum()*100,2)
    CM_insights = get_response("""Generate quantitative summary of actual Control&Test_group with 
                predictions from the following table - {}""".format(crosstab_df))
    
    Accuracy = round((crosstab_df.iloc[0,0]+crosstab_df.iloc[1,1])/crosstab_df.values.sum(),2)
    CM_accuracy_and_reliability = get_response(""" What are your top thought on the output of the machine learning 
    model? How accurate and reliable do you find it to be based on the output table with classification prediction 
    accuracy - {} and F1 score - {}? """.format(Accuracy,F1_score))
    
    R2_Scores = pd.concat([R2_score_30days,R2_score_60days,R2_score_90days],axis=0)
    R2_Scores.index=['R2_score_30days','R2_score_60days','R2_score_90days']
    # ML_insights = get_response(""" Top insights on output of the machine learning model? How accurate and reliable 
    # do you find it to be based on the output table with R2 score - {}? If the R2 score is less than 0.50 consider 
    # it as not a good prediction. R2 score greater than 70% is a good model prediction  """.format(R2_Scores))

    ML_insights = get_response("""Based on the output table and the R2 score of {} for our machine learning model, can you provide insights on its performance? How accurate and reliable do you find the model's predictions based on this R2 score?
Considering that an R2 score less than 0.50 indicates a poor prediction, how would you rate this model's performance in terms of accuracy?
Also, could you suggest any specific features or areas of the model that could be improved to achieve a higher R2 score?
Finally, how confident are you in using this model's predictions for real-world decision-making, given its R2 score and the dataset's characteristics?
Please provide detailed insights and suggestions for optimizing the model's predictive capabilities. Thank you!
""".format(R2_Scores))
    
    return (Feature_Importance,CM_insights,CM_accuracy_and_reliability,ML_insights)

def ml_model_insights_90(eligibility_table_module1,treatment_table_module1,population_table_module1,outcome_table_module1,segment_table_module1,
                      classification_model_ouput,F1_score,regression_output_data_90days,R2_score_90days):
    
    Feature_Importance = get_response("""Top 3 important features to generate 'predictions' column a predictor of 
    column 'Control/Test_group' from this table - {}. Do not consider columns such as emp_id & probability"""
                                      .format(classification_model_ouput))
    
    crosstab_df =pd.crosstab(classification_model_ouput['Control/Test_group'],classification_model_ouput['predictions'])
    crosstab_df = round(crosstab_df/crosstab_df.values.sum()*100,2)
    CM_insights = get_response("""Generate quantitative summary of actual Control&Test_group with 
                predictions from the following table - {}""".format(crosstab_df))
    
    Accuracy = round((crosstab_df.iloc[0,0]+crosstab_df.iloc[1,1])/crosstab_df.values.sum(),2)
    CM_accuracy_and_reliability = get_response(""" What are your top thought on the output of the machine learning 
    model? How accurate and reliable do you find it to be based on the output table with classification prediction 
    accuracy - {} and F1 score - {}? """.format(Accuracy,F1_score))
    
    R2_Scores = pd.concat([R2_score_90days],axis=0)
    R2_Scores.index=['R2_score_90days']
    # ML_insights = get_response(""" Top insights on output of the machine learning model? How accurate and reliable 
    # do you find it to be based on the output table with R2 score - {}? If the R2 score is less than 0.50 consider 
    # it as not a good prediction. R2 score greater than 70% is a good model prediction  """.format(R2_Scores))

    ML_insights = get_response("""Based on the output table and the R2 score of {} for our machine learning model, can you provide insights on its performance? How accurate and reliable do you find the model's predictions based on this R2 score?
Considering that an R2 score less than 0.50 indicates a poor prediction, how would you rate this model's performance in terms of accuracy?
Also, could you suggest any specific features or areas of the model that could be improved to achieve a higher R2 score?
Finally, how confident are you in using this model's predictions for real-world decision-making, given its R2 score and the dataset's characteristics?
Please provide detailed insights and suggestions for optimizing the model's predictive capabilities. Thank you!
""".format(R2_Scores))
    
    return (Feature_Importance,CM_insights,CM_accuracy_and_reliability,ML_insights)


def insights_module_3():
# Read Module 3 tables

    m3_segmentwise_output_=pd.read_parquet("artifacts/m3_segmentwise_output_.parquet")
    #treatment_id_mapping=pd.read_csv("artifacts/treatment_id_mapping.csv")
    #segment_labels=pd.read_csv("artifacts/segment_labels.csv")
    #segment_labels=pd.read_csv("artifacts/segment_labels.csv")
    df_overall = pd.read_parquet("artifacts/m3_overall_output_.parquet")
    #treatmnent_id_mapping_with_names=pd.read_csv("artifacts/treatmnent_id_mapping_with_names.csv")
    #####
    #Segment
    # M3_SEGMENTWISE_OUTPUT
    #m3_segmentwise_output_=m3_segmentwise_output_.merge(treatment_id_mapping,left_on='treatment_id',right_on='mapped_value',how='left')
    #m3_segmentwise_output_.drop(columns=['treatment_id_x','mapped_value'],axis=1,inplace=True)
    #m3_segmentwise_output_.rename(columns={'treatment_id_y':'treatment_id'},inplace=True)
    #segment_labels['Segment']=segment_labels['Segment'].astype(str)
    #m3_segmentwise_output_=m3_segmentwise_output_.merge(segment_labels,on=['Segment','Segment_Name'],how='left')
    #m3_segmentwise_output_['Segments']=m3_segmentwise_output_['Segments'].fillna("Overall")

    #m3_segmentwise_output_['Average_treatment_effect']=round(m3_segmentwise_output_['Average_treatment_effect'],0)

    #del m3_segmentwise_output_['Segment']
    #treatmnent_id_mapping_with_names=pd.read_csv("artifacts/treatmnent_id_mapping_with_names.csv")

    #m3_segmentwise_output_=m3_segmentwise_output_.merge(treatmnent_id_mapping_with_names,on='treatment_id',how='left')
    ranked_df = m3_segmentwise_output_.copy()
    ranked_df.sort_values(by=['treatment_label', 'Segment', 'Average_treatment_effect'], ascending=False, inplace=True)
    ranked_df['Rank'] = ranked_df.groupby(['treatment_label', 'Segment_Name'])['Average_treatment_effect'].rank(ascending=False)
    ranked_df.reset_index(drop=True, inplace=True)
    ranked_df_=ranked_df[(ranked_df['Rank']==1) &(ranked_df['Significance']=='Significant')][['Segment', 'treatment_label','Average_treatment_effect','KPI','Segment_Name','Significance']].\
    sort_values(by='Average_treatment_effect',ascending=False).head(10)

    insights_1 = []
    for i in range(0,ranked_df_['KPI'].shape[0]):
        insights_1.append(get_response("Below are Significant the average treatment effects, KPIs, and segment names for various segments and treatment labels and these are best acv for particular segment and segment _name. Explain in 15 words .Write this in a better way - {}. Do not expand on the abbreviation of {} .".format(ranked_df_.iloc[i],ranked_df_.iloc[i])))

    try:
        m=m3_segmentwise_output_[(m3_segmentwise_output_['Significance']=='Significant')&(m3_segmentwise_output_['Average_treatment_effect']>1000)].sort_values(by=['Average_treatment_effect'],ascending =False)
        for label in m['treatment_label'].unique():
            t1=m[m['treatment_label']==label]
        for segment in t1['Segment_Name'].unique():
            t2=t1[t1['Segment_Name']==segment]
        for kpi in t2['KPI'].unique():
            t3=t2[t2['KPI']==t2['KPI'].iloc[0]]
            t4=t3[t3['Significance']=='Significant'][['KPI','Segment_Name',"Average_treatment_effect","Segment","treatment_label"]]
            if t4.shape[0]!=0:
                insights_1.append(get_response("For {} segment {} the Average_treatment_effect of the KPI {} is max in all segments{}. Show treatment Name ,Segment,Segment_Name and KPI inside inverted commas  Do not use index and do not expand on the abbreviation of{}. Write it in better way .".format(t4['treatment_label'].iloc[0],t4['Segment'].iloc[0],t4['KPI'].iloc[0],t4['Segment_Name'].iloc[0],t4.head(1))))
    except:
        print('Average_treatment_effect is less than 1000 in all cases')
############################################

    ##Overall  INSIGHTS
    #tt=treatmnent_id_mapping_with_names.merge(treatment_id_mapping,on='treatment_id',how='left')[['mapped_value','treatment_id','treatment_label']]
    #df_overall=df_overall.merge(tt,left_on="treatment_id",right_on="mapped_value",how='left')
    #df_overall.drop(columns=['treatment_id_x','treatment_id_y','mapped_value'],inplace=True)
    insights = pd.DataFrame()
    for i in df_overall['KPI'].unique():
        df_temp = df_overall[df_overall['KPI']==i]
        df_temp = df_temp[df_temp['Average_treatment_effect']==df_temp['Average_treatment_effect'].max()]
        insight_0_message = '{} has the best average treatment effect in terms of {} by ${}. The results were found to be {} for the kpi is {},Do not expand on the abbreviation of PG, ACV {}'.format(df_temp['treatment_label'].iloc[0],df_temp['KPI'].iloc[0],round(df_temp['Average_treatment_effect'].iloc[0],0),df_temp['Significance'].iloc[0],i,i)
        insight_0 = get_response("Write this in a better way - {}. Do not expand on the abbreviation of PG, ACV {}".format(insight_0_message,i))
        insights_temp = pd.DataFrame({'KPI':pd.Series(i),'Insight':pd.Series(insight_0)})
        #insights = insights.append(insights_temp)
        insights=pd.concat([insights,insights_temp],axis=0)

    for i in df_overall['KPI'].unique():
        df_temp = df_overall[df_overall['KPI']==i]
        df_temp = df_temp[df_temp['Average_treatment_effect']<=0]
        insight_1_message = '{} had no impact in terms of {}. These treatments were {} for the kpi is {},Do not expand on the abbreviation of PG, ACV {}' .format(len(df_temp),i,df_temp['treatment_label'].unique(),i,i)
        insight_1 = get_response("Write this in a better way - {}. Do not expand on the abbreviation of PG, ACV {}.".format(insight_1_message,i))
        insights_temp = pd.DataFrame({'KPI':pd.Series(i),'Insight':pd.Series(insight_1)})
        #insights = insights.append(insights_temp)
        insights=pd.concat([insights,insights_temp],axis=0)

    for i in df_overall['KPI'].unique():
        df_temp = df_overall[df_overall['KPI']==i]
        df_temp = df_temp[df_temp['Significance']=='Significant']
        insight_2_message = 'In terms of {}, {} had a significant result. These treatments were {}for the kpi is {},Do not expand on the abbreviation of PG, ACV {}'.format(i,len(df_temp), df_temp['treatment_label'].unique(),i,i)
        insight_2 = get_response("Write this in a better way - {}. Do not expand on  the abbreviation of PG, ACV {}.".format(insight_2_message,i))
        insights_temp = pd.DataFrame({'KPI':pd.Series(i),'Insight':pd.Series(insight_2)})
        #insights = insights.append(insights_temp)
        insights=pd.concat([insights,insights_temp],axis=0)

    # Combine both results
    insights_1.extend(list(insights['Insight']))
    insights_1 = [string.replace('\n\n', '') for string in insights_1]

    return insights_1

def Module_2_Reg_Model(eligibility_table,treatment_table,population_table,segment_table,outcome_table,performance_KPI):
    try:
        cols = ['emp_id', 'treatment_id']

        # Check if performance_matrix is empty
        if 'input_dict' in requirement_dict and 'performance_matrix' in requirement_dict['input_dict']:
            performance_matrix = requirement_dict['input_dict']['performance_matrix']
            if not performance_matrix:
                raise ValueError("performance_matrix is empty")
            cols.extend(performance_matrix)
        else:
            raise KeyError("performance_matrix not found in input_dict")

        # Check if population_table_variable_for_model is empty
        if 'input_dict' in requirement_dict and 'population_table_variable_for_model' in requirement_dict['input_dict']:
            population_table_variable_for_model = requirement_dict['input_dict']['population_table_variable_for_model']
            if not population_table_variable_for_model:
                raise ValueError("Missing population_table_variable_for_model")
            cols.extend(population_table_variable_for_model)
        else:
            raise KeyError("population_table_variable_for_model not found in input_dict")

    except (KeyError, ValueError) as e:
        unique_errors = set(error)
        if str(e) not in unique_errors:
            error.append(f"{str(e)}")
            unique_errors.add(str(e))
            logging.info(f"Error occurred: {str(e)}")

    eligibility_table.drop_duplicates(subset=["emp_id","treatment_id","Control/Test_group"],
                                      keep='first', inplace=True, ignore_index=True)
    treatment_table.drop_duplicates(subset=["emp_id","treatment_label_taken"],
                                    keep='first', inplace=True, ignore_index=True)
    segment_table.drop_duplicates(subset="emp_id", keep='first', inplace=True, ignore_index=True)
    outcome_table.drop_duplicates(subset=["emp_id","treatment_id","date_diff"],
                                  keep='first', inplace=True, ignore_index=True)

    try:
        population_data = pd.merge(population_table[cols], 
                                   eligibility_table[['emp_id', 'treatment_id', 'Control/Test_group']], 
                                   on=['emp_id', 'treatment_id'], how='inner')
        # Check for unique values in the 'Control/Test_group' column
        if len(population_data['Control/Test_group'].unique()) != 2:
            raise Exception("The 'Control/Test_group' column should have two unique values (0 & 1)")
    except Exception as e:
        logging.info(f"Error occurred: {str(e)}")
    population_data = pd.merge(population_data, segment_table, on = 'emp_id', 
                               how='left')
    
    cols = [item for item in cols if item not in requirement_dict['input_dict']['population_table_variable_for_model']]
    population_data = population_data.merge(outcome_table[cols], 
                                            suffixes=('_pre', ''), on=['emp_id',"treatment_id"], how='inner')
    population_dataa = population_data
    
    cols = performance_matrix.copy()
    cols.remove(performance_KPI)
    population_data = population_data.drop(cols,axis=1)
    
    # Variance Inflation Factor (VIF) 
    if (population_data.shape[0]*population_data.shape[1]) >= 1000000:
        drop_list=[]
        while True:
                col_list = []
                for col in population_data.columns:
                    if ((population_data[col].dtype != 'object') & (col != performance_KPI)):
                        col_list.append(col)

                X = population_data[col_list]
                vif_data = pd.DataFrame() 
                vif_data["feature"] = X.columns 
                vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                                          for i in range(len(X.columns))]

                if (max(vif_data["VIF"]) >= 5):

                    drop_index=vif_data["VIF"].idxmax()
                    drop_col=vif_data.loc[drop_index,"feature"]
                    population_data=population_data.drop([drop_col], axis = 1)

                    drop_list.append(drop_col)
                else:
                    break
    
    population_data_tg = population_data[population_data["Control/Test_group"]==1].reset_index(drop=True)
    
    Input_Features = population_data_tg.drop(["emp_id","Control/Test_group",performance_KPI], axis=1)
    Output_Feature = population_data_tg[performance_KPI]
        
    # Building Various Regression Models & Hyper-parameters Tuning using GridSearchCV
    try:
        # Check if Model_tuning_requirement parameter is provided in requirement dictionary
        if "Model_tuning_requirement" in requirement_dict:
            if (requirement_dict['Model_tuning_requirement'].lower() == 'yes'):

                # Load the dataset
                X, y = Input_Features,Output_Feature

                # Define the objective function
                def objective(params, estimator):
                    # Create the classifier with the given hyperparameters
                    regressor = estimator(**params)

                    # Fit the classifier and compute the cross-validation score
                    score = cross_val_score(regressor, X, y, cv=5,scoring='r2').mean()

                    # Return the negated score since Bayesian optimization maximizes the objective
                    return -score

                # Define the search space for each model
                search_spaces = {
                    RandomForestRegressor: {
                        'n_estimators': (10,50),
                        'max_depth': (4, 10),
                        'min_samples_split': (2, 20),
                        'min_samples_leaf': (1, 10)
                    },

                    XGBRFRegressor: {
                        'learning_rate': (0.01, 0.1, 'log-uniform'),
                        'max_depth': (4, 10),
                        'min_child_weight': (1, 10),
                        'subsample': (0.5, 1.0, 'uniform'),
                        'n_estimators': (10, 100)
                    },
                    CatBoostRegressor : {
                        'learning_rate' : (0.01, 0.05),
                        'depth':(5,9),
                        'n_estimators':(10, 100),
                    }
                }

                # Perform Bayesian optimization for each model
                best_model = None
                best_params = None
                best_score = float('-inf')

                for estimator, space in search_spaces.items():
                    if estimator == CatBoostRegressor:
                        opt = BayesSearchCV(estimator=estimator(verbose=False),search_spaces=space, n_iter=10,cv=3,scoring='r2',n_jobs=-1)
                    else:
                        opt = BayesSearchCV(estimator=estimator(),search_spaces=space, n_iter=10,cv=3,scoring='r2',n_jobs=-1)    
                    
                    try:
                        opt.fit(X, y)

                        # Check if the current model is the best so far
                        if opt.best_score_ > best_score:
                            best_model = estimator
                            best_params = opt.best_params_
                            best_score = opt.best_score_  

                    except Exception as e:
                        logging.error(f"An error occurred while fitting the model for the KPI --> {performance_KPI}:\n {str(e)}")

                X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.33)
                model = best_model(**best_params)
        
                try:
                    model.fit(X_train, Y_train)
                    Y_pred =model.predict(X_test)
                    R2_score = round(r2_score(Y_test,Y_pred),2)

                    # Make predictions using the best model
                    prediction = model.predict(population_data[Input_Features.columns])
                    prediction_KPI=pd.DataFrame(prediction, columns= [performance_KPI+' predictions'])
                    feat_importances = pd.Series(model.feature_importances_, index=Input_Features.columns)

                except Exception as e:
                    prediction_KPI = pd.DataFrame(columns= [performance_KPI+' predictions'])
                    R2_score = 'Model was not trained due to data quality issue'
                    feat_importances = pd.Series(index=Input_Features.columns)
                    logging.error(f"An error occurred while fitting the model for the KPI --> {performance_KPI}:\n {str(e)}")

            else:
                # Split dataset into training & test sets
                X_train,X_test,Y_train,Y_test = train_test_split(Input_Features, Output_Feature,test_size=0.33)
                # Initialize and fit the default catboost model
                CTB_r = CatBoostRegressor(logging_level='Silent')
                try:
                    CTB_r.fit(X_train,Y_train)
                    Y_pred = CTB_r.predict(X_test)
                    R2_score = round(r2_score(Y_test,Y_pred),2)
                    
                    # Make predictions using the fitted model
                    prediction = CTB_r.predict(population_data[Input_Features.columns])
                    prediction_KPI=pd.DataFrame(prediction, columns= [performance_KPI+' predictions'])
                    feat_importances = pd.Series(CTB_r.feature_importances_, index=Input_Features.columns)
                except Exception as e:
                    prediction_KPI = pd.DataFrame(columns= [performance_KPI+' predictions'])
                    R2_score = 'Model was not trained due to data quality issue'
                    feat_importances = pd.Series(index=Input_Features.columns)
                    logging.error(f"An error occurred while fitting the model for the KPI --> {performance_KPI}:\n {str(e)}")
                
                
    
        else:
            # Split dataset into training & test sets
            X_train,X_test,Y_train,Y_test = train_test_split(Input_Features, Output_Feature,test_size=0.33)
            # Initialize and fit the default catboost model
            CTB_r = CatBoostRegressor(logging_level='Silent')
            try:
                CTB_r.fit(X_train,Y_train)
                Y_pred = CTB_r.predict(X_test)
                R2_score = round(r2_score(Y_test,Y_pred),2)

                # Make predictions using the fitted model
                prediction = CTB_r.predict(population_data[Input_Features.columns])
                prediction_KPI=pd.DataFrame(prediction, columns= [performance_KPI+' predictions'])
                feat_importances = pd.Series(CTB_r.feature_importances_, index=Input_Features.columns)
            except Exception as e:
                prediction_KPI = pd.DataFrame(columns= [performance_KPI+' predictions'])
                R2_score = 'Model was not trained due to data quality issue'
                feat_importances = pd.Series(index=Input_Features.columns)
                logging.error(f"An error occurred while fitting the model for the KPI --> {performance_KPI}:\n {str(e)}")
            raise KeyError(f"An error occurred while fitting the model for the KPI --> {performance_KPI}:\n {str(e)}")
    except KeyError as e:
        unique_errors = set(error)
        if str(e) not in unique_errors:
            error.append(f"{str(e)}")
            unique_errors.add(str(e))
            logging.info(f"{str(e)}")
    return(population_dataa, prediction_KPI, R2_score, feat_importances)

def Module_2_Classification_Model(eligibility_table,treatment_table,population_table,segment_table):
    try:
        cols = ['emp_id', 'treatment_id']

        # Check if performance_matrix is empty
        if 'input_dict' in requirement_dict and 'performance_matrix' in requirement_dict['input_dict']:
            performance_matrix = requirement_dict['input_dict']['performance_matrix']
            if not performance_matrix:
                raise ValueError("performance_matrix is empty")
            cols.extend(performance_matrix)
        else:
            raise KeyError("performance_matrix not found in input_dict")

        # Check if population_table_variable_for_model is empty
        if 'input_dict' in requirement_dict and 'population_table_variable_for_model' in requirement_dict['input_dict']:
            population_table_variable_for_model = requirement_dict['input_dict']['population_table_variable_for_model']
            if not population_table_variable_for_model:
                raise ValueError("Missing population_table_variable_for_model")
            cols.extend(population_table_variable_for_model)
        else:
            raise KeyError("population_table_variable_for_model not found in input_dict")

    except KeyError as e:
        logging.info(f"Error occurred: {str(e)}")
        error.append(f"Error occurred: {str(e)}")  
    except ValueError as e:
        logging.info(f"Error occurred: {str(e)}")
        error.append(f"{str(e)}")
    
    eligibility_table.drop_duplicates(subset=["emp_id","treatment_id","Control/Test_group"],
                                      keep='first', inplace=True, ignore_index=True)
    treatment_table.drop_duplicates(subset=["emp_id","treatment_label_taken"],
                                    keep='first', inplace=True, ignore_index=True)
    segment_table.drop_duplicates(subset="emp_id", keep='first', inplace=True, ignore_index=True)

    try:
        population_data = pd.merge(population_table[cols], 
                                   eligibility_table[['emp_id', 'treatment_id', 'Control/Test_group']], 
                                   on=['emp_id', 'treatment_id'], how='inner')
        # Check for unique values in the 'Control/Test_group' column
        if len(population_data['Control/Test_group'].unique()) != 2:
            raise Exception("The 'Control/Test_group' column should have two unique values (0 & 1)")
    except Exception as e:
        logging.info(f"Error occurred: {str(e)}")
    
    population_data = pd.merge(population_data, segment_table, on='emp_id', how='left')
    
    population_dataa = population_data
    population_data = population_dataa.drop(['emp_id'], axis=1)
    
    # Variance Inflation Factor (VIF) 
    if (population_data.shape[0]*population_data.shape[1]) >= 1000000:
        drop_list=[]
        while True:
                col_list = []
                for col in population_data.columns:
                    if ((population_data[col].dtype != 'object') & (col != 'Control/Test_group') ):
                        col_list.append(col)

                X = population_data[col_list]
                vif_data = pd.DataFrame() 
                vif_data["feature"] = X.columns 
                vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                                          for i in range(len(X.columns))]

                if (max(vif_data["VIF"]) >= 5):

                    drop_index=vif_data["VIF"].idxmax()
                    drop_col=vif_data.loc[drop_index,"feature"]
                    population_data=population_data.drop([drop_col], axis = 1)

                    drop_list.append(drop_col)
                else:
                    break
        
    Input_Features = population_data.drop(["Control/Test_group"], axis=1)
    Output_Feature = population_data["Control/Test_group"]

        
    # Building Various Classification Model & Hyper-parameters Tuning using BayesSearchCV
    try:
        # Check if Model_tuning_requirement parameter is provided in requirement dictionary
        if "Model_tuning_requirement" in requirement_dict:
            if requirement_dict['Model_tuning_requirement'].lower() == 'yes':
                #print("ssß",Output_Feature.value_counts()[0]/Output_Feature.value_counts()[1])


                # Load the dataset
                X, y = Input_Features,Output_Feature
                # Define the objective function
                def objective(params, estimator):
                    # Create the classifier with the given hyperparameters
                    classifier = estimator(**params)

                    # Fit the classifier and compute the cross-validation score
                    score = cross_val_score(classifier, X, y, cv=5,scoring='f1').mean()

                    # Return the negated score since Bayesian optimization maximizes the objective
                    return -score

                # Define the search space for each model
                search_spaces = {
                    RandomForestClassifier: {
                        'n_estimators': (10,50),
                        'max_depth': (4, 10),
                        'min_samples_split': (2, 20),
                        'min_samples_leaf': (1, 10)
                    },
                    XGBClassifier: {
                        'scale_pos_weight' :(1,3),         
                        'learning_rate': (0.01, 0.1, 'log-uniform'),
                        'max_depth': (4, 10),
                        'min_child_weight': (1, 10),
                        'subsample': (0.5, 1.0, 'uniform'),
                        'n_estimators': (10, 100)
                    },
                    CatBoostClassifier: {
                        'learning_rate' : (0.01, 0.1, 'log-uniform'),
                          'depth':(4,10),
                          'n_estimators':(10,100),
                    },
                }

                # Perform Bayesian optimization for each model
                best_model = None
                best_params = None
                best_score = float('-inf')

                for estimator, space in search_spaces.items():
                    if estimator == CatBoostClassifier:
                        opt = BayesSearchCV(estimator=estimator(verbose=False),search_spaces=space, n_iter=10,cv=3,scoring='f1',n_jobs=-1)
                    else:
                        opt = BayesSearchCV(estimator=estimator(),search_spaces=space, n_iter=10,cv=3,scoring='f1',n_jobs=-1)    

                    opt.fit(X, y)

                    # Check if the current model is the best so far
                    if opt.best_score_ > best_score:
                        best_model = estimator
                        best_params = opt.best_params_
                        best_score = opt.best_score_

                X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.33,stratify=y)
                model = best_model(**best_params)
                model.fit(X_train, Y_train)
                Y_pred =model.predict(X_test)
                F1_score = round(f1_score(Y_test,Y_pred),2)

                # Make predictions using the best model
                prediction = model.predict(Input_Features)
                population_data = pd.concat([population_dataa, pd.DataFrame(prediction, columns=['predictions'])], axis=1)
                proba_valid = model.predict_proba(Input_Features)[:, 1]
                population_data = pd.concat([population_data, pd.DataFrame(proba_valid, columns=['probability'])], axis=1)
                feat_importances = pd.Series(model.feature_importances_, index=Input_Features.columns)


            else:
                # Split dataset into training & test sets
                X_train,X_test,Y_train,Y_test = train_test_split(Input_Features, Output_Feature,test_size=0.33)
                # Initialize and fit the default catboost model
                CTB_c = CatBoostClassifier(logging_level='Silent')
                CTB_c.fit(X_train,Y_train)
                Y_pred = CTB_c.predict(X_test)
                F1_score = round(f1_score(Y_test,Y_pred),2)
                # Make predictions using the fitted model
                prediction = CTB_c.predict(Input_Features)
                population_data = pd.concat([population_dataa, pd.DataFrame(prediction, columns= ['predictions'])], axis=1)

                proba_valid = CTB_c.predict_proba(Input_Features)[:, 1]
                population_data = pd.concat([population_data, pd.DataFrame(proba_valid, columns= ['probability'])], axis=1)

                feat_importances = pd.Series(CTB_c.feature_importances_, index=Input_Features.columns)

        else:
            # Split dataset into training & test sets
            X_train,X_test,Y_train,Y_test = train_test_split(Input_Features, Output_Feature,test_size=0.33)
            # Initialize and fit the default catboost model
            CTB_c = CatBoostClassifier(logging_level='Silent')
            CTB_c.fit(X_train,Y_train)
            Y_pred = CTB_c.predict(X_test)
            F1_score = round(f1_score(Y_test,Y_pred),2)
            # Make predictions using the fitted model
            prediction = CTB_c.predict(Input_Features)
            population_data = pd.concat([population_dataa, pd.DataFrame(prediction, columns= ['predictions'])], axis=1)

            proba_valid = CTB_c.predict_proba(Input_Features)[:, 1]
            population_data = pd.concat([population_data, pd.DataFrame(proba_valid, columns= ['probability'])], axis=1)

            feat_importances = pd.Series(CTB_c.feature_importances_, index=Input_Features.columns)
            raise KeyError("Model_tuning_requirement parameter not found in requirement_dictionary")
    except Exception as e:
        #logging.info(f"Error occurred: {str(e)}")
        error.append(f"Error occurred: {str(e)}")
    return(population_data, F1_score, feat_importances)
