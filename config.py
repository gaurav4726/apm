requirement_dict = {
    "input_dict": {
        "project_name":["apm_v2"],
        "performance_matrix": ['Number of Call Records','Avg Listen Ratio','AE - AI %','% Calls with Questions','Ae - Next Step %', '% Calls with Objections', '% Calls with Competitor Mentions'],
        'agg_level_performance_Matrix':['sum','sum','sum','sum','sum','sum','sum'],
        "treatment_name": ['ECI_treated_Sep2023'],
        "treatment_start_date":['2023-09-01'],
        'treatment_end_date':['2023-09-01'],
        "population_table_variable_for_model":['ACV','PG','QUOTA']
    },
    "treatment_table_columns_mapping": {
        'EMP_ID': "emp_id",
        'START_DATE': "treatment_allocated_date",
        'END_DATE': "treatment_allocated_end_date",
        'OFFERING_ID': 'treatment_id',
        'OFFERING_LABEL': "treatment_label_taken",
        'DATE_START_TREATMENT_TAKEN': "start_date_of_treatment_taken",
        'DATE_END_TREATMENT_TAKEN': "end_date_of_treatment_taken",
    },
    
    "population_table_column_mapping":{
        "START_DATE":'start_date',
        "END_DATE":"end_date",
        "EMP_ID":'emp_id',
        'Date':'date',
        'OFFERING_ID':'treatment_id'
    },
    
    "segment_table_column_mapping":{'EMP_ID': "emp_id"
    },
    
    "segment_variable_columns":{"segment1":"REGION",
                    "segment2":"MAPPEDMARKETSEGMENT",
                    "segment3":"MAPPEDSELLINGROLE",
                    "segment4":"MACROSEGMENT"             
                    #"variable":"random_variable",                            
    },    
    "customer_segemnts_for_output":{"segment1":'REGION',
                    "segment2":"MAPPEDMARKETSEGMENT",
                    "segment3":"MAPPEDSELLINGROLE",
                    "segment4":"MACROSEGMENT"
    },
                   
    
    "Performace_days":{
        "Performace_Pre_day":[90], # Days Before start_date_of_treatment_taken
        "Performace_Post_day":[30] # Days After end_date_of_treatment_taken
    },
    "Model_tuning_requirement":"No" #If No, catboost model will be used by default. 
    #If yes, best model using grid search will be used for training machine learning model
}