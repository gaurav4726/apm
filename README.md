# Summary

A common challenge across multiple teams: measuring the impact of various treatments (such as sales plays, certifications, planning processes) on outcomes (such as ACV, pipegen, opportunity creation, customer renewal) for their employees with specific attributes. While the APM 1.0 dashboard currently addresses this problem for Account Executives, the goal is to develop **APM as a Service (APMAAS)** as a more versatile data pipeline that can support APM and other use cases with reusable code and best practices. This project aims to provide a comprehensive solution that enables the measurement and analysis of treatment impacts, empowering teams to optimize their strategies and drive success.

# Modules Overview

The APM logic is divided into four modules, each performing a specific task:

**Module 1: Data Ingestion and Basic Checks** - Ingests data, performs basic checks, and provides data for Module 2.


**Module 2: Classification and Regression Modeling** - Estimates treatment likelihood and examines the impact of treatment on performance KPIs.


**Module 3: Matching and Treatment Effect Calculation** - Matches data and analyzes treatment effects using various strategies.


**Module 4: Output Analysis and Insight Generation** - Analyzes the treatment effects and generates insights using Einstein GPT.


# Module 1 - Data Ingestion and Basic Checks

This module is responsible for ingesting data, performing basic checks, and providing data for Module 2. It consists of several components that work together to achieve the desired functionality.

## Configuration
The `config` file is provided to enable customization based on customer input. You can modify the configuration settings in this file to adapt the module to specific requirements.

## Logging and Exception Handling
The module utilizes an exception and logger file to facilitate logging and exception handling. These files help in capturing and managing errors or exceptional scenarios during the execution of the code.

## Helper Functions
The `helper` file contains various functions that are designed to assist in the data transformation process. These functions can be utilized throughout the module to streamline the data processing steps.

## Data Ingestion
The `data_ingestion` folder holds all the necessary functions related to data ingestion. These functions enable the module to import data from different sources by simply specifying the streaming details within the function. Currently, it is assumed that the data is present in a designated folder.

## Module 1 Execution
The main `module1.py` file coordinates the execution of the data transformation functions used within the module. By running this file, all the necessary functions will be executed to transform the data as required for our module.

## Artifact Folder
When running the code within the data ingestion process, an `artifact` folder will be created. This folder will contain all the files necessary for Module 2. These artifacts are generated during the execution of Module 1 and can be utilized in subsequent modules.

# Module 2 - Classification and Regression Modeling

Module 2 takes input from Module 1 and consists of two main sections: Classification Modeling and Regression Modeling. It estimates the probability of events/treatments and examines the impact of treatment on performance KPIs.

## Classification Modeling

Inputs from Module 1 include cleaned data represented in five tables: population_table, eligibility_table, treatment_table, and segment_table. The module trains a machine learning model to estimate treatment likelihood, producing the following outputs:

* Classification model F1-score
* Output table with treatment likelihood predicted by the classification model

## Regression Modeling

The module examines the impact of treatment on performance KPIs using a regression model. It considers employee characteristics, performance parameters before and after treatment, and one or more performance parameters (KPIs). The output includes:

* Regression Model R2-score
* Output table with predicted performance KPIs after treatment

The module saves the output in the pyarrow-parquet file format.
For detailed instructions on running Module 2 and understanding its code, please refer to the specific code files.

# Module 3 - Matching and Treatment Effect Calculation

Module 3 performs matching on input data and calculates treatment effects. The goal is to create matched groups by balancing covariates between treatment groups. The module utilizes various matching strategies and cutpoint configurations to explore different ways of creating matched groups.

The code in Module 3 automates the matching process, allowing the selection of the best matching strategy and cutpoint configuration. It generates results tables summarizing the outcomes of each matching strategy, including the number of matched observations, maximum standardized mean difference, and match percentage. The code also calculates average treatment effects (ATE) and their confidence intervals for the overall data and each segment.

## System Setup Requirements

To set up the system for running Module 3, follow these steps:

1. Set up an R Docker image using the provided [instructions](https://git.soma.salesforce.com/Business-Data-Science/gsi-predictive-enablement/tree/master/src/sql/production/apm-as-a-service/docker).
2. Connect the container to VS Code to access the module_3.r file located at /home/rstudio/mac/module_3.r.
3. Install the required dependencies and import the necessary libraries:

  * arrow
  * MatchIt
  * grf
  * magrittr
  * mltools
  * MLmetrics
  * dplyr
  * glue
  * knitr

4. Ensure that the required data files are available in the desired format (arrow) and directory specified in the code.

## Data Input

Module 3 requires data from Module 2 as input. To ensure compatibility, the output from Module 2 should have the following columns:

1. emp_id
2. {KPI}_pre
3. {KPI}_outcome
4. Control/Test_group
5. treatment_id

The data should also include specified segment columns (e.g., region, vertical, role) or a combined segment column like peer_characteristics (combination of different segments).

# Module 4

