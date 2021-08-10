# W261 Final Project - Flight Delay Prediction

## Introduction
Flight delays create problems in scheduling for airlines and airports, leading to passenger inconvenience, and huge economic losses. As a result there is growing interest in predicting flight delays beforehand in order to optimize operations and improve customer satisfaction. In this project, we we were able to successfully predict airline delay classification with a F1 score of about 0.90. We employed a number of different classification algorithms (Logistic Regression, Random Forests, SVM and Gradient Boosted Trees) - and we compared and contrasted their performance along different dimensions. The data was trained on around 30 million rows and we employed Databricks and PySpark to perform model training. You can view the full report and analysis [here](https://github.com/abhisha1991/w261_final_project/blob/main/Team_07/Report/W261_SU21_FINAL_PROJECT_TEAM07.ipynb)

Below is a snippet of our final reported scores against our leading hypothesis of selected features.
![image](https://user-images.githubusercontent.com/10823325/128800792-32fb6958-62f3-452c-8837-e798fd2e1d02.png)

## Problem

For now, the problem to be tackled in this project is framed as follows:
Predict departure delay/no delay, where a delay is defined as 15-minute delay (or greater) with respect to the planned time of departure. This prediction should be done two hours ahead of departure (thereby giving airlines and airports time to regroup and passengers a heads up on a delay). 


## Team
1. Emily Brantner 
2. Sarah Iranpour 
3. Michael Bollig 
4. Abhi Sharma

## Instructors and Advisors
1. Jimi Shanahan
2. Luis Villarreal

## Links and Resources
1. [Project Description](https://docs.google.com/document/d/1dIh9RDSp8TLZ1JPbuJqZn1PKIMVGv-1PeOMjSYuf7XM/edit)
2. [Official Project Repository](https://github.com/UCB-w261/main/tree/main/Assignments/FinalProject)
3. [Reference Notebook 1](https://github.com/MScatolin/W261-SP19-Team13-FinalProject/blob/master/Team13_FinalProject.ipynb)
4. [Reference Notebook 2](https://github.com/RLashofRegas/mids-w261-final/tree/main/notebooks)
5. [Understanding Reporting for Causes of Delay & Cancellations](https://www.bts.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations) 
6. [Airline Data Table Column Schema Explanation](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ) and [Glossary](https://www.transtats.bts.gov/Glossary.asp?index=C)
