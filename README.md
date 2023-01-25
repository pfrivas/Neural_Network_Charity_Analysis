# Neural_Network_Charity_Analysis

## Overview of the loan prediction risk analysis:

### Purpose of Analysis

The purpose of this analysis is to help a  nonprofit foundation, Alphabet Soup, analyze the impact of their donations and vetting potential recipients by predicting which organizations are worth donating to and which are too high risk. This was done through a design of a deep learning neural network that evaluated all types of input data and produces a clear decision making result using the Python TensorFlow library. The steps in designing the deep learning neural network involved 
 - (1) Preprocessing Data for a Neural Network Model using Pandas and the Scikit-Learn’s StandardScaler() function. 
 - (2) Compiling, Training, and Evaluating the Model using Tensorflow Keras
 - (3) Optimizing the Model



### Resources
- Data Source: [charity_data.csv](https://github.com/pfrivas/Neural_Network_Charity_Analysis/blob/main/Challenge/charity_data.csv)
- Software: Pandas, NumPy, Scikit-Learn, Tensorflow, Anaconda Navigator, Jupyter Notebook
- Jupyter Notebooks: [AlphabetSoupCharity.ipynb](https://github.com/pfrivas/Neural_Network_Charity_Analysis/blob/main/Challenge/AlphabetSoupCharity.ipynb) and [AlphabetSoupCharity.ipynb](https://github.com/pfrivas/Neural_Network_Charity_Analysis/blob/main/Challenge/AlphabetSoupCharity_Optimization.ipynb)

## Results:

### Data Preprocessing

- **What variable(s) are considered the target(s) for your model?**
 - The column IS_SUCCESSFUL was considered the target for the model because it contained binary data about if the donations were effectively used. 
 
- **What variable(s) are considered to be the features for your model?**
 - There were columns that are the features for the model because they encode the categorical variables, and were split into training and testing datasets and were standardized.
 - These columns were APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, and ASK_AMT

- **What variable(s) are neither targets nor features, and should be removed from the input data?**
 - The columns EIN and NAME were identification information and were removed from the input data.

### Compiling, Training, and Evaluating the Model

- **How many neurons, layers, and activation functions did you select for your neural network model, and why?**

- **Were you able to achieve the target model performance?**

- **What steps did you take to try and increase model performance?**

## Summary:

### Summary of the results

### Recommendation on using a different model to solve the classification problem
