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
  - There were a variety of columns that were identified as features for the model because they encode the categorical variables, and were split into training and testing datasets and were standardized.
  - These columns were:
    - APPLICATION_TYPE, 
    - AFFILIATION, 
    - CLASSIFICATION, 
    - USE_CASE, 
    - ORGANIZATION, 
    - STATUS, 
    - INCOME_AMT, 
    - SPECIAL_CONSIDERATIONS, 
    - ASK_AMT

- **What variable(s) are neither targets nor features, and should be removed from the input data?**
  - The columns EIN and NAME were identification information and were removed from the input data.

![Data_Preprocessing](https://github.com/pfrivas/Neural_Network_Charity_Analysis/blob/main/Images/Data%20Preprocessing.png)


### Compiling, Training, and Evaluating the Model

- **How many neurons, layers, and activation functions did you select for your neural network model, and why?**

- **Were you able to achieve the target model performance?**
  - The model accuracy is under 75%. This is not a satisfying performance to help predict the outcome of the charity donations.
  - ![model_accuracy](https://github.com/pfrivas/Neural_Network_Charity_Analysis/blob/main/Images/Final%20Accuracy%20after%20Optimization.png)
  
- **What steps did you take to try and increase model performance?**
  - Additional neurons are added to hidden layers
    - Code:
    ```
        # Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
        number_input_features = len(X_train_scaled[0])
        nodes_hidden_layer1 = 100
        nodes_hidden_layer2 = 30

        nn = tf.keras.models.Sequential()
    ```
     ![Neurons_Added](https://github.com/pfrivas/Neural_Network_Charity_Analysis/blob/main/Images/Additional%20Neurons%20added%20to%20Hidden%20Layers.png)
     
  - Additional hidden layers are added
    - Code:
    ```
         # Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
         number_input_features = len(X_train_scaled[0])
         nodes_hidden_layer1 = 100
         nodes_hidden_layer2 = 80
         nodes_hidden_layer3 = 50
         nodes_hidden_layer4 = 30
         nodes_hidden_layer5 = 10

         nn = tf.keras.models.Sequential()

         # First hidden layer
         nn.add(tf.keras.layers.Dense(units=nodes_hidden_layer1, activation="relu", input_dim=number_input_features))

         # Second hidden layer
         nn.add(tf.keras.layers.Dense(units=nodes_hidden_layer2, activation="relu"))

         # Third hidden layer
         nn.add(tf.keras.layers.Dense(units=nodes_hidden_layer3, activation="relu"))

         # Fourth hidden layer
         nn.add(tf.keras.layers.Dense(units=nodes_hidden_layer4, activation="relu"))

         # Fifth hidden layer
         nn.add(tf.keras.layers.Dense(units=nodes_hidden_layer5, activation="relu"))

         # Output layer
         nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

         # Check the structure of the model
         nn.summary()
     ```
      ![Hidden_Layers_Added](https://github.com/pfrivas/Neural_Network_Charity_Analysis/blob/main/Images/Additional%20Hidden%20Layers%20are%20Added.png)
     
  - The activation function of hidden layers or output layers is changed for optimization 
    - Code:
     ```
         # First hidden layer
         nn.add(tf.keras.layers.Dense(units=nodes_hidden_layer1, activation="tanh", input_dim=number_input_features))

         # Second hidden layer
         nn.add(tf.keras.layers.Dense(units=nodes_hidden_layer2, activation="tanh"))

         # Third hidden layer
         nn.add(tf.keras.layers.Dense(units=nodes_hidden_layer3, activation="tanh"))

         # Fourth hidden layer
         nn.add(tf.keras.layers.Dense(units=nodes_hidden_layer4, activation="tanh"))

         # Fifth hidden layer
         nn.add(tf.keras.layers.Dense(units=nodes_hidden_layer5, activation="tanh"))

         # Output layer
         nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

          # Check the structure of the model
          nn.summary()
      ```
      ![Activation_Func_Changed](https://github.com/pfrivas/Neural_Network_Charity_Analysis/blob/main/Images/Activation%20Function%20Changed.png)
      
## Summary:

### Summary of the results

### Recommendation on using a different model to solve the classification problem
