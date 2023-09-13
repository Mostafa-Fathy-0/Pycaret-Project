# Summary of project 
## this task is a simple shape of pycaret library it  seeks to provide a low-code machine learning ,it cosists of main aspects as follows ,it consists of two main classes and many essential functions.

# --------------------------------------------------------------------

## The first class is for Regression and contain some of important Regression algorithm methods , and these mathods are :

### 1- simple_regression() :
####  it performs simple regression model for one variable and graph it to show the fit line that shows target feature.

### 2- polynomial_features() :
#### it performs regression model for one variable and it using polynomial features to get the best fit line for target feature.

### 3- Regularization() :
#### it uses to algorithms to regularize features by using Lasso algorithm and also by using Ridge algorithm to improve model.

### 4- gradient_descent() :
#### it performs the gradient descent formula to minimize the cost function and optimize model parameters and graph target feature after applying this formula.

### 5- multilinear_regression() :
#### it performs simple regression model for multivariables features to get targets and graph the best fit line for it.

### 6- model_evaluation() :
#### it works to evaluate models by scaling training feature and apllay simple regression and polynomial features on dataset and perform a graph to show best fit line of target features.

# --------------------------------------------------------------------

## The second class is for Classification and contain some of important Classification algorithm methods , and these mathods are :

### 1- logistic_regression() :
#### it applies the logisitc regression and apply confusion matrix in dataset and perform graph  to visualize this algorithm.

### 2- support_vector_machine() :
#### it applies support vector machine algorithm on dataset and use diffirent kernals like linear , poly and rbf and perform visualization to this algorithm.

### 3- K_nearest_neighbor() :
#### it applies k nearst neighbor on dataset and use n_neighbors and polynomial feature to perform graph for this algorithm.

### 4- decision_tree() :
#### it applies descision tree algorithm on dataset and perform graph for this algorithm. 

### 5- random_forest() :
#### it applies random forest on dataset and work to show some important information and perform a graph for it.

### 6- boosting() :
#### use xgboost and xgbclassifier to apply it on dataset and work to show some important information and perform a graph for it.

### 7- cross_validation() :
#### apply cross validation algorithm on dataset and show some important statistics to user.

### 8- hyper_parameter_tuning() :
#### apply hayper parameter tuning show some important informations to user.

# --------------------------------------------------------------------

## The essential functions are :

### 1- get_files() :
#### it's main role to get dataset file from user to allow him to apply differant machine learning algorithms on it.

### 2- check_file_path() :
#### it's main role to chek type of file that user has entered to perform that it's valid type or not valid and excepected to harmfull. 

### 3- data_preprocessing() :
#### it's main role to make some of preprocessing steps on dataset including fill missing values if it is catogerical or numerical values and it also enclude step of Encoding to prepare dataset to be used in diffierent ML models.

### 4- drop_column() :
#### it's main role to allow user to drop any columns that not help him in it's work.

### 5- choose_model_and_algorithms() :
#### it's main role to allow user to choose which model he want to perform in dataset after processed and also it allow user to choose a specific algorithm perform.

### 6- choose_features() :
#### it's main role to allow user to assign values of columns to training data and target data to input them after that to different models to use them in apply diffirent algorithms

### 7- main() :
####  it's main role to run all of other method to explore and train diffirent machine learning models
