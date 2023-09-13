import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn  import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Union

from pycaret.datasets import get_data



st.header(" My Own Package Project ")
Select = st.sidebar.selectbox("Select Option",('Package','show code'))
if Select=='Package':
    data_set=st.file_uploader('Upload File',type=['csv','txt','xlsx'])
        # Select = st.sidebar.selectbox("Select Option", ('Exploratory Data Analysis ','Machine Learning Model','show Code'))
    if data_set is not None:
            df=pd.read_csv(data_set)

#  1- automate your preprocessing , detect columns types , null values 

            col=df.columns

            list_missing=df.isna().sum()
                    
            i=0 
            list_of_missing=[]


            for i in range(len(list_missing)):
                if list_missing.iloc[i]!=0:
                    list_of_missing.append(df.columns[i])
                i+=1 


            st.subheader('**Data Details**')  
            st.write('The shape of data : ',df.shape)
                
            st.subheader('**Columns of the dataset**')  
            st.write(df.columns.to_list())

            st.subheader('Data Types Of Columns')
            d_type=df.dtypes
            st.write(d_type) 


            st.subheader('Check missing values')
            list_missing=df.isna().sum()
            st.write(list_missing)
            i=0 
            list_of_missing=[]
            

            for i in range(len(list_missing)):
                if list_missing.iloc[i]!=0:
                    list_of_missing.append(df.columns[i])
                i+=1 
            if list_of_missing!=[] :    
                st.write('columns with missing values : ',list_of_missing)


 # 2- you can ask user to decide what columns to drop and what column to predict ONLY
            target_col = st.multiselect("Select Target variable",col)


 # you must detect the type and decide what is task type ( regression or classification )
            if len(np.unique(df[target_col].values) ) <=2:
                #  st.write('discrete values so that this is classification task')
                alg='classification'

            else:
                #  st.write('continuous values so that this is regression task')
                alg='Regression'
                      
            st.write(alg)     
            # x=df.drop(target_col,axis=1)
            # y=df[target_col]

            d_type=df.dtypes
            num_feature=[]
            cat_feature=[]
            for j in range(len(d_type)):
                if d_type.iloc[j]=='object':
                    cat_feature.append(df.columns[j])
                    
                elif d_type.iloc[j]=='float64'or d_type.iloc[j]=='int64' :
                    num_feature.append(df.columns[j]) 
     

# 4- after detecting null values and features types you can ask user what techniques he want to apply in the columns , 
# ask him like what do you want to do with categorical ( most frequent or just put additional class for missing value ) 
# and ask him again for continuous ( mean or median or mode ) and apply what he choose to your columns depends on every column type 
            # st.write(list_of_missing )

            # this code will runing when missing values in your data
            if  len(list_of_missing )!= 0:   
                mean_impute=SimpleImputer(strategy='mean',missing_values=np.nan)       
                mode_impute=SimpleImputer(strategy='most_frequent',missing_values=np.nan) 
                q_num=st.checkbox(' Do you want to fill missing values  numerical features ')
                if q_num:
                    i=0
                    j=0   
                    for i in range(len(list_of_missing)):
                        if list_of_missing[i]==num_feature[i]:
                            for j in  range(df[num_feature].shape[0]):
                                    df[num_feature[i]]=mean_impute.fit_transform(df[num_feature[i]].values.reshape(-1,1))
                                    j+=1    

                    st.write(df[num_feature].iloc[:,0:3])    
                
                q_cat=st.checkbox(' Do you want to fill missing values categorical  features ') 
                if q_cat:  
                        for i in range(len(list_of_missing)):
                            if   list_of_missing[i]==cat_feature[i]:
                                for j in  range(df[list_of_missing].shape[0]):
                                    df[list_of_missing[i]]=mode_impute.fit_transform(df[list_of_missing[i]].values.reshape(-1, 1))[:,0]
                                    j+=1

                                i+=1 
                        st.write(df[cat_feature]  )         

            
            x=df
            x=df.drop(target_col,axis=1)
            y=df[target_col]
            st.subheader('Data after preprocessing')
            st.write(x)
            # st.write(y)   
            if alg=='classification':
                from pycaret.classification import *
                btn1=st.button('classification by pycaret')
                if btn1:
                    # Initialize 
                    clf = setup(data = df,target=y,session_id = 123)
                    st.subheader('Initialize')
                    st.dataframe(pull())
                    # train
                    best = compare_models()  
                    # save_model(best,model_name='best model') 
                    st.subheader('Train')
                    st.dataframe(pull())
                    # create model
                    st.subheader('create best model')
                    b = create_model(best) 
                    st.dataframe(pull())

                    # plot
                    plot_model(b, plot='feature', display_format='streamlit')

                    # predict 
                    pred=predict_model(b)
                    st.subheader('predict model')
                    st.dataframe(pred)
                    
                    # save model
                    save_model(b, 'my_best model')

            elif alg=='Regression':
                from pycaret.regression import *
                btn1=st.button('Regression by pycaret')
                if btn1:
                    # Initialize 
                    clf = setup(data = df,target=y,session_id = 123)
                    st.subheader('Initialize')
                    st.dataframe(pull())
                    # train
                    best = compare_models()  
                    # save_model(best,model_name='best model') 
                    st.subheader('Train')
                    st.dataframe(pull())
                    # create model
                    st.subheader('create best model')
                    b = create_model(best) 
                    st.dataframe(pull())
                    

                    # plot
                    plot_model(b, plot='feature', display_format='streamlit')

                    # predict 
                    pred=predict_model(b)
                    st.subheader('predict model')
                    st.dataframe(pred)

                    # save model
                    save_model(b, 'my_best model')


if Select=='show code':
    st.subheader('Code Of APP') 
    code=""" 
import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn  import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Union

from pycaret.datasets import get_data



st.header(" My Own Package Project ")
Select = st.sidebar.selectbox("Select Option",('Package','show code'))
if Select=='Package':
    data_set=st.file_uploader('Upload File',type=['csv','txt','xlsx'])
        # Select = st.sidebar.selectbox("Select Option", ('Exploratory Data Analysis ','Machine Learning Model','show Code'))
    if data_set is not None:
            df=pd.read_csv(data_set)

#  1- automate your preprocessing , detect columns types , null values 

            col=df.columns

            list_missing=df.isna().sum()
                    
            i=0 
            list_of_missing=[]


            for i in range(len(list_missing)):
                if list_missing.iloc[i]!=0:
                    list_of_missing.append(df.columns[i])
                i+=1 


            st.subheader('**Data Details**')  
            st.write('The shape of data : ',df.shape)
                
            st.subheader('**Columns of the dataset**')  
            st.write(df.columns.to_list())

            st.subheader('Data Types Of Columns')
            d_type=df.dtypes
            st.write(d_type) 


            st.subheader('Check missing values')
            list_missing=df.isna().sum()
            st.write(list_missing)
            i=0 
            list_of_missing=[]
            

            for i in range(len(list_missing)):
                if list_missing.iloc[i]!=0:
                    list_of_missing.append(df.columns[i])
                i+=1 
            if list_of_missing!=[] :    
                st.write('columns with missing values : ',list_of_missing)


 # 2- you can ask user to decide what columns to drop and what column to predict ONLY
            target_col = st.multiselect("Select Target variable",col)


 # you must detect the type and decide what is task type ( regression or classification )
            if len(np.unique(df[target_col].values) ) <=2:
                #  st.write('discrete values so that this is classification task')
                alg='classification'

            else:
                #  st.write('continuous values so that this is regression task')
                alg='Regression'
                      
            st.write(alg)     
            # x=df.drop(target_col,axis=1)
            # y=df[target_col]

            d_type=df.dtypes
            num_feature=[]
            cat_feature=[]
            for j in range(len(d_type)):
                if d_type.iloc[j]=='object':
                    cat_feature.append(df.columns[j])
                    
                elif d_type.iloc[j]=='float64'or d_type.iloc[j]=='int64' :
                    num_feature.append(df.columns[j]) 
     

# 4- after detecting null values and features types you can ask user what techniques he want to apply in the columns , 
# ask him like what do you want to do with categorical ( most frequent or just put additional class for missing value ) 
# and ask him again for continuous ( mean or median or mode ) and apply what he choose to your columns depends on every column type 
            # st.write(list_of_missing )

            # this code will runing when missing values in your data
            if  len(list_of_missing )!= 0:   
                mean_impute=SimpleImputer(strategy='mean',missing_values=np.nan)       
                mode_impute=SimpleImputer(strategy='most_frequent',missing_values=np.nan) 
                q_num=st.checkbox(' Do you want to fill missing values  numerical features ')
                if q_num:
                    i=0
                    j=0   
                    for i in range(len(list_of_missing)):
                        if list_of_missing[i]==num_feature[i]:
                            for j in  range(df[num_feature].shape[0]):
                                    df[num_feature[i]]=mean_impute.fit_transform(df[num_feature[i]].values.reshape(-1,1))
                                    j+=1    

                    st.write(df[num_feature].iloc[:,0:3])    
                
                q_cat=st.checkbox(' Do you want to fill missing values categorical  features ') 
                if q_cat:  
                        for i in range(len(list_of_missing)):
                            if   list_of_missing[i]==cat_feature[i]:
                                for j in  range(df[list_of_missing].shape[0]):
                                    df[list_of_missing[i]]=mode_impute.fit_transform(df[list_of_missing[i]].values.reshape(-1, 1))[:,0]
                                    j+=1

                                i+=1 
                        st.write(df[cat_feature]  )         

            
            x=df
            x=df.drop(target_col,axis=1)
            y=df[target_col]
            st.subheader('Data after preprocessing')
            st.write(x)
            # st.write(y)   
            if alg=='classification':
                from pycaret.classification import *
                btn1=st.button('classification by pycaret')
                if btn1:
                    # Initialize 
                    clf = setup(data = df,target=y,session_id = 123)
                    st.subheader('Initialize')
                    st.dataframe(pull())
                    # train
                    best = compare_models()  
                    # save_model(best,model_name='best model') 
                    st.subheader('Train')
                    st.dataframe(pull())
                    # create model
                    st.subheader('create best model')
                    b = create_model(best) 
                    st.dataframe(pull())

                    # plot
                    plot_model(b, plot='feature', display_format='streamlit')

                    # predict 
                    pred=predict_model(b)
                    st.subheader('predict model')
                    st.dataframe(pred)
                    
                    # save model
                    save_model(b, 'my_best model')

            elif alg=='Regression':
                from pycaret.regression import *
                btn1=st.button('Regression by pycaret')
                if btn1:
                    # Initialize 
                    clf = setup(data = df,target=y,session_id = 123)
                    st.subheader('Initialize')
                    st.dataframe(pull())
                    # train
                    best = compare_models()  
                    # save_model(best,model_name='best model') 
                    st.subheader('Train')
                    st.dataframe(pull())
                    # create model
                    st.subheader('create best model')
                    b = create_model(best) 
                    st.dataframe(pull())
                    

                    # plot
                    plot_model(b, plot='feature', display_format='streamlit')

                    # predict 
                    pred=predict_model(b)
                    st.subheader('predict model')
                    st.dataframe(pred)

                    # save model
                    save_model(b, 'my_best model')



"""


    st.code(code, language='python')
    st.write("Created By: Aya ")





               
                
