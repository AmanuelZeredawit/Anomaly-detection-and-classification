import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
import math


def preprocess_data():
    """
    This function clean the dataframe
    return the splitted train and test sets, 
    return the list of the numeric and categrial columns
    """
    
    df = pd.read_csv('data/output_1.csv')
    df.drop(['coil','furnace Number','Temperature before finishing mill',
             'Temperature after finishing mill','Thickness profile','Constriction_width'],axis=1, inplace = True)
    
    return df


    #################################################################################


def balance_sample_down():
    df =preprocess_data()
    # Separate majority and minority classes
    df_majority = df[df.is_constriction == 0]
    df_minority = df[df.is_constriction == 1]
    
 
    #downsample majority class
    df_majority_downsampled = resample(df_majority, 
                                 replace=False,     # sample with replacement
                                 n_samples = 1725,    # to match majority class
                                 random_state =123) # reproducible results
 
    # Combine majority class with upsampled minority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
 
    # Display new class counts
    print("Down sampled: ", df_downsampled.is_constriction.value_counts())

    y = df_downsampled.is_constriction
    X = df_downsampled.drop('is_constriction', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/5,random_state=0) 

    return X_train, X_test, y_train, y_test, df_downsampled


################################################################################################


def build_model(model,df):

    num_attribs = df.select_dtypes(include=np.number).columns.tolist() 
    remove_attribs =['coil','is_constriction']
    num_attribs = [i for i in num_attribs if i not in remove_attribs]
    cat_attribs = ['analyse']
    
    num_tr_pipeline = Pipeline([
        ('std_scaler', StandardScaler()),])
    
    cat_tr_pipeline = Pipeline([
        ('one_hot_encoder', OneHotEncoder(sparse=False, handle_unknown='ignore')),])
    preprocessors = ColumnTransformer([
        ("num_tr_pipeline", num_tr_pipeline, num_attribs),
        ("cat_tr_pipeline", cat_tr_pipeline, cat_attribs),])
    
    pipe =Pipeline([
    ('prepocessors',preprocessors),
    ('classifier_model',model),])
    
    return pipe


    #############################################################################################

def evaluate_models(balance_method):

    result = [] 
    
    print("Model with down sampled majority class")
    X_train, X_test, y_train, y_test,df = balance_sample_down()

    model_name = "KNN"
    model = RandomForestClassifier(n_estimators=200)
    pipe = build_model(model, df)
    pipe.fit(X_train,y_train)
    y_pred = pipe.predict(X_test)
    score = pipe.score(X_test, y_test)
    cm = metrics.confusion_matrix(y_test, y_pred) # confusion matrix
    f1 = f1_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2, average='binary')
    f_beta = fbeta_score(y_test, y_pred, average='macro', beta=0.5)

    result.append({"Model": model_name, "Score":score, 'Confusion_matrix':cm, 
                       'F1_score':f1, 'F2_score':f2, 'Fbeta_score':f_beta})
    
        
    result_df = pd.DataFrame(result)
    
    return result_df
        
###############################################     
    
    
test1_matrix = evaluate_models('up_sampling')
print(test1_matrix.head())
    