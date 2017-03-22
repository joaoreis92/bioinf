
# coding: utf-8

# # Bioinformatics - Protein subcellular location

# In[75]:

import os
import xgboost
from sklearn.model_selection import cross_val_score,KFold

from  sklearn import preprocessing
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier,BaggingRegressor,VotingClassifier
from sklearn.linear_model import LogisticRegression,RandomizedLogisticRegression, Lasso
from sklearn.metrics import f1_score,confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split,GridSearchCV,ShuffleSplit
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from Bio import SeqIO
import re
import tensorflow as tf







# ##  Load data and feature extraction

# In[6]:

def preprocess_pipeline(*files):
    #p = re.compile("(\w+\|\w+)\|(\w+\s[0-9a-zA-Z_\s\(\)\-\/,\.\>\:\'\[\]\+]+)OS=([0-9a-zA-Z_\s\(\)\-\/,\.\>\:\']+)GN=([0-9a-zA-Z_\s\(\)\-\/,\.\>\:\']+)PE=([0-9])+\s[SV=]+([0-9])|(\w+\|\w+)\|(\w+\s[0-9a-zA-Z_\s\(\)\-\/,\.\>\:\'\[\]]+)OS=([0-9a-zA-Z_\s\(\)\-\/,\.\>\:\']+)PE=([0-9])+\s[SV=]+([0-9])")
    p=re.compile("\|\w+\s(.+)OS=([0-9a-zA-Z_\s\(\)\-\/,\.\>\:\']+)(?:\sGN|\sPE)")
    data_features = []
    data_labels = []
    sequence = ''
    list_meta=[]
    for file in files:
        label = os.path.splitext(file)[0]
        f = open(file, "r")
        dict_meta = defaultdict(float)
        first_line = f.readline()
        meta_info = p.search(first_line)
        try:
            dict_meta["organism"] = meta_info.group(2)
            dict_meta["protein"] = meta_info.group(1)
            dict_meta["class"] = label
        except:
            print(first_line)
        list_meta.append(dict_meta)
        for line in f:
            line = line.rstrip('\n')
            if line[0] != '>':
                sequence += line
            else:
                dict_meta["sequence"] = sequence
                dict_meta = defaultdict(float)
                meta_info = p.search(line)
                try:
                    dict_meta["organism"] = meta_info.group(2)
                    dict_meta["protein"] = meta_info.group(1)
                    dict_meta["class"] = label
                except:
                    print(line)
                list_meta.append(dict_meta)
                data_features.append(sequence)
                data_labels.append(label)
                sequence = ''
        #Last input
        list_meta[-1]["sequence"] = sequence
        data_features.append(sequence)
        data_labels.append(label)
        sequence = ''



    return data_features, data_labels,list_meta

dic_properties = {
    'small' : ['A','G','C','S','P','N','C','T','D'],
    'tiny' : ['A','G','C','S'],
    'polar' : ['K','H','R','D','E','Q','N','S','C','T','Y','W'],
    'charged' : ['K','H','R','D','E'],
    'positive' : ['K','H','R'],
    'negative' :  ['D','E'],
    'hidrophobic' : ['F','Y','W','H','I','L','V','A','G','C','M','K','T'],
    'aromatic' : ['F','Y','W','H'],
    'aliphatic' : ['I','L','V']
    
}

def feat_extract(sequences):
    list_dict_feat = []
    for sequence in sequences:
        
        protein = ProteinAnalysis(sequence)
        sequence_feat = defaultdict(float)
        sequence_len = len(sequence)

        sequence_feat["sequence_length"] = sequence_len        
        sequence_feat["aromaticty"] = protein.aromaticity()
        sequence_feat["isoeletric_point"] = protein.isoelectric_point()
        #sequence_feat["flexibility"] = protein.flexibility()
        if ('X' not in sequence) and ('O' not in sequence) and ('U' not in sequence) and ('B' not in sequence):
            sequence_feat["molecular_weight"] = protein.molecular_weight()
        for letter in sequence:
            sequence_feat["relative_fre_{}".format(letter)] += 1/sequence_len
            for property in dic_properties:
                if letter in dic_properties[property]:
                    sequence_feat['freq_{}'.format(property)] += 1
        for letter in sequence[0:50]:    
            sequence_feat["relative_fre_start{}".format(letter)] += 1/50
        for letter in sequence[-51:-1]:    
            sequence_feat["relative_fre_end{}".format(letter)] += 1/50
        list_dict_feat.append(sequence_feat)
    return list_dict_feat

label_encoder = preprocessing.LabelEncoder()
vectorizer = DictVectorizer(sparse=False)
data_sequence, data_labels, meta_info = preprocess_pipeline('cyto.fasta', 'mito.fasta','nucleus.fasta','secreted.fasta')
print("Loaded data")
 


# ## Feature Selection
# In[ ]:
def lasso_regression():
	lasso = Lasso()
	lasso.fit(data_sequence,data_labels)



# ## Linear Models

# In[64]:

def train_rf(x,y):
    model = RandomForestClassifier(class_weight='balanced',n_estimators=15)
    model.fit(x, y)
    return model
    


# In[65]:

def train_xgb(x,y):
    model = xgboost.XGBClassifier(learning_rate =0.1, n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
    model.fit(x, y)
    return model


# In[66]:

def train_svm(x,y):
    model = SVC(class_weight='balanced', probability=True)
    model.fit(x, y)
    return model


# In[67]:

def train_lr(x,y):
    model = LogisticRegression(class_weight='balanced',C=0.1)
    model.fit(x, y)
    return model


# In[70]:

def train(x,y):
    
    labels_enc = label_encoder.fit_transform(y)
    features_enc = vectorizer.fit_transform(feat_extract(x))
    print("Feature Extraction done.")
    clf_rf = train_rf(features_enc,y)
    print("Random Forest model done.")
    clf_xgb = train_xgb(features_enc,y)
    print("XGBoost model done.")
    clf_svm = train_svm(features_enc,y)
    print("SVM model done.")
    clf_lr = train_lr(features_enc,y)
    print("Logistic Regression model done.")
    model = VotingClassifier([('rf',clf_rf),('xgb',clf_xgb),('svm',clf_svm),('lr',clf_lr)],voting='soft')
    model.fit(features_enc,y)
    print("Voting classifier done.")
    return model

#def cross_validation(x,y):
	

def validate(x,model):
    
    
    features = vectorizer.transform(feat_extract(x))
    predicts = model.predict_proba(features)  
    predicts_label = np.argmax(predicts,1)
    labels_predicted = label_encoder.inverse_transform(predicts_label)
    #label_and_confidence = list(zip(labels_predicted,np.amax(predicts,1)))

    return labels_predicted#,np.amax(predicts,1)


# ## Grid Search

# In[ ]:

def grid_search(x,y,model):
    
    x = vectorizer.fit_transform(feat_extract(x))
    if model == 'svm':
        parameters = {'C':[0.001,1,0.1,10]}
        model = SVC(class_weight='balanced', probability=True)
        print("SVM model created, starting grid search...")
    elif model == 'lr':
        parameters = {'C':[10,50,100]}
        model = LogisticRegression(class_weight='balanced')
        print("LR model created, starting grid search...")
    rs = ShuffleSplit(n_splits=3, random_state=0)    
    grid_search = GridSearchCV(model,parameters,verbose=5,cv=rs)
    grid_search.fit(x,y)
    print("Grid Search finished")
    print(pd.DataFrame(grid_search.cv_results_ ).to_string())
    return grid_search.best_estimator_








def test_blind(file,model):
    f = open(file,'r')
    preds = open('blind_predictions.txt','w')
    sequence = ''
    
    first_line= f.readline()
    first_line = first_line.rstrip('\n')
    preds.write(first_line + ' ')
    
    for line in f.readlines():
        line = line.rstrip('\n')
        if line[0] != '>':
            sequence += line
        else:
            #import pdb;pdb.set_trace()
            feature = vectorizer.transform(feat_extract([sequence]))
            predict = model.predict_proba(feature)
            predict_label = np.argmax(predict,1)
            label_predicted = label_encoder.inverse_transform(predict_label)
            #preds.write(label_predicted[0] + ' \t\t' + str(np.amax(predict,1)[0]) + '\n' + line + ' ')
            preds.write("{0} {1:>8} \n{2} ".format(label_predicted[0],str(np.amax(predict,1)[0]),line))
            sequence = ''
    feature = vectorizer.transform(feat_extract([sequence]))
    predict = model.predict_proba(feature)
    predict_label = np.argmax(predict,1)
    label_predicted = label_encoder.inverse_transform(predict_label)
    #preds.write(label_predicted[0] + ' \t\t' + str(np.amax(predict,1)[0]) + '\n' + line + ' ')
    preds.write("{0} {1}".format(label_predicted[0],str(np.amax(predict,1)[0]),line))
    sequence = ''
    preds.close()
    f.close()
    

# In[ ]:



