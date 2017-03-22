
# coding: utf-8

# # Bioinformatics - Protein subcellular location

# In[2]:

import os
#from IPython.display import display, HTML
#import matplotlib.pyplot as plt
import xgboost

from  sklearn import preprocessing
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression,RandomizedLogisticRegression
from sklearn.metrics import f1_score,confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
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







# ##  Load data and feature extraction

# In[3]:

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

label_encoder = preprocessing.LabelBinarizer()
vectorizer = DictVectorizer(sparse=False)


# ## Linear Models

# In[4]:

def train(x,y):
    
    labels_enc = label_encoder.fit_transform(y)
    features_enc = vectorizer.fit_transform(feat_extract(x))
    
    
    #model = RandomForestClassifier(class_weight='balanced',n_estimators=15)
    
    model = xgboost.XGBClassifier(
                 learning_rate =0.1,
                 n_estimators=1000,
                 max_depth=5,
                 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
    
    #model = SVC(class_weight='balanced', probability=True)
    #model = LogisticRegression(class_weight='balanced')
    #model = RandomizedLogisticRegression()
    model.fit(features_enc, labels_enc)
    
    return model

def validate(x,model):
    
    
    features = vectorizer.transform(feat_extract(x))
    predicts = model.predict_proba(features)
    predicts_label = np.argmax(predicts,1)
    labels_predicted = label_encoder.inverse_transform(predicts_label)
    label_and_confidence = list(zip(labels_predicted,np.amax(predicts,1)))

    return labels_predicted#,np.amax(predicts,1)


# ## Neural Network

# In[23]:

def create_model():
    model = Sequential()
    model.add(Dense(64,input_dim=80,activation="relu"))
    model.add(Dense(4,activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1, momentum=0.9, nesterov=True))
    return model
    
def train_nn(x,y):
    labels_enc = label_encoder.fit_transform(y)
    features_enc = vectorizer.fit_transform(feat_extract(x))
    print(labels_enc.shape)
    print(features_enc.shape)
    model = create_model()
    model.fit(features_enc, labels_enc, nb_epoch=10, batch_size=32,verbose=1)
    return model

def validate_nn(x,y,model):
    labels_enc = label_encoder.fit_transform(y)
    features_enc = vectorizer.fit_transform(feat_extract(x))
    loss_and_metrics = model.evaluate(features_enc,labels_enc, batch_size=32)
    return loss_and_metrics


# In[24]:

data_sequence, data_labels, meta_info = preprocess_pipeline('cyto.fasta', 'mito.fasta','nucleus.fasta','secreted.fasta')
train_x, val_x, train_y, val_y = train_test_split(data_sequence,data_labels,test_size=0.3,random_state=3)

#rf = train(train_x,train_y)
#pred_y = validate(val_x,rf)
nn = train_nn(train_x,train_y)
#print(validate_nn(val_x,val_y,nn))
#print(meta_info)

#df = pd.DataFrame(meta_info)

#print(df)
#with open('ola.csv','w') as f:
#    df.to_csv(f)
#df.groupby("organism").count()


# ## Validation

# In[6]:

cm = confusion_matrix(val_y,pred_y)
stats =  precision_recall_fscore_support(val_y,pred_y)

stats = pd.DataFrame(data=np.transpose(np.array(stats[0:3])),columns=['precision','recall','f1'])
print(cm)

display(stats)


# In[ ]:

# #ola = feat_extract(val_x)
# #ola[23]
# count = 0
# for i in range(0,len(val_y)):
#     if val_y[i] == pred_y[i]:
#         count += 1
# print(count/len(val_y))
# #len(pred_y)


# ## Test set

# In[ ]:

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

print('a')
print('b')


# # # Playground
#
# # In[ ]:
#
# #test_blind('blind.fasta',rf)
#
#
# # In[ ]:
#
# data_sequence, data_labels = preprocess_pipeline('cyto.fasta', 'mito.fasta','nucleus.fasta','secreted.fasta')
# train_x, val_x, train_y, val_y = train_test_split(data_sequence,data_labels,test_size=0.3,random_state=3)
#
# hist_train = Counter(train_y)
# hist_val = Counter(val_y)
#
# print(hist_train)
# print(hist_val)
#
#
# # In[ ]:
#
# hist = Counter([len(x) for x in data_sequence])
# #df = pd.DataFrame(hist,index=[0])
# #df = pd.DataFrame.from_dict(hist,orient='index')
# transposed = np.array(list(hist.items())).T
# x, y = transposed
#
# plt.plot(x,y)
# plt.show()
#
#
# # In[ ]:
#
# vectorizer.fit_transform(feat_extract(train_x)).shape[0]
#
#
# # In[ ]:
#
# with open('cyto.fasta') as fasta_file:  # Will close handle cleanly
#     identifiers = []
#     lengths = []
#     for seq_record in SeqIO.parse(fasta_file, 'fasta'):  # (generator)
#         identifiers.append(seq_record.id)
#         lengths.append(len(seq_record.seq))
#
#
# # In[82]:
#
# p=re.compile("OS=([0-9a-zA-Z_\s\(\)\-\/,\.\>\:\']+) (?:GN|PE)")#
# meta_info = p.search('OS=Penicillium funiculosum PE=1 SV=1')
# meta_info.group(1)
#
#
# # In[51]:
#
# import re
# (\w+\|\w+)\|(\w+\s[0-90-9a-zA-Z_\s\(\)\-\/]+)OS=([0-90-9a-zA-Z_\s\(\)\-\/]+)GN=([0-90-9a-zA-Z_\s-]+)PE=([0-9])+\s[SV=]+([0-9])
#
#
# # In[125]:
#
# label_encoder.fit_transform(train_y)
#
#
# # In[129]:
#
# train_y[-4:-1]
#
#
# # In[20]:
#
# nn.predict(vectorizer.transform(feat_extract(['MCSLGLFPPPPPRGQVTLYEHNNELVTGSSYESPPPDFRGQWINLPVLQLTKDPLKTPGRLDHGTRTAFIHHREQVWKRCINIWRDVGLFGVLNEIANSEEEVFEWVKTASGWALALCRWASSLHGSLFPHLSLRSEDLIAEFAQVTNWSSCCLRVFAWHPHTNKFAVALLDDSVRVYNASSTIVPSLKHRLQRNVASLAWKPLSASVLAVACQSCILIWTLDPTSLSTRPSSGCAQVLSHPGHTPVTSLAWAPSGGRLLSASPVDAAIRVWDVSTETCVPLPWFRGGGVTNLLWSPDGSKILATTPSAVFRVWEAQMWTCERWPTLSGRCQTGCWSPDGSRLLFTVLGEPLIYSLSFPERCGEGKGCVGGAKSATIVADLSETTIQTPDGEERLGGEAHSMVWDPSGERLAVLMKGKPRVQDGKPVILLFRTRNSPVFELLPCGIIQGEPGAQPQLITFHPSFNKGALLSVGWSTGRIAHIPLYFVNAQFPRFSPVLGRAQEPPAGGGGSIHDLPLFTETSPTSAPWDPLPGPPPVLPHSPHSHL'])))
#
#
# # In[ ]:



