import os
from  sklearn import preprocessing
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import numpy as np


p=re.compile("|\w+\s(.)OS=([0-9a-zA-Z_\s\(\)\-\/,\.\>\:\']+)(?:\sGN|\sPE)")
def preprocess_pipeline(*files):
    data_features = []
    data_labels = []
    sequence = ''
    for file in files:
        label = os.path.splitext(file)[0]
        f = open(file, "r")
        next(f)
        for line in f:
            line = line.rstrip('\n')
            if line[0] != '>':
                sequence += line
            else:
                data_features.append(sequence)
                data_labels.append(label)
                sequence = ''



    return data_features, data_labels

def feat_extract(sequences):
    dict_feat = []
    for sequence in sequences:
        sequence_feat = defaultdict(float)
        sequence_len = len(sequence)

        sequence_feat["sequence_length = {0}".format(sequence_len)] = 1
        dict_feat.append(sequence_feat)
    return dict_feat


def train(x,y):

    return model

def validate(x,model):

    return predictions

#Preparing input
data_sequence, data_labels = preprocess_pipeline('cyto.fasta', 'mito.fasta','nucleus.fasta','secreted.fasta')

label_encoder = preprocessing.LabelEncoder()
vectorizer = DictVectorizer()

labels_enc = label_encoder.fit_transform(train_Y)
features_enc = vectorizer.fit_transform(feat_extract(train_X))

train_features_enc, test_features_enc, train_labels_enc, test_labels_enc = train_test_split(features_enc,labels_enc,test_size=0.3)


#Model
rf = RandomForestClassifier(class_weight='balanced')
rf.fit(train_features_enc, train_labels_enc)

#Prediction
predicts = rf.predict_proba(test_features_enc)
predicts_label = np.argmax(predicts,1)
labels_predicted = label_encoder.inverse_transform(predicts_label)
label_and_confidence = list(zip(labels_predicted,np.amax(predicts,1)))

#Metric
cm = confusion_matrix(train_Y,labels_predicted)
stats =  precision_recall_fscore_support(label_encoder.inverse_transform(train_Y),labels_predicted)
print(f1_score(test_labels_enc,predicts_label,average='micro'))
print(cm)

print(stats)
print(len(data_labels))

#fazer train e predict em duas funcs. 

