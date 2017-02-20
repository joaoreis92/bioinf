import os
from  sklearn import preprocessing
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix



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


#Preparing input
data_sequence, data_labels = preprocess_pipeline('cyto.fasta', 'mito.fasta','nucleus.fasta','secreted.fasta')

train_X = data_sequence
train_Y = data_labels


label_encoder = preprocessing.LabelEncoder()
vectorizer = DictVectorizer()

train_labels_enc = label_encoder.fit_transform(train_Y)
train_features_enc = vectorizer.fit_transform(feat_extract(train_X))

#Model
rf = RandomForestClassifier(class_weight='balanced')
rf.fit(train_features_enc, train_labels_enc)

#Prediction
predicts = rf.predict(train_features_enc)
labels_predicted = label_encoder.inverse_transform(predicts)

#Metric
cm = confusion_matrix(train_Y,labels_predicted)
print(f1_score(train_labels_enc,predicts,average='micro'))
print(cm)


print(len(data_labels))

