import os


def preprocess_pipeline(*files):
    data_features = []
    data_labels = []
    sequence = ''
    for file in files:
        label = os.path.splitext(file)[0]
        f = open(file, "r")
        for line in f:
            line = line.rstrip('\n')
            if line[0] != '>':
                sequence += line
            else:
                data_features.append(sequence)
                data_labels.append(label)
                sequence = ''



    return data_features, data_labels

data_features, data_labels = preprocess_pipeline('cyto.fasta', 'mito.fasta','nucleus.fasta','secreted.fasta')

print(len(data_labels))

