#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import string
import pickle


# In[ ]:


#This function parses and preprocesses the data and puts them into data structures that can be utilized for the further processing.
#Input : path of training_data_file/ validation_data_file/ testing_data_file ( These files can be downloaded  from link provided in README)
#output: list of instances, where each instance has 4 lists, representing [words][pos][phrase_chunks][labels]
def process_data_file(file_name):
    f = open(file_name, "r")
    tokens = []
    pos = []
    chunks = []
    ner_lables = []
    data = []
    for line in  f:
        if line == "\n" or line.startswith('-DOCSTART'):
            if len(tokens) > 0:
                sent = []
                sent.append(tokens)
                sent.append(pos)
                sent.append(chunks)
                sent.append(ner_lables)
                data.append(sent)

                # refresh the arrays
                tokens = []
                pos = []
                chunks = []
                ner_lables = []

        else:
            try:
                parts = line.strip("\n").split(" ")
                tokens.append(parts[0])
                pos.append(parts[1])
                chunks.append(parts[2])
                ner_lables.append(parts[3])
            except Exception as e:
                print(e)

    #data = data[1:]
    data = np.asarray(data)
    print(data.shape)
    return data


# In[54]:


data = process_data_file("NER-datasets/CONLL2003/train.txt")
valid_data = process_data_file("NER-datasets/CONLL2003/valid.txt")
test_data = process_data_file("NER-datasets/CONLL2003/test.txt")


# In[211]:


data[2]


# In[4]:


# this function divides the data into batches equal size
# Input: training data (array of instances)
# Output: list of batches that are of equal size
def createEqualBatches(data):
    n_batches = 100
    batch_size = len(data) // n_batches
    #print(batch_size)
    indices = []
    for i in range(n_batches):
        indices.append(batch_size*(i+1))
    #print(indices)
    
    batches = []
    batch_len = []
    z = 0
    start = 0
    print(len(indices))
    for end in indices:
        #print("start, end", start, end)
        batches.append(data[start:end])
        start = end

    return batches


# In[112]:


batches = createEqualBatches(data)
print(len(batches))


# In[121]:


print(data[1])


# In[4]:


# Intializing some global vars
# Reads from the glove embedding file and loads the embeddings into a dictionary and stores them in a pickle file
#Input: glove embeddings file
#Output: pickled embeddings
# TO DO : MAKE A DIRECTORY "embeddings"
Embeddings = {}
word_emb_dim  = 50
embedding_file = "embeddings/glove.6B.50d.txt"
def LoadEmbeddings(embedding_file):
    global Embeddings, word_emb_dim
    f = open(embedding_file, "r", encoding= "utf-8", errors= "ignore")
    for line in f :
        tokens = line.strip("\n").split()
        word = tokens[0].lower()
        vec = tokens[1:]
        vec = " ". join(vec)
        Embeddings[word] = vec
    Embeddings["zero_vec"] = "0.0 " * word_emb_dim
    Embeddings["zero_vec"] = Embeddings["zero_vec"].rstrip()
    f.close()
    
    g = open ("embeddings/EmbedDict.pkl", "wb")
    pickle.dump(Embeddings, g)
    g.close()


# In[5]:


LoadEmbeddings("embeddings/glove.6B.50d.txt")


# In[55]:


GloveEmbeddings = {}
pos_emebddings = {}
max_words = 15
case_emb_dim = 8
pos_emb_dim = 40
num_classes =  10

# Loads the word embeddings into a dictionary
def load_word_embeddings():
    #load the embeddings from the pickle file
    global GloveEmbeddings
    f = open("embeddings/EmbedDict.pkl", "rb")
    GloveEmbeddings = pickle.load(f)
    f.close()

# gets the word features
# Input: List of words and left_over is an integer that says how much padding is required
# Output: list of feature vectors for the words/ sentence ( with pads)
def get_word_feat_vecs(words, left_over):
    global max_words, GloveEmbeddings
    #remove extra words
    #print(words, left_over)
    if left_over < 0:
        words = words[:max_words]
    elif(left_over > 0): #padding req
        for i in range(left_over):
            words.append("zero_vec")
    
    # now obtain the feature vector
    feat_vec = []
    for word in words:
        word = word.lower()
        try:
            glove_vec = []
            splits = GloveEmbeddings[word].split()
            #print(splits)
            for v in splits:
                glove_vec.append(float(v))
            #glove_vec = [float(v for v in GloveEmbeddings[word].split())]
        except:
            glove_vec = []
            splits = GloveEmbeddings["zero_vec"].split()
            #print(splits)
            for v in splits:
                glove_vec.append(float(v))
        feat_vec.append(glove_vec)
    return feat_vec

#Gets the case embedding (Also one of the features)
#Input : A single word
#Output: Case embedding for that
def  get_case_embeddings(word):
    global case_emb_dim
    case_map = {'numeric': 0, 'all_lower': 1, 'all_upper': 2, 'initial_upper': 3, 'other': 4, 'mainly_numeric': 5,
                    'contains_digit': 6, 'PADDED_TOKEN': 7}
    num_of_digits = 0
    for char in word:
        if char.isdigit():
            num_of_digits += 1
    number_part = float(num_of_digits/len(word))
    
    casing = 'other'
    
    if num_of_digits == len(word):
        casing = "numeric"
    elif number_part >= 0.5:
        casing = "mainly_numeric"
    elif num_of_digits > 0:
        casing = "contains_digit"
    elif word.isupper():
        casing = "all_upper"
    elif word.islower():
        casing = "all_lower"
    elif word[0].lower():
        casing = "initial_upper"
    elif word[0].lower() == "zero_vec":
        casing = "PADDING_TOKEN"
    
    case_vector = [0]*case_emb_dim
    case_vector[case_map[casing]] = 1
    
    return case_vector

###################### Create pickle files for  pos and label embeddings #####################

def create_pos_embeddings(data):
    print(len(data))
    pos_statistics = {}
    pos_set = []
    for instance in data:
        pos_tags = instance[1]
        #print(pos_tags)
        for tag in pos_tags:
            if tag in string.punctuation:
                tag = "PUNCT"
            try:
                pos_statistics[tag] = pos_statistics[tag] + 1
            except:
                pos_statistics[tag] = 1
    
    print(pos_statistics)
    for pos in pos_statistics:
        pos_set.append(pos)
    pos_set.append("PADDED_POS_TAG")
    
    print(pos_set)
    print(len(pos_set))
    pos_map = {}
    for pos_idx in range(len(pos_set)):
        pos_vec = [0]* len(pos_set)
        pos_vec[pos_idx] = 1
        pos_map[pos_set[pos_idx]] = pos_vec
    
    f = open("POSEmbeddings.pickle", "wb")
    pickle.dump(pos_map, f)
    f.close()
    
def create_label_embeddings(data):
    label_set = []
    label_statistics = {}
    for instance in data:
        labels = instance[3]
        for label in labels:
            try:
                label_statistics[label] = label_statistics[label] + 1
            except:
                label_statistics[label] = 1
    ############### COMMENT WHEN LABEL STATISTICS IS NOT NEEDED ##################
    print(label_statistics)
    total_count = 0
    for label in label_statistics:
        total_count += label_statistics[label]
    
    for label, count in label_statistics.items():
        print("{}: {}%".format(label, round((count/total_count)*100, 2)))
    #####################################################################################
#     for label in label_statistics:
#         label_set.append(label)
#     label_set.append("PADDED_LABEL")
#     print(label_set)
#     print(len(label_set))
    
#     label_map = {}
    
#     for label_idx in range(len(label_set)):
#         label_vec = [0] * len(label_set)
#         label_vec[label_idx] = 1
#         label_map[label_set[label_idx]] = label_vec
    
#     f = open("LabelEmbed.pickle", "wb")
#     pickle.dump(label_map, f)
#     f.close()
        


# In[56]:


# Gives the class statistics for Train_Data, Valid_Data, Test_Data
create_label_embeddings(data)
print("\n")
create_label_embeddings(valid_data)
print("\n")
create_label_embeddings(test_data)
print("\n")


# In[8]:


create_label_embeddings(data) # Label Statistics
create_pos_embeddings(data)


# In[221]:


#gives average length of the sentence in trainig  data, useful to fix on a sentence length
def average_sent_len(data):
    word_lens = []
    len_dict ={}
    for instance in data:
        words = instance[0]
        word_lens.append(len(words))
    print(float(sum(word_lens)/len(word_lens)))
    for length in word_lens:
        try:
            len_dict[length] += 1
        except:
            len_dict[length] = 1
    print(len_dict)

average_sent_len(data) #Average length of the snetences
        


# In[9]:


# Gives how many words are needded to be padded given max words a sentence can take is  15.
# Input: List of words [Represents a sentence]
# output: Padding required (Integer Value)
def remaining(words):
    global max_words
    word_count = len(words)
    left_over = max_words - word_count
    return left_over
# Gets label vectors
def get_label_vectors(data):
    global num_classes, max_words
    data_label_vectors = []
    f = open("LabelEmbed.pickle", "rb")
    label_map = pickle.load(f)
    f.close()
    
    for  instance in data:
        labels = instance[3]
        left_over = remaining(labels)
        if left_over < 0:
            #print("no pad")
            labels =  labels[:max_words]
        temp = []
        for label in labels:
            label_vec = label_map[label]
            temp.append(label_vec)
        if left_over > 0:
            #print("Padded")
            label_vec = [0] * num_classes
            label_vec[9] = 1
            for i in range(left_over):
                temp.append(label_vec)
        #print(len(temp))
        #print(np.asarray(temp).shape)
        data_label_vectors.append(temp)
    
    return data_label_vectors


# In[10]:


data_label_vecs = get_label_vectors(data)
a = np.asarray(data_label_vecs)
print(a.shape)


# In[11]:


#gives case features
#Input: list of words, padding_req
#Output: list of case vectors

def get_case_feat_vecs(words, left_over):
    global case_emb_dim, max_words
    case_vecs = []
    if left_over < 0:
        words = words[:max_words]
    for word in words:
        case_embedding = get_case_embeddings(word)
        case_vecs.append(case_embedding)
    if left_over > 0:
        case_embedding = [0] * case_emb_dim
        case_embedding[7] = 1
        for i in range(left_over):
            case_vecs.append(case_embedding)
    return case_vecs
    
    


# In[85]:


print(len(get_case_feat_vecs(data)))


# In[12]:


def load_pos_embeddings():
    global pos_embeddings
    f = open("POSEmbeddings.pickle", "rb")
    pos_embeddings = pickle.load(f)
    f.close()
    
def get_pos_feat_vecs(pos, left_over):
    global pos_embeddings, pos_emb_dim, max_words
    pos_vecs = []
    if left_over < 0:
        pos = pos[:max_words]
    for tag in pos:
        if tag in string.punctuation:
            tag = "PUNCT"
        pos_vec = pos_embeddings[tag]
        pos_vecs.append(pos_vec)
    if left_over > 0:
        pos_vec = pos_embeddings["PADDED_POS_TAG"]
        for i in range(left_over):
            pos_vecs.append(pos_vec)
    return pos_vecs
        
    


# In[87]:


load_pos_embeddings()


# In[13]:


def get_feature_vectors(data):
    ## This function gets the feature vectors (word, case and pos) for the entire training data ###
    
    # get feats for words
    global max_words
    data_word_feats = []
    data_case_feats = []
    data_pos_feats = []
    load_word_embeddings()
    load_pos_embeddings()
    for instance in data:
        tokens = instance[0]
        pos = instance[1]
#         if (len(tokens) != len(pos)):
#             print("WRONG")
#             print(tokens)
#             print(pos)
        left_over = remaining(tokens)
        
       
        #get feats for POS
        pos_feat_vecs = get_pos_feat_vecs(pos, left_over)
        data_pos_feats.append(pos_feat_vecs)
        #get feats for case
        case_feat_vecs = get_case_feat_vecs(tokens, left_over)
        data_case_feats.append(case_feat_vecs)
        #get for words
        words_feat_vecs = get_word_feat_vecs(tokens, left_over)
        data_word_feats.append(words_feat_vecs)
        
       
    
    return data_word_feats, data_case_feats, data_pos_feats
        
        


# In[14]:


from keras.models import Sequential, Model, load_model, save_model
from keras.layers import Bidirectional, LSTM, Dense, Activation, Input, concatenate, TimeDistributed
def Build_Model(sent_max_words = 15, word_emb_dim = 50 , case_emb_dim = 7, pos_emb_dim = 46):
    lstm_dim = 200
    global num_classes
    #word input
    word_input = Input(shape=(sent_max_words, word_emb_dim))
    #case input
    case_input = Input(shape=(sent_max_words, case_emb_dim))
    #pos_input
    pos_input = Input(shape=(sent_max_words, pos_emb_dim))
    #Concatenate the three inputs
    merged_input = concatenate([word_input, case_input, pos_input])
    #pass the merged input to a BiLSTM
    lstm_output = Bidirectional(LSTM(lstm_dim, return_sequences=True, dropout = 0.2),merge_mode=None)(merged_input)
    merged_output = concatenate([lstm_output[0], lstm_output[1]], axis = 2)
    #Add a Time Distributed Layer
    output = TimeDistributed(Dense(num_classes, activation = "softmax"), name = "Softmax_Layer")(merged_output)
    
    ## Model ##
    model = Model(inputs = [word_input, case_input, pos_input], outputs = output)
    model.summary()
    
    return model
    
    
    
    


# In[15]:


Build_Model()


# In[108]:


# Do not RUN #
#get epoch wise train error and validation error
import numpy as np
def train_model(data):
    print(data.shape)
    global max_words, word_emb_dim, case_emb_dim, pos_emb_dim, num_classes
    total_epochs = 30
    batch_size = 149 #equal batches
    #batches = createEqualBatches(data)
    model = Build_Model(max_words, word_emb_dim, case_emb_dim, pos_emb_dim)
    model.compile(loss= "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    train_label_vectors = get_label_vectors(data)
    word_feats, case_feats, pos_feats =  get_feature_vectors(data)
    #print(len(word_feats), len(case_feats), len(pos_feats))
    
    print(len(train_label_vectors))
    
    #Metrics for each epoch
    word_feats = np.asarray(word_feats)
    case_feats = np.asarray(case_feats)
    pos_feats = np.asarray(pos_feats)
    train_label_vectors = np.asarray(train_label_vectors)
    print(word_feats.shape)
    print(case_feats.shape)
    print(pos_feats.shape)
    print(train_label_vectors.shape)
    model.fit([word_feats, case_feats, pos_feats], train_label_vectors, batch_size = 149, validation_split = 0.2, initial_epoch=0, epochs = total_epochs)
#     for epoch in  range(total_epochs):
#         model.fit([word_feats, case_feats, pos_feats], train_label_vectors, batch_size = 149, validation_split = 0)

        

train_model(data)

    
    


# In[21]:


#Parsing and preprocessing of the validation_data
#ALWAYS RUN BEFORE GETTING THE RESULTS AND TRAINING THE MODEL
data = process_data_file("NER-datasets/CONLL2003/train.txt")
valid_data = process_data_file("NER-datasets/CONLL2003/valid.txt")


# In[17]:


#FOR THE PURPOSE OF USING SKLEARN
def convert_to_scalar(label_vectors):
    scalar_vec = []
    for sentence in label_vectors:
        labels = []
        for word in sentence:
            #print(word)
            idx = np.argmax(word)
            labels.append(idx)
        scalar_vec.append(labels)
    scalar_vec = np.asarray(scalar_vec)
    
    return scalar_vec


# In[22]:


#get epoch wise train error and validation error
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, classification_report
def train_model(train_data, valid_data):
    print(train_data.shape)
    print(valid_data.shape)
    global max_words, word_emb_dim, case_emb_dim, pos_emb_dim, num_classes
    total_epochs = 50
    batch_size = 149 #equal batches
    #batches = createEqualBatches(data)
    model = Build_Model(max_words, word_emb_dim, case_emb_dim, pos_emb_dim)
    model.compile(loss= "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    train_label_vectors = get_label_vectors(train_data)
    train_word_feats, train_case_feats, train_pos_feats =  get_feature_vectors(train_data)
    
    valid_label_vectors = get_label_vectors(valid_data)
    valid_word_feats, valid_case_feats, valid_pos_feats =  get_feature_vectors(valid_data)
    
    
    #Conversion to numpy arrays (train)
    train_word_feats = np.asarray(train_word_feats)
    train_case_feats = np.asarray(train_case_feats)
    train_pos_feats = np.asarray(train_pos_feats)
    train_label_vectors = np.asarray(train_label_vectors)
    print(train_word_feats.shape, train_case_feats.shape, train_pos_feats.shape, train_label_vectors.shape)
    
    #conversion to numpy arrays (test)
    valid_word_feats = np.asarray(valid_word_feats)
    valid_case_feats = np.asarray(valid_case_feats)
    valid_pos_feats = np.asarray(valid_pos_feats)
    valid_label_vectors = np.asarray(valid_label_vectors)
    print(valid_word_feats.shape, valid_case_feats.shape, valid_pos_feats.shape, valid_label_vectors.shape)
    
    #valid_label_vectors =  convert_to_scalar(valid_label_vectors)
    
    
    
    best = 0.0
    model.fit([train_word_feats, train_case_feats, train_pos_feats], train_label_vectors, batch_size = batch_size, validation_data = ([valid_word_feats, valid_case_feats, valid_pos_feats],valid_label_vectors), initial_epoch=0, epochs = total_epochs)
    model.save("best_model2_ep_50.h5")
train_model(data, valid_data)


# In[57]:


#Parsing and preprocessing of the training_data
valid_data = process_data_file("NER-datasets/CONLL2003/valid.txt")
test_data = process_data_file("NER-datasets/CONLL2003/test.txt")


# In[59]:


def exlude_padded_class(scalar_label_vectors):
    new_scalar_label_vectors = []
    for instance in scalar_label_vectors:
        temp = []
        for label in instance:
            if label != 9:
                temp.append(label)
        new_scalar_label_vectors.append(temp)
    new_scalar_label_vectors = np.asarray(new_scalar_label_vectors)
    return new_scalar_label_vectors
    


# In[234]:


# FIRST RUN ON VALIDATION DATA ##### ###### NOT THE FINAL MODEL ###########
from sklearn.metrics import classification_report
def metric(model_file, test_data):
    model = load_model(model_file)
    test_label_vectors = get_label_vectors(test_data)
    test_word_feats, test_case_feats, test_pos_feats =  get_feature_vectors(test_data)
    predicted_label_vectors = model.predict(([test_word_feats, test_case_feats, test_pos_feats]))
    test_label_vectors =  convert_to_scalar(test_label_vectors)
    #test_label_vectors = exlude_padded_class(test_label_vectors)
    
    predicted_label_vectors = convert_to_scalar(predicted_label_vectors)
    
    #predicted_label_vectors = exlude_padded_class(predicted_label_vectors)
    #labels = ['B-ORG', 'O', 'B-MISC', 'B-PER', 'I-PER', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC', 'PADDED_LABEL']
    
    cf = confusion_matrix(test_label_vectors.ravel(), predicted_label_vectors.ravel())
    print(cf)
#     macro_score = f1_score(test_label_vectors.ravel(), predicted_label_vectors.ravel(), average='macro')
#     micro_score = f1_score(test_label_vectors.ravel(), predicted_label_vectors.ravel(), average='micro')
#     weighted_score = f1_score(test_label_vectors.ravel(), predicted_label_vectors.ravel(), average='weighted')
#     print(macro_score, micro_score, weighted_score)
    cf_report = classification_report(test_label_vectors.ravel(), predicted_label_vectors.ravel(), target_names= ['B-ORG', 'O', 'B-MISC', 'B-PER', 'I-PER', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC', 'PADDED_LABEL'])
    return cf_report

model_file = "best_model2.h5"
cf_report = metric(model_file,valid_data)
print(cf_report)


# In[239]:


#Removing the padded_token class the over_all precision, recall and f-score of the system is as follows:
Validation Data
Precision: 91.67
Recall: 90.12
F-score: 91.23


# In[45]:


def metric(model_file, test_data):
    model = load_model(model_file)
    test_label_vectors = get_label_vectors(test_data)
    test_word_feats, test_case_feats, test_pos_feats =  get_feature_vectors(test_data)
    predicted_label_vectors = model.predict(([test_word_feats, test_case_feats, test_pos_feats]))
    test_label_vectors =  convert_to_scalar(test_label_vectors)
    predicted_label_vectors = convert_to_scalar(predicted_label_vectors)
    cf = confusion_matrix(test_label_vectors.ravel(), predicted_label_vectors.ravel())
    print(cf)
#     macro_score = f1_score(test_label_vectors.ravel(), predicted_label_vectors.ravel(), average='macro')
#     micro_score = f1_score(test_label_vectors.ravel(), predicted_label_vectors.ravel(), average='micro')
#     weighted_score = f1_score(test_label_vectors.ravel(), predicted_label_vectors.ravel(), average='weighted')
#     print(macro_score, micro_score, weighted_score)
    print(classification_report(test_label_vectors.ravel(), predicted_label_vectors.ravel(), target_names= ['B-ORG', 'O', 'B-MISC', 'B-PER', 'I-PER', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC', 'PADDED_LABEL']))

model_file = "best_model2_ep_50.h5"
metric(model_file,test_data)


# In[ ]:


#Removing the padded_token class the over_all precision, recall and f-score of the system is as follows:
Test Data
# batch_size = 149, epochs = 30
Precision: 84.67
Recall: 85.56
F1-score: 85

# batch size = 256 , epochs = 30
Test Data
Precision: 84.3
Recall: 85.78
F1-score: 85.23

# batch size = 149, epochs = 50
Test Data
Precision: 84.78
Recall: 87.12
F1-score: 86


# In[25]:


import string
print(string.punctuation)


# In[33]:


def collapse_classes(ravel_label_vec):
    new_label_vec = []
    for label in ravel_label_vec:
        if label == 6 or label == 0:
            label = 0
        elif label == 7 or label == 2:
            label = 2
        elif label == 4 or label == 3:
            label = 3
        elif label == 8 or label == 5:
            label = 4
        elif label == 9:
            label = 5
        new_label_vec.append(label)
    new_label_vec = np.asarray(new_label_vec)
    return new_label_vec   


# In[53]:


def metric(model_file, test_data):
    model = load_model(model_file)
    test_label_vectors = get_label_vectors(test_data)
    test_word_feats, test_case_feats, test_pos_feats =  get_feature_vectors(test_data)
    predicted_label_vectors = model.predict(([test_word_feats, test_case_feats, test_pos_feats]))
    test_label_vectors =  convert_to_scalar(test_label_vectors)
    predicted_label_vectors = convert_to_scalar(predicted_label_vectors)
    
    test_label_vectors = collapse_classes(test_label_vectors.ravel())
    predicted_label_vectors = collapse_classes(predicted_label_vectors.ravel())
    cf = confusion_matrix(test_label_vectors.ravel(), predicted_label_vectors.ravel())
    print(cf)
    target_names = ['ORG', 'O', 'MISC', 'PER','LOC', 'PADDED_CLASS']
    print(classification_report(test_label_vectors, predicted_label_vectors, target_names = target_names))
    return cf

model_file = "best_model2_ep_50.h5"
cf = metric(model_file,test_data)


# In[ ]:


#for 50 epochs and a batch size of 149 (Overall System Precision, Recall and F1-score)
Test Data
Precision: 84.78
Recall: 87.12
F1-score: 86


# In[60]:


#Plot a confusion Matrix
from matplotlib import pyplot as plt
labels = ['B-ORG', 'O', 'B-MISC', 'B-PER','B-LOC', 'PADDED_CLASS']
#cm = confusion_matrix(y_test, pred, labels)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cf)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[ ]:




