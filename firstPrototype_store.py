
# coding: utf-8

# In[13]:


# import numpy as np
# import functools
# import sets
# import tensorflow as tf

# output = [1,2,3]
# temp = np.array(output)
# output = np.zeros((len(output), 4))
# output[np.arange(len(output)), temp] = 1
# print(output)

# a = [[[1,2],[3,4]], [[5,6],[7,8]]]

# print(a)
# input = [list(map(list, zip(*i))) for i in a]
# print(input)
# import sets
# def get_dataset():
#     """Read dataset and flatten images."""
#     dataset = sets.Ocr()
#     dataset = sets.OneHot(dataset.target, depth=2)(dataset, columns=['target'])
#     dataset['data'] = dataset.data.reshape(
#         dataset.data.shape[:-2] + (-1,)).astype(float)
#     train, test = sets.Split(0.66)(dataset)
#     return train, test

# a, b = get_dataset()
# batch = a.sample(10)
# print(batch.target)


# In[22]:


import os
import re
import glob 
import matplotlib.pyplot as plt
import numpy as np
from os.path import basename
import audiosegment
from multiprocessing import Pool
modulePath = 'ChristiansPythonLibrary/src' 
import sys
import numpy
sys.path.append(modulePath)
import generalUtility
import dspUtil
import matplotlibUtil
import librosa
import pickle







#Constant
EMOTION_ANNOTATORS = {'anger': 0, 'happiness' : 1, 'sadness' : 2, 'neutral' : 3, 'frustration' : 4, 'excited': 5,
           'fear' : 6,'surprise' : 7,'disgust' : 8, 'other' : 9}

EMOTION = {'ang': 0, 'hap' : 1, 'sad' : 2, 'neu' : 3, 'fru' : 4, 'exc': 5,
           'fea' : 6,'sur' : 7,'dis' : 8, 'oth' : 9, 'xxx':10}

METHOD = {'audio_feature':0, 'LSTM':1}

#Method for classification
method = METHOD['LSTM']

#If data is processed and saved into files, just reload, dont need to re-process
isRawDataProcessed = True

#Development mode. Only run with small data.
dev = False



#Define class
class Input:
    ##spectral, prosody, erergy are dict type
    def __init__(self, spectral, prosody, energy, spectrogram):
        self.spectral = spectral
        self.prosody = prosody
        self.energy = energy
        self.spectrogram = spectrogram
        
    def print(self):
        print("spectral  features: ", spectral)
        print("prosody features: ", prosody)
        print("energy: ", energy)
        print("spectrogram: ", spectrogram)
        
    def input2Vec(self, onlySpectrogram):
        if (onlySpectrogram ==  False):
            features = []
            s = list(self.spectral.values())
            p = list(self.prosody.values())
            e = list(self.energy.values())
            [features.extend(x) for x in [s, p, e]]
            return features
        else :
            return self.spectrogram
    
class Output:
    def __init__(self, duration, code, category_origin, category_evaluation, attribute):
        self.duration = duration
        self.code = code
        self.category_origin = category_origin
        self.category_evaluation = category_evaluation
        self.attribute = attribute
        
     
    def print(self):
        print("duration: ", self.duration)
        print("code: ", self.code)
        print("category_origin: ", self.category_origin)
        print("category_evaluation: ", self.category_evaluation)
        print("attribute: ", self.attribute)
        
    def output2Vec(self):
        emotion = EMOTION[self.category_origin]
        return emotion
    
    
    
#Functions for get features from audio file
def amp2Db(samples):
    dbs = []
    for  x in samples:
        if x < 0:
            v = - dspUtil.rmsToDb(np.abs(x))
        elif x == 0:
            v = 0
        else :
            v = dspUtil.rmsToDb(np.abs(x))
        dbs.append(v)
    return dbs

def getF0Features(file):
    features = {}
    sound = audiosegment.from_file(file)
    voiced = sound.filter_silence(duration_s=0.2)
    frame_rate = sound.frame_rate
    frames = sound.dice(0.032)

    f0s = []
    for f in frames:
        f0 = dspUtil.calculateF0once(amp2Db(f.get_array_of_samples()), frame_rate)
        if(f0 != 0):
            f0s.append(f0)
    
    features['f0_min'] = np.min(f0s)
    features['f0_max'] = np.max(f0s)
    features['f0_range'] = np.max(f0s) - np.min(f0s)
    features['f0_mean'] = np.mean(f0s)
    features['f0_median'] = np.median(f0s)
    features['f0_25th'] = np.percentile(f0s, 25)
    features['f0_75th'] = np.percentile(f0s, 75)
    features['f0_std'] = np.std(f0s)
    
  
    return features

def getEnergyFeatures(file):
    features = {}
    sound = audiosegment.from_file(file)
    voiced = sound.filter_silence(duration_s=0.2)
    samples = voiced.get_array_of_samples()
    frame_rate = sound.frame_rate
    frames = sound.dice(0.032)
    
    e = []
    for f in frames:
        e.append(np.abs(f.max_dBFS))
    
    
    features['energy_min'] = np.min(e)
    features['energy_max'] = np.max(e)
    features['energy_range'] = np.max(e) - np.min(e)
    features['energy_mean'] = np.mean(e)
    features['energy_median'] = np.median(e)
    features['energy_25th'] = np.percentile(e, 25)
    features['energy_75th'] = np.percentile(e, 75)
    features['energy_std'] = np.std(e)   

    return features
    
def audio2Features(file):
    spectral = {}
    prosody = {}
    energy = {}
    try:
        prosody = getF0Features(file)
        energy = getEnergyFeatures(file)
        
        y, sr = librosa.load(file)
        spectrogram = librosa.stft(y)
        spectrogram = np.abs(spectrogram)
        #To be continued....
    
        return Input(spectral, prosody, energy, spectrogram)
    except Exception as e:
        print(e)
        
        
#Function for getting input vector and corresponding output      
def parallel_task(d0, d1):
    print("task...")
    # Each input diectory contains many file
    # This fucntion will walk through all valid 'wav'files in this directory and get features like engergy, frequency...
    def parseInput(dir):
        dicts = {} 
        for f in os.listdir(dir):
            if not f.startswith(".") and os.path.splitext(f)[1] == ".wav":
                dicts[os.path.splitext(f)[0]] = audio2Features(dir + "/" + f)


        return dicts
    
    # Each output file contains label of many diffrent 'wav' file.
    # This function will parse content of text file using 'regrex'. Then turn it into label
    def parseOutput(file):
        dict_namefile_output = {}
        # Open file to get all contents excepts the first line.
        f = open(file, 'r')
        content = ""
        index = 0
        for line in f:
            index = index + 1
            if index == 1:
                continue
            content  = content + line

        # Find all matched patterns in the content
        ps = re.findall(r'\[.*?\)\n\n', content, re.DOTALL)

        # Parse each matched pattern into  'Output' object
        try:
            for p in ps:
                ls = p.split("\n")
                ls = list(filter(lambda x: len(x) > 0 ,ls))

                # Split elements of the first line which looks like : 
                # [147.0300 - 151.7101]	Ses01F_impro02_M012	neu	[2.5000, 2.0000, 2.0000]
                ele_line0 = re.search(r'(\[.*?\])(\s)(.*?)(\s)(.*?)(\s)(\[.*?\])', ls[0]).groups()

                # Split time components which looks like:
                # [147.0300 - 151.7101]
                time_dur = ele_line0[0]
                ele_time_dur = re.findall(r"[-+]?\d*\.\d+|\d+", time_dur)
                ele_time_dur = [float(x) for x in ele_time_dur]

                # Get code and category_origin which looks like:
                # Code: Ses01F_impro02_M012
                # Category_origin: neu
                code = ele_line0[2]
                category_origin = ele_line0[4]

                # Split attribute components which looks like:
                # [2.5000, 2.0000, 2.0000]
                attribute = ele_line0[6]
                ele_attribute = re.findall(r"[-+]?\d*\.\d+|\d+", attribute)
                ele_attribute = [float(x) for x in ele_attribute]

                # Get categorial_evaluation:
                lines_categorical = list(filter(lambda x : x[0] == 'C', ls))
                rex = re.compile(r'C.*?:(\s)(.*?)(\s)\(.*?\)')

                category_evaluation = []
                for l in lines_categorical:
                    elements = rex.search(l).groups()
                    cat = elements[1]
                    cat = cat.split(";")
                    cat = map(lambda x: x.lstrip(), cat)
                    cat = list(filter(lambda x: len(x)>0, cat))
                    category_evaluation.extend(cat)


                # Make list distinct
                category_evaluation = list(set(category_evaluation))
                
                

                # Make dict {name_file : parsed_output}
                dict_namefile_output[code] = Output(ele_time_dur, code, category_origin, category_evaluation, ele_attribute)
            return dict_namefile_output
        except Exception as e:
            print(e)


    ### Parse input and output files and get input and output as vector
    dicts_in = parseInput(d0)
    dicts_out = parseOutput(d1)
    in_out = []
    
    keys = list(dicts_in.keys())
    for key in keys:
        if(dicts_out[key].category_origin != 'xxx'):
            if (method == METHOD['LSTM']):
                in_out.append((dicts_in[key].input2Vec(onlySpectrogram=True), dicts_out[key].output2Vec()))
            else:
                in_out.append((dicts_in[key].input2Vec(onlySpectrogram=False), dicts_out[key].output2Vec()))
    return in_out
    
    
def createInput_Output():
    ### Get directories of input and output
    DATA_DIR = "IEMOCAP_full_release"
    NUM_SESSION = 5
    input_output = []
    for i in range (1, NUM_SESSION + 1):
        name_session = "Session" + str(i)
        root_dir_of_wav = DATA_DIR + "/" + name_session + "/sentences" + "/wav"
        root_dir_of_labels = DATA_DIR + "/" + name_session + "/dialog" + "/EmoEvaluation"

        for x in os.walk(root_dir_of_wav):
            if(x[0] == root_dir_of_wav):
                dirs_of_wav = x[1]
                index = -1
            else:
                index = index + 1
                input_output.append((x[0], root_dir_of_labels + "/" + dirs_of_wav[index] + ".txt"))
                
    
    ds = input_output
    in_out = []
    input = []
    out = []
    
    # Multi processing
    with Pool(processes=8) as pool:
         in_out = pool.starmap(parallel_task, ds)
   
    r = []
    for e in in_out:
        r = r + e
    
    input = [x[0] for x in r]
    out = [x[1] for x in r]
    print("Finished creating input output into txt file")
    return (input, out)
 


#If have not processed data yet then process, otherwise loading data from file.
if isRawDataProcessed == False:

    ##Get input, normalize input, get output
    input, output = createInput_Output()
    output = np.array(output)
    
    if(method == METHOD['audio_feature']):
        input = np.array(input)
        input = input / input.max(axis=0)
        filehandlerInput = open('input0.obj', 'wb')
        filehandlerOutput = open('output0.obj', 'wb')
    elif(method == METHOD['LSTM']):
        # After this operator, each sample will be a 2-D array, Each row includes magnitude energy values in range of frquencies
        # Rows will have the same length in all samples.
        # Each sample will have different number of rows beacause their difference of length in seconds 
        #input = [list(map(list, zip(*i))) for i in input]
        filehandlerInput = open('input1.obj', 'wb')
        filehandlerOutput = open('output1.obj', 'wb')

        
    pickle.dump(input, filehandlerInput)
    pickle.dump(output, filehandlerOutput)
    print("Finish write processed data (input, output) to file!!!")
    
    


# In[2]:


#### Training using LSTM and CNN


# Example for my blog post at:
# https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/
import functools
import sets
import tensorflow as tf


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


class SequenceClassification:

    def __init__(self, data, target, dropout, num_hidden=200, num_layers=1):
        self.data = data
        self.target = target
        self.dropout = dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction
        self.error
        self.optimize


    

    @lazy_property
    def prediction(self):
        # Recurrent network.
        network = tf.contrib.rnn.GRUCell(self._num_hidden)
        network = tf.contrib.rnn.DropoutWrapper(
            network, output_keep_prob=self.dropout)
        network = tf.contrib.rnn.MultiRNNCell([network] * self._num_layers)
        output, _ = tf.nn.dynamic_rnn(network, self.data, dtype=tf.float32)
        # Select last output.
    

        print("shape of output:(batch_size,max_time,feture ):  ",output.get_shape())
        output = tf.transpose(output, [1, 0, 2])
        
        last = output[-1]
        # Softmax layer.
        weight, bias = self._weight_and_bias(
            self._num_hidden, int(self.target.get_shape()[1]))
        prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        return prediction

    @lazy_property
    def cost(self):
        cross_entropy = -tf.reduce_sum(self.target * tf.log(self.prediction))
        return cross_entropy

    @lazy_property
    def optimize(self):
        learning_rate = 0.003
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        #print("self.prediction: ", self.prediction)
        #mistakes = tf.not_equal(
         #   tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        #return tf.reduce_mean(tf.cast(mistakes, tf.float32))
        mistake =  tf.not_equal(tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
        a = tf.reduce_mean(tf.cast(mistake, tf.float32))
        return 1 -a

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)




##Running
if (method == METHOD['LSTM']):
    
    ##Loading  data from files
    filehandlerInput = open('input1.obj', 'rb')
    filehandlerOutput = open('output1.obj', 'rb')
    input = pickle.load(filehandlerInput)
    output = pickle.load(filehandlerOutput)
    print("input size: ", len(input))    
    #if in development mode, just use the small data!
    if (dev):
        data_size = 100
    else:
        data_size = len(input)
    
    input = [list(map(list, zip(*i))) for i in input[0:data_size]]

    #Normalize input
    max_val = -1
    for i in input:
        b = [max(x) for x in i]
        c = max(b)
        if c >  max_val:
            max_val = c

    print("This is max_value of input: ", max_val)
    
    for i in range(0, len(input)):
        for j in range(0, len(input[i])):
            input[i][j] = [x / max_val for x in input[i][j]]

    print("Finish normalize input.")

    output = output[0:data_size]
    
    #One-hot encoding output
    temp = np.array(output)
    output = np.zeros((len(output), 10))
    output[np.arange(len(output)), temp] = 1
    
    # Split train, test
    trainlen = int(0.8*data_size)
    train_in, train_out = input[0:trainlen], output[0:trainlen]
    test_in, test_out = input[trainlen:data_size], output[trainlen:data_size]
    print("Finished split data!")
    
    
    num_classes = 10
    row_size = len(input[0][0])
    data = tf.placeholder(tf.float32, [None, None, row_size])
    target = tf.placeholder(tf.float32, [None, num_classes])
    dropout = tf.placeholder(tf.float32)
    model = SequenceClassification(data, target, dropout)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(10):
        print("epoch!")
        for i in range(len(train_in)):
            #print("train.....")
            sess.run(model.optimize, {
                data: [train_in[i]], target: [train_out[i]], dropout: 0.5})
    #    error = sess.run(model.error,{data: [test_in[0]], target: [test_out[0]], dropout: 1})
        right = 0
        for j in range(0, len(test_in)):
            right = right + sess.run(model.error,{data: [test_in[j]], target: [test_out[j]], dropout:1})
        print("accuracy: ", right / len(test_in))
        #print('Epoch {:2d} error {:3.1f}%'.format(epoch + 1, 100 * error))
    
    
    








# In[21]:


#### Training based on features of audio

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sknn.mlp import Classifier, Layer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix


def training(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    kf = KFold(n_splits=10, random_state=None, shuffle=False)
    i_fold = 0
    accuracy_train_results = []
    accuracy_valid_results = []

    for train_index, valid_index in kf.split(X_train):
        i_fold = i_fold + 1
        
        x_train_sub, x_valid_sub = X_train[train_index], X_train[valid_index]
        y_train_sub, y_valid_sub = y_train[train_index], y_train[valid_index]
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,), random_state=1)
        clf.fit(x_train_sub, y_train_sub)
        
        score = clf.score(x_train_sub, y_train_sub)
        score1 = clf.score(x_valid_sub, y_valid_sub)
        accuracy_train_results.append(score)
        accuracy_valid_results.append(score1)
        
        print("Score of training set: ", score)
        print("Score of validation set: ", score1)
     
       
    
    avg_accuracy_train_result = np.sum(accuracy_train_results) / len(accuracy_train_results)
    avg_accuracy_valid_result = np.sum(accuracy_valid_results) / len(accuracy_valid_results)
    print("Average accuracy training set, std:", avg_accuracy_train_result, " ",          np.std(accuracy_train_results))
    print("Average accuracy validation set, std:", avg_accuracy_valid_result," ",           np.std(accuracy_valid_results))     
    
    clf.fit(X_train, y_train)
   

    predicts = clf.predict(X_test)
    pro = clf.predict_proba(X_test)
 #   print("predicts: ", predicts)
 #   print("prob: ", pro[0])
    score_test = clf.score(X_test, y_test)
    print("\nScore for test set: ", score_test)
    print ("\nConfusion matrix:..................... ")
    matrix = confusion_matrix(y_test, predicts)
    matrix_ratio = matrix/matrix.sum(1, keepdims=True)
    print(matrix)
    print("\n", "Confusion matrix ratio:")
    print(matrix_ratio)
    print("\n", "Horizontal of confusion matrix ratio:")
    hor = [matrix_ratio[i,i] for i in range(0, len(matrix_ratio))]
    print(hor)
 

if (method == METHOD['audio_feature']):

    ##Loading  data from files
    filehandlerInput = open('input0.obj', 'rb')
    filehandlerOutput = open('output0.obj', 'rb')
    input = pickle.load(filehandlerInput)
    output = pickle.load(filehandlerOutput)

    # Get quantiry of each label
    y = np.bincount(output)
    ii = np.nonzero(y)[0]
    a = list(zip(ii,y[ii]))
    print("EMOTION_ANNOTATE: ", EMOTION_ANNOTATORS)
    print("\nThe quantity of each label: ", a, "\n")

    #Remove labels that have small quantity.
    indices = [] 
    for i in range(0, len(output)):
        if output[i] >= 6:
            indices.append(i)
    input = np.delete(input, indices, axis = 0)
    output = np.delete(output, indices)

    #Training and testing
    training(input, output)





# In[ ]:





# In[16]:





# In[ ]:




