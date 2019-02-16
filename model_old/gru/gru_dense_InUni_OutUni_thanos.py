
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
try:
    import cPickle as pickle
except:
    import pickle
import json
import itertools
import time


# In[2]:


# use cpu if model with loop
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""


# In[3]:


from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.layers import *
from keras.models import Model,load_model
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
import keras.backend as K
#K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=3, inter_op_parallelism_threads=3)))


# In[4]:


#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import seaborn as sns
#get_ipython().magic('matplotlib inline')
#mpl.rcParams['figure.figsize'] = (16,8)


# # Prediction t+15min des entrants  / stations 
# ## (RER / metro / tram / train de banlieue)
# 
# Objectifs :
# - mettre au point tester architecture de RNN
# - influence du contexte (variable calendaire)
# - prise en compte des serie temporelle voisines
# 
# Données :
# - ensemble d'apprentissage 2014 (cross-validation 80/20)
# - ensemble de test 4 trimestre de 2015
# - 30 stations autour du pole de la défense
# 
# 

#  * [Model multi](#modelmulti)
#  * [Model uni](#modeluni)
#  * [Model fusion](#modelfusion)
# 

# In[5]:


# model dense
def create_xy_dense(X, look_back=100):
    dataX, dataY = [], []
    for i in range(len(X)-look_back):
        y = X[i+look_back]
        x = X[i:(i+look_back)]
        dataX.append(x)
        dataY.append(y)
    return np.array(dataX), np.array(dataY)

# model time distributed dense
def create_xy_jump(X, look_back=100):
    dataX, dataY = [], []
    for i in range(len(X)-look_back):
        y = X[i+1:i+look_back+1]
        x = X[i:(i+look_back)]
        dataX.append(x)
        dataY.append(y)
    return np.array(dataX), np.array(dataY)

# model time distributed dense with end data
def create_xy_jump_end(X, look_back=100):
    #len(X)-1 because prediction timestep+1
    n = int(np.floor((len(X)-1)/look_back)*look_back+1)
    diff=len(X)-n

    try:
        dataX = np.reshape(X[diff+0:(n+diff-1),:],((n/look_back),look_back,X.shape[1]))
        dataY = np.reshape(X[diff+1:n+diff,:],((n/look_back),look_back,X.shape[1]))
    except:
        print("WARNING date?")
        dataX = np.reshape(X[diff+0:(n+diff-1)],((n/look_back),look_back))
        dataY = np.reshape(X[diff+1:n+diff],((n/look_back),look_back))
    
    return np.array(dataX), np.array(dataY)


# In[6]:


def yes_or_no(question):
    while "the answer is invalid":
        reply = str(input(question+' (y/n): ')).lower().strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False
        
def PickleRick(path,name,obj,force=False):
    if force:
        try:
            with open(path+name, 'wb') as f:
                pickle.dump(obj, f)
        except:
            os.makedirs(path)
            with open(path+name, 'wb') as f:
                pickle.dump(obj, f)
    else:
        try:
            if os.path.isfile(path+name):
                if yes_or_no("Do you want to erase file : "+path+name):
                    with open(path+name, 'wb') as f:
                        pickle.dump(obj, f)
            else:
                with open(path+name, 'wb') as f:
                    pickle.dump(obj, f)
        except:
            os.makedirs(path)
            if os.path.isfile(path+name):
                if yes_or_no("Do you want to erase file : "+path+name):
                    with open(path+name, 'wb') as f:
                        pickle.dump(obj, f)
            else:
                with open(path+name, 'wb') as f:
                    pickle.dump(my_model, f)


# # MAIN

# In[7]:




path_vals = "../../../data/valid_metro_15min_2015_2016_2017_sumpass_nodayfree.csv"
path_cont = "../../../data/2013-01-01-2019-01-01_new.csv"
path_info = "../../../data/station_info.csv"

Vals = pd.read_csv(path_vals)
Cont = pd.read_csv(path_cont)
Info = pd.read_csv(path_info)




def buildGRUUNI(drop, nbneurones, lag_window, len_time_series, len_context):
    vals=Input(shape=(lag_window,len_time_series,))
    xc=Input(shape=(lag_window,len_context,))
    yc=Input(shape=(len_context,))
    
    drop = Dropout(drop)(vals)
    gru  = []
    xh   = [] 
    x    = []
    xp   = []
    xpr  = []
    pred = []

    for i in range(len_time_series):
        xp.append(Lambda(lambda x:x[:,:,i])(drop)) #add values dropouted (drop) and lambda filter the station
        xpr.append(Reshape((lag_window,1))(xp[i])) #reshape dropouted values
        x.append(concatenate([xc,xpr[i]])) #add context and values
        #add gru layer with input=x
        gru.append(CuDNNGRU(nbneurones, return_sequences=False, input_shape=(lag_window, len_context+1))(x[i]))
        xh.append(concatenate([yc,gru[i]])) # add context plus gru output
        pred.append(Dense(1,activation='softplus')(xh[i]))
        
        
    predc=concatenate(pred)
    m = Model(inputs=[vals,xc,yc],outputs=predc)
    return m


# In[9]:


#tochange
model_name="gru_dense_InUni_OutUni"
path_save_optimize="../../../data/forecast/model/gru_dense_InUni_OutUni/optimize/"+model_name+"/"
path_save_fit="../../../data/forecast/model/gru_dense_InUni_OutUni/fit/"+model_name+"/"
path_save_prediction="../../../data/forecast/model/gru_dense_InUni_OutUni/prediction/"+model_name+"/"

#tochange
time_series = ['11', '32', '34', '15', '44', '65', '31', '33', '35', '47', '13',
       '14', '1', '9', '5', '18', '36', '24', '68', '43', '8', '64', '10',
       '55', '3', '49', '51', '2', '19', '56', '7', '6', '4', '48', '66',
       '25', '23', '28', '39', '54', '60', '27', '20', '46', '12', '21',
       '62', '52', '41', '50', '30', '16', '37', '40', '26', '67', '57',
       '61', '42', '45', '38', '29', '58', '63', '22', '59', '53', '17']


features_tminuslag_t0 = ["hms_int_15min", 'Day_id', 'Mois_id','vac_noel_quebec', 'day_off_quebec', '24DEC', '31DEC',
                    'renov_beaubien', 'vac_udem1', 'vac_udem2']


ContF=Cont[["Datetime"]+features_tminuslag_t0]
X=Vals.set_index('Datetime')[time_series].join(ContF.set_index('Datetime'))
                                
# optim parameters model                     

list_drop = [0.001]
list_neurones = [100,200,300]
list_forget = [0.8]
list_epoch = [15,20,25,30,35,40]

batch_size=128
lag_window = 100

norm=True
scaler=None
n_split = 5
val_size=0.2




# creation of my model that contains information about the model
my_model={}
my_model["path_values"] = path_vals
my_model["path_context"]= path_cont
my_model["path_info"]= path_info
my_model["model_name"]= model_name
my_model["path_save_optimize"]= path_save_optimize
my_model["path_save_fit"]= path_save_fit
my_model["path_save_prediction"]= path_save_prediction
my_model["time_series"] = time_series
my_model["features_tminuslag_t0"] = features_tminuslag_t0
my_model["list_drop"] = list_drop
my_model["list_neurones"] = list_neurones
my_model["list_forget"] = list_forget
my_model["list_epoch"] = list_epoch
my_model["lag_window"] = lag_window
my_model["batch_size"] = batch_size
my_model["val_size"] = val_size
my_model['norm'] = norm
my_model["scaler"] = scaler
my_model["n_split"] = n_split
my_model["function_model"] = buildGRUUNI

enc = OneHotEncoder()
enc = enc.fit(X[features_tminuslag_t0].values)
my_model["encoder"] = enc


my_model["best_drop"]  = 0.001
my_model["best_neurones"]  = 300
my_model["best_forget"] = 0.8
my_model["best_epoch"] = 30
PickleRick(path=my_model["path_save_optimize"],name="model_infos.pkl",obj=my_model,force=True)






# In[15]:


with open(path_save_optimize+"model_infos.pkl", 'rb') as f:
    my_model = pickle.load(f)
    
start_date_train_list=["2014-01-01","2014-01-01","2014-01-01","2014-01-01"]
end_date_train_list=["2015-01-01","2015-04-01","2015-07-01","2015-10-01"]

start_date_train_list=["2015-01-01",]
end_date_train_list=["2017-01-01",]


for i,j in zip(start_date_train_list, end_date_train_list):
    
    DatavPred,DatacPred=None,None
    XvPred,YvPred,XcPred,YcPred=None,None,None,None,
    DatavTrain,DatacTrain=None,None
    XvTrain,YvTrain,Xco,Yco=None,None,None,None,
    
    DatavTrain=X[my_model['time_series']][i:j].values
    DatacTrain=X[my_model['features_tminuslag_t0']][i:j].values
    
    # normalisation
    norm=my_model['norm']
    scaler = my_model["scaler"]
    if norm:
        #scaler = MinMaxScaler(feature_range=[0,1])
        #DatavTrainS = scaler.fit_transform(DatavTrain.astype(float))
        mean_train =  DatavTrain.mean(axis=0)
        my_model["mean_train"]= mean_train
        DatavTrainS = DatavTrain/mean_train
        #scaler = StandardScaler()
       # DatavTrainS = scaler.fit_transform(DatavTrain.astype(float))
    else:
        DatavTrainS = DatavTrain
        
    # one hot encoder contextual values
    enc = my_model["encoder"]
    DatacTrainOh = np.array(enc.transform(DatacTrain).todense())



    XvTrain,YvTrain=create_xy_dense(DatavTrainS,look_back=my_model["lag_window"])
    XcTrain,YcTrain=create_xy_dense(DatacTrainOh,look_back=my_model["lag_window"])
    


    ml = buildGRUUNI(my_model["best_drop"] ,int(my_model["best_neurones"]), my_model["lag_window"],
                        len(my_model["time_series"]),YcTrain.shape[1])
    ml.compile(optimizer='adam',loss='mae',metrics=['mse','mae'],sample_weight_mode=None)
    w = np.linspace(1,my_model["best_forget"],XvTrain.shape[0])
    #w = np.reshape(np.repeat(np.flip(w,0),my_model['lag_window']),(w.shape[0],my_model['lag_window']))
    h = ml.fit([XvTrain,XcTrain,YcTrain],YvTrain,epochs=my_model['best_epoch'],sample_weight=w,batch_size=my_model["batch_size"])
    
    def reload_model(weights, my_model):
        m = my_model['function_model'](my_model["best_drop"] ,int(my_model["best_neurones"]), my_model["lag_window"],
                        len(my_model["time_series"]),my_model['YcTrain_shape1'])
        m.compile(optimizer='adam',loss='mae',metrics=['mse','mae'],sample_weight_mode=None)
        m.set_weights(weights)
        return m
    
    my_model['YcTrain_shape1'] = YcTrain.shape[1]
    my_model['reload_model'] = reload_model
    PickleRick(path=my_model["path_save_fit"]+i+"_"+j+"/",name="model_infos.pkl",obj=my_model,force=True)
    
    if not os.path.exists(my_model["path_save_fit"]+i+"_"+j+"/"):
        os.makedirs(my_model["path_save_fit"]+i+"_"+j+"/")
    weights=ml.get_weights()
    output_filename = my_model["path_save_fit"]+i+"_"+j+"/"+ 'weights.npy'
    np.save(output_filename, weights)
        
    #PREDICTION    
    start_date_pred = "2015-01-01"
    end_date_pred = "2018-01-01"
    
    XvTrain,YvTrain,Xco,Yco=None,None,None,None,
    DatavTrain,DatacTrain=None,None
    DatavPred=X[my_model['time_series']][start_date_pred:end_date_pred].values
    DatacPred=X[my_model['features_tminuslag_t0']][start_date_pred:end_date_pred].values
    DatePred=X[my_model['time_series']][start_date_pred:end_date_pred].index.values
    
    norm = my_model['norm']
    scaler = None
    if norm:
        DatavPredS = DatavPred/my_model['mean_train']
    else: 
        DatavPredS = DatavPred

    # one hot encoder contextual values
    enc = my_model["encoder"]
    DatacPredOh = np.array(enc.transform(DatacPred).todense())


    XvPred,YvPred=create_xy_dense(DatavPredS,look_back=my_model["lag_window"])
    XcPred,YcPred=create_xy_dense(DatacPredOh,look_back=my_model["lag_window"])
    XDate,YDate=create_xy_dense(DatePred,look_back=my_model["lag_window"])
    YDate=YDate.ravel()
    
    t1=time.time()
    prediction = ml.predict([XvPred,XcPred,YcPred],verbose=1)
    print("Time prediction: ",time.time()-t1)
    #prediction = np.reshape(prediction,(prediction.shape[0]*prediction.shape[1],prediction.shape[2]))
    prediction = my_model["mean_train"]*prediction
    df_pred = pd.DataFrame()
    df_pred["Datetime"]=YDate
    for ix,id_ in enumerate(my_model["time_series"]):
        df_pred[id_]=prediction[:,ix]
    try:
        df_pred.to_csv(path_save_prediction+i+"_"+j+".csv",index=False,float_format='%.3f')
    except:
        os.makedirs(path_save_prediction)
        df_pred.to_csv(path_save_prediction+i+"_"+j+".csv",index=False,float_format='%.3f')





