# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 14:54:37 2020

@author: Cihan Yatbaz
"""


######################
# Import libraries
######################


from PIL import Image
#from sklearn.neural_network import MLPRegressor
#import predefined_models
import base64


import pandas as pd
import streamlit as st
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
#import xgboost as xgb

#For KERAS
import random
import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
import time

import numpy
from sklearn.model_selection import GridSearchCV


#import keras
import tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

#from keras.layers import Dense
#from keras.layers import Dropout
# Function to create model, required for KerasClassifier


def create_model(optimizer='RMSprop', learn_rate=0.1, momentum=0.4, activation='sigmoid', dropout_rate=0.0):
    
    keras_model = Sequential()
    keras_model.add(Dense(128, input_dim=train_encoded.shape[1], activation=activation))
    keras_model.add(Dropout(dropout_rate))
    keras_model.add(Dense(32, activation=activation)) 
    keras_model.add(Dropout(dropout_rate))
    keras_model.add(Dense(8,activation=activation)) 
    keras_model.add(Dropout(dropout_rate))
    keras_model.add(Dense(1,activation='linear'))
    keras_model.summary()
    # Compile model
    keras_model.compile(loss='mean_squared_error', optimizer=optimizer)

    return keras_model


######################
# Custom function
######################
## Calculate molecular descriptors



def get_ecfc(smiles_list, radius=2, nBits=2048, useCounts=True):
    """
    Calculates the ECFP fingerprint for given SMILES list
    
    :param smiles_list: List of SMILES
    :type smiles_list: list
    :param radius: The ECPF fingerprints radius.
    :type radius: int
    :param nBits: The number of bits of the fingerprint vector.
    :type nBits: int
    :param useCounts: Use count vector or bit vector.
    :type useCounts: bool
    :returns: The calculated ECPF fingerprints for the given SMILES
    :rtype: Dataframe
    """     
    
    ecfp_fingerprints=[]
    erroneous_smiles=[]
    for smiles in smiles_list:
        mol=Chem.MolFromSmiles(smiles)
        if mol is None:
            ecfp_fingerprints.append([None]*nBits)
            erroneous_smiles.append(smiles)
        else:
            mol=Chem.AddHs(mol)
            if useCounts:
                ecfp_fingerprints.append(list(AllChem.GetHashedMorganFingerprint(mol, radius, nBits)))  
            else:    
                ecfp_fingerprints.append(list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits).ToBitString()))  
    
    # Create dataframe of fingerprints
    df_ecfp_fingerprints = pd.DataFrame(data = ecfp_fingerprints, index = smiles_list)
    # Remove erroneous data
    if len(erroneous_smiles)>0:
        print("The following erroneous SMILES have been found in the data:\n{}.\nThe erroneous SMILES will be removed from the data.".format('\n'.join(map(str, erroneous_smiles))))           
        df_ecfp_fingerprints = df_ecfp_fingerprints.dropna(how='any')    
    
    return df_ecfp_fingerprints




## generate dataset it is diffrent from origin one  
import deepchem as dc
from deepchem.models import GraphConvModel

def generate(SMILES, verbose=False):

    featurizer = dc.feat.ConvMolFeaturizer()
    gcn = featurizer.featurize(SMILES)
    properties = [random.randint(-1,1)/100  for i in range(0,len(SMILES))]
    dataset = dc.data.NumpyDataset(X=gcn, y=np.array(properties))
    
    return dataset


######################
# Page Title
######################



#st.beta_set_page_config(page_title="AqSolPred: Online Solubility Prediction Tool")


st.write("""# RedDB: RedDB ML Project""")

image = Image.open('solubility-factors.png')
st.image(image, use_column_width=False)


######################
# Input molecules (Side Panel)
######################

st.sidebar.write('**Type SMILES below**')

## Read SMILES input
SMILES_input = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C\nCC(=O)OC1=CC=CC=C1C(=O)O"

SMILES = st.sidebar.text_area('then press ctrl+enter', SMILES_input, height=20)
SMILES = SMILES.split('\n')
SMILES = list(filter(None, SMILES))


st.sidebar.write("""---------**OR**---------""")
st.sidebar.write("""**Upload a file with a column named 'reactant_smiles'** (Max:2000)""")

   
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # data
    SMILES=data["reactant_smiles"]  
    
    
    data_expander = st.beta_expander("Explore the Dataset", expanded=False)
    with data_expander:
        st.dataframe(data)


# st.header('Input SMILES')
# SMILES[1:] # Skips the dummy first item

# Use only top 300
if len(SMILES)>2000:
    SMILES=SMILES[0:2000]
	
## Calculate molecular descriptors
ecfc_encoder = get_ecfc(SMILES)

#Import pretrained models

#---------------------------------------------------------------------------------
### generate dataset from SMILES and function generate
generated_dataset = generate(SMILES)

### transformer for gcn 
filename = 'transformers.pkl'
infile = open(filename,'rb')
transformers = pickle.load(infile)
infile.close()


## model for gcn 
model_dir = 'tf_chp_initial'
gcne_model = dc.models.GraphConvModel(n_tasks=1, batch_size=100, mode='regression', dropout=0.25,model_dir= model_dir,random_seed=0)
gcne_model.restore('tf_chp_initial/ckpt-94/ckpt-197')
#print(gcne_model)


## predict energy from gcn model 
pred_gcne = gcne_model.predict(generated_dataset, transformers)


#---------------------------------------------------------------------------------
##keras model load
from keras.models import model_from_json

keras_final_model = model_from_json(open('./final_models/keras_final_model_architecture.json').read())
keras_final_model.load_weights('./final_models/keras_final_model_weights.h5')

#keras_final_model = pickle.load(open(r'./final_models/keras_final_model.txt', "rb"))
rf_final_model = pickle.load(open(r'./final_models/rf_final_model.txt', "rb"))
#xgbm_final_model = pickle.load(open(r'.\final_models\xgbm_final_model.txt', "rb"))



#predict test data (Keras,RF, GCN)
pred_keras = keras_final_model.predict(ecfc_encoder)   
pred_rf  = rf_final_model.predict(ecfc_encoder)

##reshape (n,)    ----> (n,1)

pred_rf_r = pred_rf.reshape((len(pred_rf),1))
#pred_xgb = xgbm_final_model.predict(ecfc_encoder)   


#calculate consensus
pred_consensus = (pred_keras + pred_gcne + pred_rf)/3
# predefined_models.get_errors(test_logS_list,pred_enseble)

#%% Weighted 

#------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------






from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

## Test 1 Experiments

test1_mae = []

test1_mae.append(0.00705) # 0 - GCN
test1_mae.append(0.00416) # 1 - Keras
test1_mae.append(0.0035) # 3 - RF



## Test 2 Experiments

test2_mae = []

test2_mae.append(0.00589) # 0 - GCN
test2_mae.append(0.00483) # 1 - Keras
test2_mae.append(0.00799) # 3 - RF


####????????

### if it is weightred prediction  check array shape?


weighted_pred_0_1_3=( np.power(2/(test1_mae[0]+test2_mae[0]),3) * pred_gcne + 
            np.power(2/(test1_mae[1]+test2_mae[1]),3) * pred_keras + 
            np.power(2/(test1_mae[2]+test2_mae[2]),3) * pred_rf_r ) / (
            np.power(2/(test1_mae[0]+test2_mae[0]),3) + np.power(2/(test1_mae[1]+test2_mae[1]),3) + np.power(2/(test1_mae[2]+test2_mae[2]),3)) 



#--------

#### ????  array shape not correct and no difference with pred_consensus

pred_weighted = (pred_gcne + pred_keras + pred_rf_r)/3







#%%
# results=np.column_stack([pred_mlp,pred_xgb,pred_rf,pred_consensus])

df_results = pd.DataFrame(SMILES, columns=['reactant_smiles'])
df_results["reaction_energy"]= weighted_pred_0_1_3
#df_results["reaction_energy"]= pred_weighted
df_results=df_results.round(6)

# df_results.to_csv("results/predicted-"+test_data_name+".csv",index=False)

st.header('Predicted reaction_energy values')
df_results # Skips the dummy first item

# download=st.button('Download Results File')
# if download:
csv = df_results.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()  # some strings
linko= f'<a href="data:file/csv;base64,{b64}" download="aqsolpred_predictions.csv">Download csv file</a>'
st.markdown(linko, unsafe_allow_html=True)
 

st.header('Model Performances')

df_models = pd.DataFrame(SMILES, columns=['reactant_smiles'])

df_models["GCN"]=pred_gcne
df_models["Keras"]=pred_keras
df_models["RF"]=pred_rf
df_models=df_models.round(5)

df_models # Skips the dummy first item





#%%
# About PART

about_part = st.beta_expander("About RedDB ML Project", expanded=False)
with about_part:
    st.write("""
                 AqSolPred is an highly accurate solubility prediction model that consists consensus of 3 ML algorithms (Neural Nets, Random Forest, and XGBoost). AqSolPred is developed using a quality-oriented data selection method described in [1] and trained on AqSolDB [2] largest publicly available aqueous solubility dataset.

                AqSolPred showed a top-performance (0.348 LogS Mean Absolute Error) on Huuskonen benchmark dataset [3].
                
                **version:** 1.0s (lite version of v1.0 described in the paper with reduced RFs(n_estimators=200,max_depth=10) but the same performance)
                
                If you are using the predictions from AqSolPred on your work, please cite these papers: [1, 2]
                
                [1] Sorkun, M. C., Koelman, J.M.V.A. & Er, S. (2021). [Pushing the limits of solubility prediction via quality-oriented data selection](https://www.cell.com/iscience/fulltext/S2589-0042(20)31158-5), iScience, 24(1), 101961.
                
                [2] Sorkun, M. C., Khetan, A., & Er, S. (2019).  [AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds](https://www.nature.com/articles/s41597-019-0151-1). Scientific data, 6(1), 1-8.
                
                [3] Huuskonen, J. Estimation of aqueous solubility for a diverse set of organic compounds based on molecular topology. Journal of Chemical Informationand Computer Sciences 40, 773–777 (2000).
                
                Special thanks: This web app is developed based on the tutorials and the template of [DataProfessor's repository](https://github.com/dataprofessor/code/tree/master/streamlit/part7). 
                
                                                                                                         
                **Contact:** [Murat Cihan Sorkun](https://www.linkedin.com/in/murat-cihan-sorkun/)
                
                """)


contacts = st.beta_expander("Contact", expanded=False)
with contacts:
    st.write('''
             #### Report an Issue 
             
             You are welcome to report a bug or contribuite to the web 
             application by filing an issue on [Github] x.
             
             #### Contact
             
             For any question you can contact us through email:
                 
             - [Murat Cihan Sorkun] (mailto:mcsorkun@gmail.com)
             - [-] 
             ''')

