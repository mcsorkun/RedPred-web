# -*- coding: utf-8 -*-
"""
Created on Fri Sep 09 14:54:37 2021

@author: Cihan Yatbaz, Elham Nour Gassemi
"""


######################
# Import libraries
######################

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import warnings
warnings.filterwarnings("ignore")



from PIL import Image
import base64
import pandas as pd
import streamlit as st
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor

#For KERAS
import random
import numpy as np
# from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
import time

import numpy
from sklearn.model_selection import GridSearchCV

# import tensorflow 
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Dropout

#from keras.layers import Dense
#from keras.layers import Dropout
# Function to create model, required for KerasClassifier


# def create_model(optimizer='RMSprop', learn_rate=0.1, momentum=0.4, activation='sigmoid', dropout_rate=0.0):
    
#     keras_model = Sequential()
#     keras_model.add(Dense(128, input_dim=train_encoded.shape[1], activation=activation))
#     keras_model.add(Dropout(dropout_rate))
#     keras_model.add(Dense(32, activation=activation)) 
#     keras_model.add(Dropout(dropout_rate))
#     keras_model.add(Dense(8,activation=activation)) 
#     keras_model.add(Dropout(dropout_rate))
#     keras_model.add(Dense(1,activation='linear'))
#     keras_model.summary()
#     # Compile model
#     keras_model.compile(loss='mean_squared_error', optimizer=optimizer)

#     return keras_model


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



#111111111111
# ## generate dataset it is diffrent from origin one  
# import deepchem as dc
# from deepchem.models import GraphConvModel

# def generate(SMILES, verbose=False):

#     featurizer = dc.feat.ConvMolFeaturizer()
#     gcn = featurizer.featurize(SMILES)
#     properties = [random.randint(-1,1)/100  for i in range(0,len(SMILES))]
#     dataset = dc.data.NumpyDataset(X=gcn, y=np.array(properties))
    
#     return dataset
#111111111111

######################
# Page Title
######################



#st.beta_set_page_config(page_title="AqSolPred: Online Solubility Prediction Tool")


st.write("""# RedPred: Redox Energy Prediction Tool """)

image = Image.open('image1.jpg')
st.image(image, use_column_width=True)


######################
# Input molecules (Side Panel)
######################

st.sidebar.write('**Type SMILES below**')

## Read SMILES input
SMILES_input = "c1cc(O)cc(c12)cc(nn2)O\nOc1ccc(O)c(c12)cc(nn2)O"

SMILES = st.sidebar.text_area('then press ctrl+enter', SMILES_input, height=20)
SMILES = SMILES.split('\n')
SMILES = list(filter(None, SMILES))


st.sidebar.write("""---------**OR**---------""")
st.sidebar.write("""**Upload a csv file with a column named 'reactant_smiles'** (Max:1000)""")



uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # data
    SMILES=data["reactant_smiles"]  

    # About dataset
    st.write('Data Dimensions: ' + str(data.shape[0]) + ' rows and ' + str(data.shape[1]) + ' columns.')
    
    data_expander = st.expander("Explore the Dataset", expanded=False)
    with data_expander:
		
        st.dataframe(data[0:1000])






# st.header('Input SMILES')
# SMILES[1:] # Skips the dummy first item

# Use only top 1000
if len(SMILES)>1000:
    SMILES=SMILES[0:1000]
	
## Calculate molecular descriptors
ecfc_encoder = get_ecfc(SMILES)

#Import pretrained models

#222222222222222
#---------------------------------------------------------------------------------
### generate dataset from SMILES and function generate
# generated_dataset = generate(SMILES)

# ### transformer for gcn 
# filename = 'final_models/transformers.pkl'
# infile = open(filename,'rb')
# transformers = pickle.load(infile)
# infile.close()


# ## model for gcn 
# model_dir = 'final_models/tf_chp_initial'
# gcne_model = dc.models.GraphConvModel(n_tasks=1, batch_size=100, mode='regression', dropout=0.25,model_dir= model_dir,random_seed=0)
# gcne_model.restore('final_models/tf_chp_initial/ckpt-94/ckpt-197')
# #print(gcne_model)


# ## predict energy from gcn model 
# pred_gcne = gcne_model.predict(generated_dataset, transformers)
# #222222222222222

#---------------------------------------------------------------------------------
##keras model load
# from keras.models import model_from_json

# keras_final_model = model_from_json(open('./final_models/keras_final_model_architecture.json').read())
# keras_final_model.load_weights('./final_models/keras_final_model_weights.h5')

#keras_final_model = pickle.load(open(r'./final_models/keras_final_model.txt', "rb"))
rf_final_model = pickle.load(open(r'./final_models/rf_final_model.txt', "rb"))
#xgbm_final_model = pickle.load(open(r'.\final_models\xgbm_final_model.txt', "rb"))



#predict test data (Keras,RF, GCN)
# pred_keras = keras_final_model.predict(ecfc_encoder)   
pred_rf  = rf_final_model.predict(ecfc_encoder)

##reshape (n,)    ----> (n,1)

pred_rf_r = pred_rf.reshape((len(pred_rf),1))
#pred_xgb = xgbm_final_model.predict(ecfc_encoder)   


#calculate consensus
# pred_consensus = (pred_keras + pred_gcne + pred_rf)/3
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


# weighted_pred_0_1_3=( np.power(2/(test1_mae[0]+test2_mae[0]),3) * pred_gcne + 
#             np.power(2/(test1_mae[1]+test2_mae[1]),3) * pred_keras + 
#             np.power(2/(test1_mae[2]+test2_mae[2]),3) * pred_rf_r ) / (
#             np.power(2/(test1_mae[0]+test2_mae[0]),3) + np.power(2/(test1_mae[1]+test2_mae[1]),3) + np.power(2/(test1_mae[2]+test2_mae[2]),3)) 



#--------

#### ????  array shape not correct and no difference with pred_consensus

# pred_weighted = (pred_gcne + pred_keras + pred_rf_r)/3
pred_weighted = (pred_gcne + pred_rf_r)/2
pred_weighted = pred_rf_r





#%%
# results=np.column_stack([pred_mlp,pred_xgb,pred_rf,pred_consensus])

df_results = pd.DataFrame(SMILES, columns=['reactant_smiles'])
# df_results["reaction_energy"]= weighted_pred_0_1_3
df_results["reaction_energy"]= pred_weighted
df_results=df_results.round(6)

# df_results.to_csv("results/predicted-"+test_data_name+".csv",index=False)


# Results DF

st.header('Predicted Reaction Energy (Hartree) ')
df_results # Skips the dummy first item






# download=st.button('Download Results File')
# if download:
csv = df_results.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()  # some strings
linko= f'<a href="data:file/csv;base64,{b64}" download="redpred_predictions.csv">Download csv file</a>'
st.markdown(linko, unsafe_allow_html=True)
 

# st.header('Individual Predictions')

# df_models = pd.DataFrame(SMILES, columns=['reactant_smiles'])

# df_models["GCN"]=pred_gcne
# # df_models["DNN"]=pred_keras
# df_models["RF"]=pred_rf
# df_models=df_models.round(5)

# df_models # Skips the dummy first item





#%%
# About PART

about_part = st.expander("About RedPred", expanded=False)
with about_part:
    st.write('''
	     #### About
	     RedPred is an reaction energy prediction model for redox flow battery molecules that consists ensemble of 3 ML algorithms (Graph Conv Neural Nets, Deep Neural Nets, and Random Forest). 
	     
	     You can upload or type your SMILES used as a reactant in the redox reaction to get the reaction energy (Hartree).
	     
	     RedPred is trained on RedDB [1] publicly available redox flow battery candidate molecules dataset.
	     
	     The performance of the RedPred is 0.0036 Hartree MAE on the test set.
	     
	     If you are using the predictions from RedPred on your work, please cite these papers: [1, 2]

	     [1] Sorkun, Elif, et al. (2021). [RedDB, a computational database of electroactive molecules for aqueous redox flow batteries.](https://www.nature.com/articles/s41597-022-01832-2)
	     
	     [2] Sorkun, M. C., Ghassemi, E. N., Yatbaz, C., Koelman, J. V. A., & Er, S. (2023). [RedPred, a machine learning model for the prediction of redox reaction energies of the aqueous organic electrolytes.](https://chemrxiv.org/engage/chemrxiv/article-details/6467754bfb40f6b3eed6e874)

	     #### Developers

	     - [Murat Cihan Sorkun](https://www.linkedin.com/in/murat-cihan-sorkun/)
	     - [Cihan Yatbaz] (https://www.linkedin.com/in/cihanyatbaz/)
	     - [Elham Nour Ghassemi] (https://www.linkedin.com/in/elhamnourghassemi/)

	''')


contacts = st.expander("Contact", expanded=False)
with contacts:
    st.write('''
             #### Report an Issue 
             
             You are welcome to report a bug or contribuite to the web 
             application by filing an issue on [Github] (https://github.com/mcsorkun/RedPred-web).
             
             #### Contact
             
             For any question you can contact us through email:
                 
             - [Murat Cihan Sorkun] (mailto:mcsorkun@gmail.com)
             ''')

  
