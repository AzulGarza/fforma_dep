#!/usr/bin/env python
# coding: utf-8

# ## Testing if there's transfer learning 

# In[10]:


from fforma import *
import numpy as np
import pickle

#Disabling warnings
import warnings
warnings.filterwarnings("ignore")


# ### Train data

# In[2]:

print("Reading and processing data")

monthly = pd.read_csv('data/data_m4/Monthly-train.csv', nrows=4000).set_index('V1')


# In[3]:


list_series = [ts.dropna().to_numpy() for idx, ts in monthly.iterrows()]


# In[4]:


n_series = len(list_series)

print(f"Total series: {n_series}")
# Getting chunks of data of different size.

# In[5]:

#print("Making chunks of data")
#size_chunks = (n_series*np.arange(10, 50, 10)/100).round().astype(int).tolist()
#size_chunks.insert(0, 1000)


# In[6]:


#print(size_chunks)


# In[7]:

print("Generating samples of data") 
size_chunks =  [100, 1000, 2000, 3000, 4000]
chunks_data = [list_series[:n] for n in size_chunks]


# ### Training models 

# In[8]:

print("Declaring models")
models = [Naive(), SeasonalNaive(), RandomWalkDrift(), ETS()]#AutoArima()]


# In[9]:

print("Declaring frequency")
frequency = 12 # Monthly data


# In[ ]:


print("Training models")
for idx, list_ts in enumerate(chunks_data):
    n = size_chunks[idx]
    print(f"Chunk {idx+1} of size {n}")
    
    
    model = FForma().train(models, list_ts, frequency)
    
    with open(f'fforma_{n}', 'wb') as outfile:
        pickle.dump(model, outfile)
    print(f"Saved model {idx+1} of size {n}")


# In[27]:





# In[ ]:




