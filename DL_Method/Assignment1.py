#!/usr/bin/env python
# coding: utf-8

# In[79]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize,curve_fit


# In[80]:


df = pd.read_csv("../data/04_cricket_1999to2011.csv")
df.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


df.info()


# In[4]:


df = df[df['Error.In.Data']==0]


# In[5]:


df['Over.Remaining'] = df['Total.Overs']-df['Over']
df = df[['Innings','Innings.Total.Runs', 'Runs.Remaining','Wickets.in.Hand','Over.Remaining']]
df.head()


# In[6]:


df1= df[df['Innings']==1]
df1.head()


# In[7]:


in_params  =[]
def getMeanRunByWicketRemaining():
    '''in_params = array of(z1,z2,...z10,L)'''
    for i in range(1,11):
        in_params.append(np.mean(df[df['Wickets.in.Hand']==i]['Runs.Remaining']))
    in_params.append(sum(in_params)*0.005)
getMeanRunByWicketRemaining()
print(in_params)


# In[36]:


def run_production_function(over_rem,Z,L):
    return Z*(1- np.exp((-L*over_rem)/Z))


# In[9]:


def loss_function(params,runs_rem, wickets_rem, overs_rem,n):
    squared_errors =[]
    for i in range(n):
        pred = run_production_function(overs_rem[i],params[wickets_rem[i]-1],params[10])
        squared_errors.append((pred- runs_rem[i])**2)
         
    return np.sum(squared_errors)


# In[10]:


df = df[df['Wickets.in.Hand']!=0]
runs_rem = df['Runs.Remaining'].tolist()
wickets_rem =df['Wickets.in.Hand'].tolist()
overs_rem=df['Over.Remaining'].tolist()
n = len(overs_rem)



# In[11]:


model = minimize(loss_function,in_params,args=(runs_rem, wickets_rem, overs_rem,n),method='L-BFGS-B')
print(model)


# In[39]:


def plot(model):
    fig  = plt.figure(figsize= (15,10))
    for i in range(10):
        u= range(51)
        n = len(u)
        y=[]
        Z = model.x[i]
        L = model.x[10]
        for j in range(51):
            y.append(run_production_function(u[j],Z,L))
        plt.xlabel("Overs Remaining")
        plt.ylabel("Runs to Score")
        plt.plot(u,y)


# In[40]:


plot(model)


# In[ ]:





# In[74]:


def plot_against_resources(model):
    fig  = plt.figure(figsize= (15,10))
    for i in range(1,11):  #wicket_rem
        u= range(0,51)   #over_rem
        n = len(u)  
        y=[]
        Z = model.x[i-1]
        L = model.x[10]
        #(resource_rem) = run_production_fnc / highest resource
        for j in range(51):
            y.append(100*run_production_function(u[j],Z,L)/highest_score)
        
        plt.plot(u,y)
    plt.ylabel("Resources Remaining")
    plt.xlabel("Overs To go")
    plt.savefig("Resources_vs_over_rem.jpg")


# In[75]:


highest_score = run_production_function(50,model.x[9],model.x[10])


# In[ ]:





# In[76]:


plot_against_resources(model)


# In[77]:


print("Z1 ,Z2,Z3....Z10,L parameter values : \n",model.x)
print("Mean Squared Error: ", model.fun)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




