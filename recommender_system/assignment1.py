#!/usr/bin/env python
# coding: utf-8

# In[40]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy as np
import string
import random
import string
from sklearn import linear_model
import tensorflow as tf
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import itertools


# In[4]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[5]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[6]:


def readJSON(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        d = eval(l)
        u = d['userID']
        g = d['gameID']
        yield u,g,d


# In[7]:


allHours = []
for l in readJSON("train.json.gz"):
    allHours.append(l)


# In[8]:


hoursTrain = allHours[:155000]
hoursValid = allHours[155000:]


# In[9]:


gamePlayedPerUser = defaultdict(set)
itemSet = set()
for u,i,_ in allHours:
    gamePlayedPerUser[u].add(i)
    itemSet.add(i)


# In[10]:


new_train = []
# for u,i,_ in hoursTrain:
for u,i,_ in allHours:
    #     positive
    new_train.append((u,i,1))
    #     negative, (u,i,0) not in train games
    not_played_game = random.choice(list(itemSet.difference(gamePlayedPerUser[u])))
    new_train.append((u,not_played_game, 0))


# In[28]:


new_valid = []
for u,i,_ in hoursValid:
    #     positive
    new_valid.append((u,i,1))
    #     negative, (u,i,0) not in train games
    not_played_game = random.choice(list(itemSet.difference(gamePlayedPerUser[u])))
    new_valid.append((u,not_played_game, 0))


# In[11]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[29]:


true_labels = [d[2] for d in new_valid]


# In[72]:


#popularity
gameCount = defaultdict(int)
totalPlayed = 0

for user,game,_ in allHours:
    gameCount[game] += 1
    totalPlayed += 1

mostPopular = [(gameCount[x], x) for x in gameCount]
mostPopular.sort()
mostPopular.reverse()

# return2 = set()
# count2 = 0
# for ic, i in mostPopular:
#     count2 += ic
#     return2.add(i)
#     if count2 > totalPlayed/1.5: break

# predictions2 = []
# for u,g,_ in new_valid:
#     if g in return2:
#         predictions2.append(1)
#     else:
#         predictions2.append(0)


# In[74]:


gamePopRank


# In[73]:


gamePopRank = {x[1]:x[0] for x in mostPopular}


# In[29]:


# ///// tf: BPR


# In[13]:


userIDs = {}
itemIDs = {}
interactions = []

for d in allHours:
    u = d[0]
    i = d[1]
    t = d[2]['hours_transformed']
    if not u in userIDs: userIDs[u] = len(userIDs)
    if not i in itemIDs: itemIDs[i] = len(itemIDs)
    interactions.append((u,i,t))


# In[14]:


items = list(itemIDs.keys())


# In[15]:


class BPRbatch(tf.keras.Model):
    def __init__(self, K, lamb):
        super(BPRbatch, self).__init__()
        # Initialize variables
        self.betaI = tf.Variable(tf.random.normal([len(itemIDs)],stddev=0.001))
        self.gammaU = tf.Variable(tf.random.normal([len(userIDs),K],stddev=0.001))
        self.gammaI = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001))
        # Regularization coefficient
        self.lamb = lamb

    # Prediction for a single instance
    def predict(self, u, i):
        p = self.betaI[i] + tf.tensordot(self.gammaU[u], self.gammaI[i], 1)
        return p

    # Regularizer
    def reg(self):
        return self.lamb * (tf.nn.l2_loss(self.betaI) +\
                            tf.nn.l2_loss(self.gammaU) +\
                            tf.nn.l2_loss(self.gammaI))
    
    def score(self, sampleU, sampleI):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        gamma_u = tf.nn.embedding_lookup(self.gammaU, u)
        gamma_i = tf.nn.embedding_lookup(self.gammaI, i)
        x_ui = beta_i + tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1)
        return x_ui

    def call(self, sampleU, sampleI, sampleJ):
        x_ui = self.score(sampleU, sampleI)
        x_uj = self.score(sampleU, sampleJ)
        return -tf.reduce_mean(tf.math.log(tf.math.sigmoid(x_ui - x_uj)))


# In[16]:


optimizer = tf.keras.optimizers.Adam(0.1)


# In[17]:


modelBPR = BPRbatch(5, 0.00001)


# In[18]:


def trainingStepBPR(model, interactions):
    Nsamples = 50000
    with tf.GradientTape() as tape:
        sampleU, sampleI, sampleJ = [], [], []
        for _ in range(Nsamples):
            u,i,_ = random.choice(interactions) # positive sample
            j = random.choice(items) # negative sample
            while j in itemsPerUser[u]:
                j = random.choice(items)
            sampleU.append(userIDs[u])
            sampleI.append(itemIDs[i])
            sampleJ.append(itemIDs[j])

        loss = model(sampleU,sampleI,sampleJ)
        loss += model.reg()
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients((grad, var) for
                              (grad, var) in zip(gradients, model.trainable_variables)
                              if grad is not None)
    return loss.numpy()


# In[19]:


itemsPerUser = defaultdict(list)
usersPerItem = defaultdict(list)
for u,i,_ in new_train:
    if i not in usersPerItem.keys():
        usersPerItem[i] = [u]
    else:
        usersPerItem[i].append(u)
    if u not in itemsPerUser.keys():
        itemsPerUser[u] = [i]
    else:
        itemsPerUser[u].append(i)


# In[20]:


obj = 10
for i in range(300):
    obj = trainingStepBPR(modelBPR, interactions)
    
    if (i % 10 == 9): 
        print("iteration " + str(i+1) + ", objective = " + str(obj))
    


# In[22]:


played_pairs = []
for l in open("pairs_Played.csv"):
    if l.startswith("userID"):
        continue
    u,g = l.strip().split(',')
    played_pairs.append((u, g))


# In[32]:


bpr_ranks = defaultdict(list)

for i in played_pairs:
    rank = -1000
    u = i[0]
    g = i[1]
#     check if it is seen in train data
    if (u in userIDs) and (g in itemIDs):
        rank = modelBPR.predict(userIDs[u], itemIDs[g]).numpy()
    else:
#         check if it is in most popular games
        if g in dict(itertools.islice(gamePopRank.items(), 1650)):
            rank = 999
        # unseen data
        else:
            rank = -999
    bpr_ranks[u].append((g,rank))


# In[33]:


bpr_ranks


# In[36]:


#half and half of each label
predictions_log_reg = {}

for u in bpr_ranks:
    sorted_rank = sorted(bpr_ranks[u], key=lambda x: x[1], reverse=True)
    thres = int(len(sorted_rank) / 2)
    counter = 0
    for g,r in sorted_rank:
        if counter < thres:
            pred_label = 1
        else:
            pred_label = 0
        predictions_log_reg[(u, g)] =  pred_label
        counter += 1


# In[37]:


predictions_log_reg


# In[39]:


predictions = open("predictions_Played.csv", 'w')
for l in open("./pairs_Played.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    # Logic...
    _ = predictions.write(u + ',' + g + ',' + str(predictions_log_reg[(u,g)]) + '\n')

predictions.close()


# In[ ]:


# ##################


# In[ ]:


# ///////// time prediction //////////


# In[41]:


trainHours = [r[2]['hours_transformed'] for r in allHours]
globalAverage = sum(trainHours) * 1.0 / len(trainHours)


# In[42]:


hoursPerUser = defaultdict(list)
hoursPerItem = defaultdict(list)

for u,i,h in allHours:
    user = u
    item = i
    hrs = h['hours_transformed']
    
    hoursPerUser[user].append((item, hrs))
    hoursPerItem[item].append((user, hrs))


# In[43]:


sum_hoursPerUser =  defaultdict(float)
sum_hoursPerItem = defaultdict(float)
for u in hoursPerUser:
    sum_hoursPerUser[u] = sum([h for g,h in hoursPerUser[u]])
for g in hoursPerItem:
    sum_hoursPerItem[g] = sum([h for u,h in hoursPerItem[g]])

hours_sum = sum([hours['hours_transformed'] for _, _, hours in allHours])


# In[75]:


def iterate(lamb):
    ntrain = len(allHours) 
    temp_alpha = globalAverage
    betaU = defaultdict(float)
    betaI = defaultdict(float)
    alpha = globalAverage
    threshold = 0.51
    def alpha_beta(lamb):
        
        alpha = (hours_sum - sum([(betaU[u] + betaI[i]) for u,i,_ in allHours])) / ntrain 
        for u in hoursPerUser:
            new_Bu = (sum_hoursPerUser[u] - sum([(alpha + betaI[i]) for i,h in hoursPerUser[u]]))/ (lamb + len(hoursPerUser[u]))
            betaU[u] = new_Bu
        for g in hoursPerItem:
            new_Bi = (sum_hoursPerItem[g] - sum([(alpha + betaU[u]) for u,h in hoursPerItem[g]]))/ (lamb + len(hoursPerItem[g]))
            betaI[g] = new_Bi
        return alpha
    
    
    res = alpha_beta(lamb)
#     while abs(temp_alpha - res) <= threshold:
    for i in range(400):
        alpha_beta(lamb)
        alpha = alpha_beta(lamb)
        temp_alpha = alpha
        
    return betaU, betaI, alpha


# In[76]:


def model6_predict(alpha, betaU, betaI, u, i):
    return (alpha + betaU[u] + betaI[i])


# In[77]:


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)


# In[78]:


# yTrue = []
# yPred = []
betaU, betaI, alpha = iterate(5)

# for u, i, h in allHours:
#     yTrue.append(h['hours_transformed'])
#     pred = model6_predict(alpha, betaU, betaI, u, i)
#     yPred.append(pred)


# In[79]:


time_played = []
for l in open("pairs_Hours.csv"):
    if l.startswith("userID"):
        continue
    u,g = l.strip().split(',')
    time_played.append((u,g))


# In[80]:


predictions_time = {}
for d in time_played:
    u = d[0]
    g = d[1]
    pred_t = model6_predict(alpha, betaU, betaI, u, g)
    predictions_time[(u,g)] = pred_t


# In[70]:


predictions_time


# In[82]:


predictions = open("predictions_Hours.csv", 'w')
for l in open("./pairs_Hours.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    # Logic... 
    
    bu = betaU[u]
    bi = betaI[g]
    _ = predictions.write(u + ',' + g + ',' + str(predictions_time[(u,g)]) + '\n')

predictions.close()


# In[ ]:





# In[ ]:




