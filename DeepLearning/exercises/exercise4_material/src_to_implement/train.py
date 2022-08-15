import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
from torch import nn
import pandas as pd
import torchvision as tv

from sklearn.model_selection import train_test_split

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
data = pd.read_csv('data.csv', sep=';')
train, val = train_test_split(data, test_size=0.25, random_state=123)
print(train)
train['powerlabel'] = train.apply(lambda x : 2*x['crack']+1*x['inactive'],axis=1)
print(train)
train['powerlabel'].hist(bins=np.unique(train['powerlabel']))

# print(val[['crack', 'inactive']].value_counts())
# print(train[['crack', 'inactive']].value_counts())
# print(data[['crack', 'inactive']].value_counts())

# # set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# params = {'batch_size':64,
#           'shuffle': True}

# training_set = ChallengeDataset(train, mode='train')
# training_generator = t.utils.data.DataLoader(training_set, **params)

# val_set = ChallengeDataset(val, mode='val')
# val_generator = t.utils.data.DataLoader(val_set, **params)
# # create an instance of our ResNet model
# my_model = tv.models.efficientnet_b0(pretrained=True)
# num_ftrc = my_model.fc.in_features
# my_model.fc = nn.Sequential(
#     nn.Linear(num_ftrc, 2),
#     nn.Sigmoid()
# )
# # print(my_model)

# # my_model = model.ResNet()

# # set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# # loss = t.nn.BCEWithLogitsLoss()
# loss = t.nn.BCELoss()

# # set up the optimizer (see t.optim)
# optimizer = t.optim.Adam(my_model.parameters(), lr=0.001)

# # create an object of type Trainer and set its early stopping criterion
# model = Trainer(model=my_model, crit=loss, optim=optimizer, train_dl=training_generator,
#                 val_test_dl=val_generator, cuda=True, early_stopping_patience=20)

# # go, go, go... call fit on trainer
# res = model.fit(120)

# # plot the results
# plt.plot(np.arange(len(res[0])), res[0], label='train loss')

# plt.title(f'min_train: {round(min(res[0]), 3)}, min_val: {round(min(res[1]), 3)}, min_val_epoch: {res[1].index(min(res[1]))}')
# plt.legend()
# plt.savefig('losses_train.png')
# plt.close()

# plt.plot(np.arange(len(res[1])), res[1], label='val loss')

# plt.title(f'min_train: {round(min(res[0]), 3)}, min_val: {round(min(res[1]), 3)}, min_val_epoch: {res[1].index(min(res[1]))}')
# plt.legend()
# plt.savefig('losses_val.png')
