# %% [markdown]
import tensorflow as tf
import numpy as np
"""
Homework:

The folder '~//data//homework' contains a folder 'Data', containing hand-digits of letters a-z stored in .txt.

Try to establish a network to classify the digits.

`dataLoader.py` offers APIs for loading data.
"""
# %%
import dataLoader as dl
features,labels=dl.readData('../data/homework/Data')

class_names = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# %%
import matplotlib.pyplot as plt
plt.plot(features[5,0:30],features[5,30:])
plt.suptitle="Real: "+labels[5]
plt.show()
# %%
# feature engineering (if necessary)
# %%
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=1)
# train-test split
# %%
# build the network
model = tf.keras.Sequential([
    tf.keras.Input(shape=(60)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(26, activation=tf.nn.softmax )
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
labels=[]
for letter in labels_train:
    labels.append(dl.letter2Number(letter))
labels_train2=np.array(labels)

labels=[]
for letter in labels_test:
    labels.append(dl.letter2Number(letter))
labels_test2=np.array(labels)
# %%
model.fit(features_train, labels_train2,batch_size=32, epochs=300)
# training
# %%
test_loss, test_acc = model.evaluate(features_test,  labels_test2)
print('Test accuracy:', test_acc)

labels_hat=model.predict(features_test)

labels_hat=np.argmax(labels_hat,axis=1)
acc=sum((labels_hat==labels_test2).tolist())/labels_hat.size
print('Test accuracy:', acc)
labels_hat_letter=[]
for num in labels_hat:
    labels_hat_letter.append(dl.number2Letter(num))

test_sample=55
plt.plot(features_test[test_sample,0:30],features_test[test_sample,30:])
result="Real"+labels_test[test_sample]+"Predicted:"+labels_hat_letter[test_sample]
print(result)
plt.suptitle=result
plt.show()
# %%
from sklearn.metrics import confusion_matrix
con_mat=confusion_matrix(labels_test,labels_hat_letter,labels=dl.getAlphabet(),normalize="true")
plt.matshow(con_mat)
plt.xticks(np.arange(26),dl.getAlphabet())
plt.yticks(np.arange(26),dl.getAlphabet())
plt.show()
# predict and evaluate



