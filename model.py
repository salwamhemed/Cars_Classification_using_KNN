import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn 
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle 
df = pd.read_csv(r"C:\Users\salwa\OneDrive\Desktop\machine learning projects\Safety Car\Car Data Set\car.data")
print(df.head())

#Transforming the data into numerical data 

label_encoder = preprocessing.LabelEncoder()
df['buying'] = label_encoder.fit_transform(list(df['buying']))
df['maint'] = label_encoder.fit_transform(list(df['maint']))
df['door'] = label_encoder.fit_transform(list(df["door"]))
df["persons"] = label_encoder.fit_transform(list(df["persons"]))
df['lug_boot'] = label_encoder.fit_transform(list(df['lug_boot']))
df['safety'] = label_encoder.fit_transform(list(df['safety']))
df['class'] = label_encoder.fit_transform(list(df['class']))
predict = "class"


 
print(df.head())
X = df[['buying', 'maint', 'lug_boot', 'safety']]  # Features
y = df['class']  # Target variable


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
k_values = range(1, 31)
test_acc = []
best_score = 0
best_model = None
best_k =0
# Trying different values of k
for k in k_values:
    
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)  

    predictions = model.predict(X_test)
    
    acc = model.score(X_test, y_test)
    print("Accuracy: " + str(acc))
    
    test_acc.append(acc)
    if acc > best_score :
        best_k = k
        best_score= acc
        best_model= model

#Saving the best Model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print(f"the accuracy of the best model is {best_score}")
print(f"the best value of k is {best_k}")

plt.plot(k_values, test_acc, label='Test Accuracy')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('Model Evaluation with KNN')
plt.legend()
plt.show()



