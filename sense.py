import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, precision_score
from sklearn.svm import SVC
import seaborn
import matplotlib.pyplot as plt
import joblib

#Reads Dataset
data = pd.read_csv('ISL_dataset.csv')
data.columns = [i for i in  range(data.shape[1])]
data = data.rename(columns={42: 'Output'})
data
print("Uncleaned dataset size =", data.shape)

#Removing Null Values
null_values = data[data.iloc[:, 0] == 0]
print("Null Values =", len(null_values.index))

#deletes null values from dataset
data.drop(null_values.index, inplace=True)
print("Cleaned dataset size =", data.shape)

#Preparing data
X = data.iloc[:, :-1]
Y = data.iloc[:, :-1]
print("Landmarks size =", X.shape)
print("Labels shape =", Y.shape)

#Seperating data
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

#Creating Model
model = SVC(C=50, gamma=0.1, kernel='rbf')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Training Score =", model.score(y_test, y_pred))
#Saving Model for future callback in main program
filename ='ISL_model.pkl'
joblib.dump(model, filename)

#Viusualising
conm = confusion_matrix(y_test, y_pred)
#Accuracy Scores
f1 = f1_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, averaage='micro')
precision = precision_score(y_test, y_pred, average='micro')
f1, recall, precision

#Visualises a plot of accuracy
labels = sorted(list(set(data['Output'])))
labels = [x.upper() for x in labels]
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_title('Irish Sign Language Confusion Matrix')

#creates heatmap of accuracy
maping = seaborn.heatmap(conm, annot=True, cmap="rocket", linewidths=0.5, xticklabels=labels, yticklabels=labels, vmax=8, ax=ax )
maping
maping.figure.savefig("heatmap.png")
