import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, precision_score
from sklearn.svm import SVC
import seaborn
import matplotlib.pyplot as plt
import joblib

#Reads Dataset
labels = ("A", "B","C","D","E","F","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","Y")
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
#X = data.iloc[:,0:63]
#Y = data.iloc[:, -1]
X = data.iloc[:, :-1]
print("Features shape =", X.shape)
Y = data.iloc[:, -1]
print("Labels shape =", Y.shape)
X.columns = X.columns.astype(str)
#print("Landmarks size =", X.shape)
#print("Labels shape =", Y.shape)

#Seperating data
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

#Creating Model
x_test = x_test.astype(str)
x_train.columns = x_train.columns.astype(str)
model = SVC(C=50, gamma=0.1, kernel='rbf')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Training Score =", model.score(x_test, y_pred))
#Saving Model for future callback in main program
filename ='ISL_model.pkl'
joblib.dump(model, filename)

#Viusualising
conm = confusion_matrix(y_test, y_pred)
#Accuracy Scores
f1 = f1_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
precision = precision_score(y_test, y_pred, average='micro')
f1, recall, precision

#Visualises a plot of accuracy
seaborn.set(font_scale=1)
fig, ax = plt.subplots(figsize=(24, 24))
ax.set_title('Irish Sign Language Detection Confusion Matrix', fontdict = { 'fontsize': 50})

#creates heatmap of accuracy
maping = seaborn.heatmap(conm, annot=True, cmap = "mako_r", linewidths=.2,vmax=1.0e+2, ax=ax)
maping.set_xticklabels(labels, fontsize = 26)
maping.set_yticklabels(labels, fontsize = 26)
plt.xlabel("Actual Sign" , size = 40 )
plt.ylabel("Predicted Sign" , size = 40 )
maping
maping.figure.savefig("heatmap.png")
print("Complete")
