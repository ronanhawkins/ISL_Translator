# ISL_Translator
A translator for Irish Sign Language

Files needed are:

isl_dataset (folder) --> A, B, C, D, etc. (folders) --> picture.jpeg, etc.

datamaker.py (writes data to ISL_dataset.py by analysing isl_dataset)

ISL_dataset.py (created by datamaker.py, stores hand landmark data)

sense.py (creates model using ISL_dataset.py, saves model as ISL_model.pkl)

isl_model.pkl

transl.py (live program : displays video, makes prediction using ISL_model.pkl)


# datamaker.py
Searches the isl_dataset folder for folders

The folders will be contain pictures and they will be named after the sign they contain. eg. A

Writes a line of hand lanmark data for each picture into the ISL_dataset.csv

# sense.py
Creates and tests the AI model

Uses ISL_dataset.csv to make predictions

Creates a heatmap to show accuracy (heatmap.png)

# transl.py
Actively used program to add hand landmarks to video

Makes and displays live predictions using ISL_model.pkl

# python libraries used (download these using pip3)
opencv-python : video proccessing

mediapipe : analyses images for hand landmarks

pandas : data manipulation

numpy : data manipulations

sklearn : creates AI model

seaborn : creates heatmap

matplotlib : plots data

joblib : creates and loads .pkl files
