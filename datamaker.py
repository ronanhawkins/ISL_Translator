import cv2
import mediapipe as mp
import pandas as pd  
import os
import numpy as np 

def image_processed(file_path):
    
    #reads image
    hand_img = cv2.imread(file_path)

    # Image processing
    #Converts BGR to RGB
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    #Flips the img in Y-axis
    img_flip = cv2.flip(img_rgb, 1)

    #Accessing Mediapipe
    mp_hands = mp.solutions.hands

    #Initializing Hands
    hands = mp_hands.Hands(static_image_mode=True,
    max_num_hands=1, min_detection_confidence=0.7)

    #Creates Results
    output = hands.process(img_flip)

    hands.close()

    try:
        #Defines Results
        data = output.multi_hand_landmarks[0]

        #Converts Results to String
        data = str(data)

        data = data.strip().split('\n')

        #Removes uneccesary data
        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_garbage = []

        for i in data:
            if i not in garbage:
                without_garbage.append(i)

        clean = []

        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])

        
        return([clean])

    except:
        return(np.zeros([1,63], dtype=int)[0])

def make_csv():

    #Defines directory and csv files
    mypath = '/Users/ronanhawkins/btyse/isl_dataset'
    file_name = open('ISL_dataset.csv', 'a')
    
    #for each folder
    for each_folder in os.listdir(mypath):
        if each_folder.startswith('.'):
            pass

        else:
            #creates line of landmark data for each picture file in each folder
            for each_number in os.listdir(mypath + '/' + each_folder):
                if each_number.startswith('.'):
                    pass
                
                else:
                    print(each_folder)
                    label = each_folder

                    file_loc = mypath + '/' + each_folder + '/' + each_number
                    print(each_number)
                    data = image_processed(file_loc)
                    
                    try:
                        for i in data:
                            file_name.write(str(i))
                            file_name.write(',')

                        file_name.write(label)
                        file_name.write('\n')
                    
                    except:
                        file_name.write('0')
                        file_name.write(',')

                        file_name.write('None')
                        file_name.write('\n')
       
    file_name.close()
    print('Data Creation Process Complete')

if __name__ == "__main__":
    make_csv()
