import os
import cv2
import sys
import numpy as np
import matplotlib.image as img
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
np.set_printoptions(threshold=sys.maxsize)


path = "/Study Namal/5th Semester/Artificial Intelligence/Classification/data"
dir_list = os.listdir(path) 

'''
Image slicing or croping feature
'''
def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

for i in dir_list:
    Train_path = "/Study Namal/5th Semester/Artificial Intelligence/Classification/data/train"
    folder = os.listdir(Train_path)
images = []
LIST = []
for i in folder:
    count=len(os.listdir(Train_path+"\\"+i))
    LIST.append(count)
    for filename in os.listdir(Train_path+"\\"+i):
        if any([filename.endswith(x) for x in ['.jpeg', '.jpg','.png']]):
            image = img.imread(Train_path+"\\"+i+"\\"+filename)
            # image = np.resize(image, (300, 300))
            if image is not None:
                image=crop_center(image,10,10)
                images.append([image, i])
                
train_x = []
train_y = []
for x in images:
    train_x.append(x[0])
    train_y.append(x[1])

train_x_final = []
'''
Count number of zero's in total image for training data
'''

# for i in train_x:
#     i=np.round(i)
#     train_x_final.append(cv2.countNonZero(i))

'''
Diagonal of each image
'''

# for i in train_x:
#     i=np.round(i)
#     diagonal = i.diagonal()
#     train_x_final.append(cv2.countNonZero(diagonal))

'''
Flip Diagonal of each image
'''
# for i in train_x:
#     i=np.round(i)
#     flip_diagonal = np.fliplr(i).diagonal()
#     train_x_final.append(np.count_nonzero(flip_diagonal==0))

'''
Three feature extraction in one
'''
# for i in train_x:
#     i=np.round(i)
#     count=np.count_nonzero(i==0)
#     diagonal = i.diagonal()
#     diagonal = np.count_nonzero(diagonal==0)
#     flip_diagonal = np.fliplr(i).diagonal()
#     flip_diagonal=np.count_nonzero(flip_diagonal==0)
#     train_x_final.append([diagonal,flip_diagonal,count])
    
path = "/Study Namal/5th Semester/Artificial Intelligence/Classification/data"
dir_list = os.listdir(path) 

for i in dir_list:
    val_path = "/Study Namal/5th Semester/Artificial Intelligence/Classification/data/val"
    folder = os.listdir(val_path)
images = []
LIST = []
for i in folder:
    count=len(os.listdir(val_path+"\\"+i))
    LIST.append(count)
    for filename in os.listdir(val_path+"\\"+i):
        if any([filename.endswith(x) for x in ['.jpeg', '.jpg','.png']]):
            image = img.imread(val_path+"\\"+i+"\\"+filename)
            image = np.resize(image, (300, 300))
            if image is not None:
                image=crop_center(image,10,10)
                images.append([image, i])


val_x = []
val_y = []
for x in images:
    val_x.append(x[0])
    val_y.append(x[1])
    
val_x_final = []
'''
Count number of zero's in total image for validation data
'''
# for i in val_x: 
#     i=np.round(i)
#     val_x_final.append(cv2.countNonZero(i))   

'''
Diagonal of each image
'''

# for i in val_x:
#     i=np.round(i)
#     diagonal = i.diagonal()
#     val_x_final.append(cv2.countNonZero(diagonal))

'''
Flip Diagonal of each image
'''
# for i in val_x:
#     i=np.round(i)
#     flip_diagonal = np.fliplr(i).diagonal()
#     val_x_final.append(np.count_nonzero(flip_diagonal==0))

'''
Three feature extraction in one
'''
# for i in val_x:
#     i=np.round(i)
#     count=np.count_nonzero(i==0)
#     diagonal = i.diagonal()
#     diagonal = np.count_nonzero(diagonal==0)
#     flip_diagonal = np.fliplr(i).diagonal()
#     flip_diagonal=np.count_nonzero(flip_diagonal==0)
#     val_x_final.append([diagonal,flip_diagonal,count])

train_x_final = np.array(train_x_final)
train_x_final = train_x_final.reshape(len(train_x_final), -1)

train_y = np.array(train_y)
train_y = train_y.reshape(len(train_y), -1)

val_x_final = np.array(val_x_final)
val_x_final = val_x_final.reshape(len(val_x_final), -1)

val_y = np.array(val_y)
val_y = val_y.reshape(len(val_y), -1)

'''
MLP Classifier
'''

clf = MLPClassifier(hidden_layer_sizes=(200,100,50), activation='logistic', learning_rate_init=0.01,solver='sgd',
                    verbose = True, max_iter=400).fit(train_x_final, train_y)

'''
MLP Classifier
'''
# mlp_gs = MLPClassifier(max_iter=100, verbose = True)
# parameter_space  = {
#     'hidden_layer_sizes' : (10,30,10),
#     'activation' : ['relu','tanh','logistic'],
#     'solver' : ['sgd','ada,'],
#     'alpha' : [0.0001,0.05],
#     'learning_rate' : ['constant', 'adaptive'],
# }
# clf = GridSearchCV(mlp_gs, parameter_space, n_jobs = -1, cv = 2)
# clf.fit(train_x_final, train_y)

result = clf.predict(val_x_final)
print(result)
accuracy = 0
for i in range(len(result)):
    accuracy += 1 if result[i] == val_y[i] else 0
learning = (accuracy/len(result))*100
print(learning)