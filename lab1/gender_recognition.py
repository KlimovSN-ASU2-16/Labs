import cv2
import math
import os
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
    
def pixel_camparision(img):
    list = []
    simple_features = []
    
    for i in img:
        for j in i:
             list.append(j)
             
    for i, val_i in enumerate(list):
            for j, val_j in enumerate(list):
                if i==j:
                    continue
                else:
                    simple_features.append(int(val_i-val_j>0))
                    simple_features.append(int(math.fabs(val_i-val_j)<5))
                    simple_features.append(int(math.fabs(val_i-val_j)<10))
                    simple_features.append(int(math.fabs(val_i-val_j)<25))
                    simple_features.append(int(math.fabs(val_i-val_j)<50))
                    """ Если используем 10 параметров
                    simple_features.append(int(val_i-val_j<0))
                    simple_features.append(int(math.fabs(val_i-val_j)>5))
                    simple_features.append(int(math.fabs(val_i-val_j)>10))
                    simple_features.append(int(math.fabs(val_i-val_j)>25))
                    simple_features.append(int(math.fabs(val_i-val_j)>50))"""
    #print(len(simple_features))  
    return simple_features
    #return list

def do_train_dataset(dataset):
    
    os.chdir('./train_set/male')
    print('Working ',end=' ')
    for i in os.listdir("./"):
        print('.',end=' ')
        img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        list = pixel_camparision(img)    
        dataset.addSample(list, (1, ))
    
    os.chdir('../famale')
    
    for i in os.listdir("./"):
        print('.',end=' ')
        img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        list = pixel_camparision(img)
        dataset.addSample(list, (0, ))
        
    return dataset

def main():

    print ('20 f, 25 m, 5 par')
    net = buildNetwork(798000, 3, 1) "798000 = 399 * 400 * 5 ot 1596000 = 399*40*10"
    dataset = SupervisedDataSet(798000, 1)
    dataset = do_train_dataset(dataset)
 
    trainer = BackpropTrainer(net, dataset)
    #trainer.trainUntilConvergence()
    for epoch in range(50): "Число эпох. Ошибка почти не меняется после 30 эпох"
        trainer.train()
        print('.',end=' ')
    print()
    print ('Value of error = ',trainer.train())
    #print (trainer.trainUntilConvergence())
    
        
    os.chdir('../test')
    
    for i in os.listdir("./"):
        print(i)
        img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        list = pixel_camparision(img)
        print(net.activate(list))
    
if __name__=="__main__":
    main()
    
       
