import shutil
import os
import random

source='/path/to/data/source/folder'
test='/path/to/test/data'
train='/path/to/train/data'

output_file = 'out.txt'

def write_to_file(listOfLabelFiles):
    with open(output_file, 'w') as f:
        for i in listOfLabelFiles:
            f.write(i+"\n")

def names_no_repeats(listFiles):
    finalList=[]
    for item in listFiles:
        finalList.append(item[:-4])
    return set(finalList)


def split_test_train(noRepeatsList, testPercentage:int):
    size=len(noRepeatsList)
    testNumber=size*testPercentage/100
    print(testNumber)
    randomList=random.sample(noRepeatsList, int(testNumber))
    for item in randomList:
        shutil.move(source+item+".txt", test +item+".txt")
        shutil.move(source+item+".png", test +item+".png")
    #write_to_file(randomList)
        

listFiles=os.listdir(source)
noRepeatsList=names_no_repeats(listFiles)
split_test_train(noRepeatsList, 20)


