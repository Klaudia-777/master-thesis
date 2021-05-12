import os
from PIL import Image

def resize(listFiles, path):
    for item in listFiles:
        if(item.find("png") != -1):
            image = Image.open(path+item)
            new_image = image.resize((640, 480))
            new_image.save(path+item)


test='/path/to/test/data'
train='/path/to/train/data'

testList=os.listdir(test)
trainList=os.listdir(train)

resize(testList, test)
resize(trainList, train)
