import os
import shutil

def write_to_file(listOfLabelFiles):
    with open(output_file, 'w') as f:
        for i in listOfLabelFiles:
            f.write(i[:-4]+"\n")

def match_pics_to_labels(labelPath, listOfLabelFiles, dirPath, listDir, destination):
    for labelFile in listOfLabelFiles:
        for picture in listDir:
            if(labelFile[:-4]==picture[:-4]):
                shutil.copyfile(labelPath+labelFile, destination+labelFile)
                shutil.copyfile(dirPath+picture, destination+picture)
                
            
        
output_file = 'out.txt'
destination='/path/to/destination/folder'

dir1='/path/to/pictures/set/1'
dir2='/path/to/pictures/set/2'
dir3='/path/to/pictures/set/3'

labelPath='/path/to/labels'

listDir1=os.listdir(dir1)
listDir2=os.listdir(dir2)
listDir3=os.listdir(dir3)
listOfLabelFiles=os.listdir(labelPath)


#write_to_file(listOfLabelFiles)
#write_to_file(listDir3)
#shutil.copyfile('/media/radek/8fcc6fb9-4004-493e-b4b6-b7efdc1df57b/klaudiamgr/labels/test/test.txt', '/media/radek/8fcc6fb9-4004-493e-b4b6-b7efdc1df57b/klaudiamgr/labels/train/test.txt')

match_pics_to_labels(labelPath, listOfLabelFiles, dir1, listDir1, destination)
match_pics_to_labels(labelPath, listOfLabelFiles, dir2, listDir2, destination)
match_pics_to_labels(labelPath, listOfLabelFiles, dir3, listDir3, destination)
