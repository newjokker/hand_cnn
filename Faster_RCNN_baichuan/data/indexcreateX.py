import os
import shutil
import random
ROOT_PATH = r"./VOC2007/"
path = ROOT_PATH + r"JPEGImages"
ANNO_PATH = ROOT_PATH + r"Annotations"
TEST_PATH = ROOT_PATH + r"testSet"
output1 = ROOT_PATH + r"ImageSets/Main/trainval.txt"
output4 = ROOT_PATH + r"ImageSets/Main/test.txt"
testImgPath = ROOT_PATH + r"testImg"
def main():
    result = []
    for dir,folder,file in os.walk(path):
        for name in file:
            if("jpg" in name or "JPG" in name):
                fname , extension = os.path.splitext(name)
                result.append(fname)
    print(len(result))
    testNum = 0
    totalNum = len(result)
    trainNum = int(round((totalNum-testNum) / 100 * 90))
    valNum = totalNum-testNum - trainNum
    print(trainNum)
    print(valNum)
    random.seed(10)
    test = random.sample(result,testNum)
    trainval = list(set(result)^set(test))
    val = random.sample(trainval,valNum)
    train = list(set(trainval) ^ set(val))
    #train = result[:trainNum+1]
    #val = result[trainNum+1:]
    with open(output1,'w+') as fp:
        for item in train:
            fp.write(item+'\n')
    with open(output4,'w+') as fp: 
        for item in val:
            fp.write(item+'\n')
    for item in test:
        jpgpath = os.path.join(path,item+'.jpg')
        xmlpath = os.path.join(ANNO_PATH,item+'.xml')
        newjpgpath = os.path.join(TEST_PATH,item+'.jpg')
        newxmlpath = newjpgpath.replace('.jpg','.xml')
        shutil.copy(jpgpath,newjpgpath)
        shutil.copy(xmlpath,newxmlpath)
    
if __name__=="__main__":
	main()
	
