import os

imgpath = r'./VOC2007/JPEGImages'
xmlpath = r'./VOC2007/Annotations'


if __name__ == '__main__':
    print("images", len(os.listdir(imgpath)))
    print("annos", len(os.listdir(xmlpath)))

    count = 0
    for img in os.listdir(imgpath):
        xml = img.replace('.jpg', '.xml')
        testpath = os.path.join(xmlpath, xml)
        if not os.path.exists(testpath):
            print(img)
            count += 1
    print(count)
#    for xml in os.listdir(xmlpath):
#        img = xml.replace('.xml', '.jpg')
#        
#        testpath = os.path.join(imgpath, img)
#        if not os.path.exists(testpath):
#            print(xml)

