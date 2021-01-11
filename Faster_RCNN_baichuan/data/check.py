import os
import xml.etree.ElementTree as ET


def xmlRead(xmlpath):
    voc_labels = []
    tree = ET.parse(xmlpath)
    root = tree.getroot()

    # size = root.find('size')
    # h = int(size.find('height').text)
    # w = int(size.find('width').text)
    for obj in root.iter('object'):
        label = obj.find('name').text
        # prob = obj.find('prob').text
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymin = int(float(bndbox.find('ymin').text))
        ymax = int(float(bndbox.find('ymax').text))

        voc_labels.append([label, xmin, ymin, xmax, ymax])
    return voc_labels

xml_list = [xml for xml in os.listdir("VOC2007/Annotations") if xml.endswith(".xml")]
for xml in xml_list:
    xml_path = os.path.join("VOC2007/Annotations", xml)
    objects = xmlRead(xml_path)

    if len(objects) == 0:
        print(xml)
