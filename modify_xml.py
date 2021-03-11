import xml.etree.ElementTree as ET
import random as rand

def modify_r2d2(x,y):
    file = ET.parse("r2d22.xml")
    root = file.getroot()
    for elem in root.iter('geom'):
        if elem.get("name") == "target":
            elem.set("pos", str(x) + " " + str(y) + " 0")
    file.write("r2d22.xml")