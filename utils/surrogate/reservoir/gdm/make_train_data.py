import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import multiphases.model.multipolygon as mpg
import multiphases.model.polygon as pg

spoly = pg.Polygon()
spoly.read_history_csv("data/dataset/orig/gdm_(1, 0).txt")  

poly = mpg.MultPolygon()
#poly.read_history("data/dataset/train.txt")
#poly.add_polygon(spoly)
poly.from_polygon(spoly)

poly.save("data/dataset/train3.txt")