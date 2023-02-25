from abaqus import *
from abaqusConstants import *

# Create a new model database
mdb.Model(name='Plate')

# Create a 2D part
myPlate = mdb.models['Plate'].Part(name='Plate', dimensionality=TWO_D_PLANAR)

# Create a rectangular sketch
mySketch = myPlate.MakeSketch(sketchPlane=myPlate.XYPlane, name='Rectangle')
mySketch.Rectangle(point1=(0,0), point2=(10,5))

# Create a circular sketch
mySketch.CircleByCenterPerimeter(center=(5,2.5), point1=(6,2.5))

# Extrude the part
myPlate.BaseSolidExtrude(sketch=mySketch, depth=1.0)

# Create a mesh
myPlate.seedPart(size=0.5)
myPlate.generateMesh()

# Save the model database
mdb.saveAs('Plate.inp')
