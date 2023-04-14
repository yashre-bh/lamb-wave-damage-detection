from abaqus import *
from abaqusConstants import *

# Create model and part
myModel = mdb.Model(name='FlatPlate')
myPart = myModel.Part(name='Plate', dimensionality=2, type=DEFORMABLE_BODY)
mySketch = myPart.BaseSketch(plane=XYPLANE)

# Draw plate
mySketch.rectangle(point1=(0.0, 0.0), point2=(10.0, 1.0))
myPart.BaseWire(sketch=mySketch)

# Create actuator and receiver
myPart.DatumPointByCoordinate(coords=(1.0, 0.0, 0.0))
myPart.DatumPointByCoordinate(coords=(9.0, 0.0, 0.0))

# Create circular notch
myPart.DatumPointByCoordinate(coords=(5.0, 0.5, 0.0))
myPart.CircleByCenterPerimeter(center=(5.0, 0.5, 0.0), point1=(5.1, 0.5, 0.0))

# Mesh part
elemType1 = mesh.ElemType(elemCode=STANDARD, elemLibrary=STANDARD)
elemType2 = mesh.ElemType(elemCode=STANDARD, elemLibrary=STANDARD,
                          secondOrderAccuracy=OFF, hourglassControl=DEFAULT)
myPart.setElementType(regions=(myPart.cells,), elemTypes=(elemType1, elemType2))
myPart.seedPart(size=0.1)
myPart.generateMesh()

# Create assembly and instance of part
myAssembly = myModel.rootAssembly
myAssembly.Instance(name='Plate-1', part=myPart)

# Create step and load
myModel.StaticStep(name='Step-1', previous='Initial')
myAssembly.Set(name='Actuator', vertices=myAssembly.instances['Plate-1'].vertices.getClosest(
    coordinates=(1.0, 0.0, 0.0), numberOfVertices=1))
myAssembly.Set(name='Receiver', vertices=myAssembly.instances['Plate-1'].vertices.getClosest(
    coordinates=(9.0, 0.0, 0.0), numberOfVertices=1))
myModel.FieldOutputRequest(name='F-Output-1',
                            createStepName='Step-1', variables=('S', 'U', 'RF'))

# Run simulations and record notch position
notch_pos = []
for i in range(260):
    myPart.features['Point-2'].move((0.5, 0.0, 0.0))
    myModel.jobName = 'Sim-' + str(i)
    myJob = mdb.Job(name=myModel.jobName, model=myModel)
    myJob.submit()
    myJob.waitForCompletion()
    odb = session.openOdb(name=myModel.jobName + '.odb')
    lastFrame = odb.steps['Step-1'].frames[-1]
    notch_pos.append(lastFrame.fieldOutputs['U'].getScalarField(invariant=MAGNITUDE).values[2].data)
    odb.close()

print(notch_pos)
