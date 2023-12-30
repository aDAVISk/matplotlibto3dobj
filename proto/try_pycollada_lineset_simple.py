# https://pycollada.readthedocs.io/en/latest/creating.html
# https://github.com/pycollada/pycollada/blob/master/collada/tests/test_lineset.py

import numpy as np
import pprint
ppbase = pprint.PrettyPrinter(indent=3)
pp = lambda x: ppbase.pprint(x)


import collada
from collada.xmlutil import etree
tostring = etree.tostring

obj = collada.Collada(validate_output=True)
#pp(obj) #=> <Collada geometries=0>
linefloats = np.array([1, 1, -1, 1, -1, -1, -1, -0.9999998, -1, -0.9999997, 1, -1, 1, 0.9999995, 1, 0.9999994, -1.000001, 1])
linefloatsrc = collada.source.FloatSource("mylinevertsource",linefloats, ('X', 'Y', 'Z'))
#pp(linefloatsrc) #=> <FloatSource size=6>
geom = collada.geometry.Geometry(obj, "geometry0", "mygeometry", [linefloatsrc])
#pp(geom) #=> <Geometry id=geometry0, 0 primitives>
input_list = collada.source.InputList()
input_list.addInput(0, 'VERTEX', "#mylinevertsource")
#pp(input_list) #=> <InputList>
indices = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5])
lineset = geom.createLineSet(indices, input_list, "mymaterial")
#pp(lineset) #=> <LineSet length=5>

# Put all the data to the internal xml node (xmlnode) so it can be serialized.
# https://pycollada.github.io/reference/generated/collada.DaeObject.html#collada.DaeObject.save
#lineset.save()
#pp(lineset.xmlnode) #=> <Element '{http://www.collada.org/2005/11/COLLADASchema}lines' at 0x0000014EF3A94680>
#xmltxt = tostring(lineset.xmlnode)
#pp(xmltxt) #=> (b'<lines xmlns="http://www.collada.org/2005/11/COLLADASchema" count="5" materi'\n b'al="mymaterial"><input offset="0" semantic="VERTEX" source="#mylinevertsourc'\n b'e" /><p>0 1 1 2 2 3 3 4 4 5</p></lines>')

geom.primitives.append(lineset)
obj.geometries.append(geom)



effect = collada.material.Effect("effect0", [], "phong", diffuse=(1,0,0), specular=(0,1,0))
#pp(effect) #=> <Effect id=effect0 type=phong>
#pp(material) #=> <module 'collada.material' from 'C:\\Programs\\anaconda3\\envs\\py393D\\lib\\site-packages\\collada\\material.py'>
obj.effects.append(effect)

mat = collada.material.Material("material0", "mymaterial", effect)
#pp(mat) #=> <Material id=material0 effect=effect0>
obj.materials.append(mat)


matnode = collada.scene.MaterialNode("materialref", mat, inputs=[])
geomnode = collada.scene.GeometryNode(geom, [matnode])
node = collada.scene.Node("node0", children=[geomnode])
myscene = collada.scene.Scene("myscene", [node])
obj.scenes.append(myscene)
obj.scene = myscene

obj.write("proto/tmp/testline.dae")


#obj.effects.append(effect)
#obj.materials.append(mat)
#pp(obj) #=> <Collada geometries=0>




#vert_floats = [-50,50,50,50,50,50,-50,-50,50,50,-50,50,-50,50,-50,50,50,-50,-50,-50,-50,50,-50,-50]
#normal_floats = [0,0,1, 0,1,0, 0,-1,0, -1,0,0, 1,0,0, 0,0,-1]
#vert_src = source.FloatSource("cubeverts-array", numpy.array(vert_floats), ('X', 'Y', 'Z'))
#normal_src = source.FloatSource("cubenormals-array", numpy.array(normal_floats), ('X', 'Y', 'Z'))
#geom = geometry.Geometry(mesh, "geometry0", "mycube", [vert_src, normal_src])
