# https://pycollada.readthedocs.io/en/latest/creating.html
# https://github.com/pycollada/pycollada/blob/master/collada/tests/test_lineset.py

import collada

from collada import *
import numpy as np
import pprint
ppbase = pprint.PrettyPrinter(indent=3)
pp = lambda x: ppbase.pprint(x)

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
lineset.save()

#effect = material.Effect("effect0", [], "phong", diffuse=(1,0,0), specular=(0,1,0))
#pp(effect) #=> <Effect id=effect0 type=phong>

#pp(material) #=> <module 'collada.material' from 'C:\\Programs\\anaconda3\\envs\\py393D\\lib\\site-packages\\collada\\material.py'>

#mat = material.Material("material0", "mymaterial", effect)
#pp(mat) #=> <Material id=material0 effect=effect0>
#obj.effects.append(effect)
#obj.materials.append(mat)
#pp(obj) #=> <Collada geometries=0>




#vert_floats = [-50,50,50,50,50,50,-50,-50,50,50,-50,50,-50,50,-50,50,50,-50,-50,-50,-50,50,-50,-50]
#normal_floats = [0,0,1, 0,1,0, 0,-1,0, -1,0,0, 1,0,0, 0,0,-1]
#vert_src = source.FloatSource("cubeverts-array", numpy.array(vert_floats), ('X', 'Y', 'Z'))
#normal_src = source.FloatSource("cubenormals-array", numpy.array(normal_floats), ('X', 'Y', 'Z'))
#geom = geometry.Geometry(mesh, "geometry0", "mycube", [vert_src, normal_src])
