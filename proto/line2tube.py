import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

def generate_tube_surface_line(line_points, radius=0.1, num_segments=20):
    # Create vertices for the tube surface
    vertices = []
    num_verts_per_point = num_segments
    num_points = len(line_points)

    for i in range(num_points):
        # Find neighboring points for tangent calculation
        prev_index = max(0, i - 1)
        next_index = min(num_points - 1, i + 1)

        # Calculate tangent and perpendicular vectors
        tangent = line_points[next_index] - line_points[prev_index]
        tangent /= np.linalg.norm(tangent)
        rotvec = np.array([-tangent[1], tangent[0], 0])
        rotvec *= np.arccos(tangent[2])/np.linalg.norm(rotvec)
        rot = Rotation.from_rotvec(rotvec)
        #perpendicular = np.cross(tangent, np.array([0, 0, 1]))  # Choose an arbitrary vector not parallel to tangent
        #perpendicular /= np.linalg.norm(perpendicular)

        # Generate points around the circumference of the tilted circle at the current line point
        for j in range(num_segments):
            angle = 2 * np.pi * j / num_segments
            circle_point = np.array([radius * np.cos(angle), radius * np.sin(angle), 0])

            # Tilt the circular points based on tangent and perpendicular vectors
            #tilted_point = circle_point[0] * tangent + circle_point[1] * perpendicular
            tilted_point = rot.apply(circle_point)
            vertices.append(line_points[i] + tilted_point)

    # Create faces by connecting the vertices
    faces = []
    for i in range(num_points - 1):
        for j in range(num_segments):
            v0 = i * num_segments + j
            v1 = ((i + 1) * num_segments + j) % (num_segments * len(line_points))
            v2 = ((i + 1) * num_segments + (j + 1) % num_segments) % (num_segments * len(line_points))
            v3 = i * num_segments + (j + 1) % num_segments

            faces.extend([(v0, v1, v2), (v0, v2, v3)])

    # Create faces for the caps of the tube's start point
    for j in range(1,num_segments-1):
        v1 = j
        v2 = (j + 1) % num_segments
        faces.extend([(0, v1, v2)])

    # Create faces for the caps of the tube's end point
    print((num_points, num_segments))
    v0 = (num_points-1) * num_segments
    for j in range(1,num_segments-1):
        v1 = v0 + j
        v2 = v0 + ((j + 1) % num_segments)
        faces.extend([(v0, v1, v2)])

    return vertices, faces

# Example usage:
line_points = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 1, 2],
    [2, 2, 2],
    [2, 2, 3],
    [1, 1, 3],
    [0, 1, 2]
], dtype=float)  # List of x, y, z coordinates representing points along the line
tube_radius = 0.2
num_segments_per_unit_length = 10

vertices, faces = generate_tube_surface_line(line_points, radius=tube_radius, num_segments=num_segments_per_unit_length)

#print(len(vertices))
# The resulting 'vertices' and 'faces' can be used to create a mesh in your preferred 3D environment
# For instance, using a library like PyOpenGL or exporting to a suitable 3D file format

cxs = [pt[0] for pt in line_points]
cys = [pt[1] for pt in line_points]
czs = [pt[2] for pt in line_points]

sxs = [pt[0] for pt in vertices]
sys = [pt[1] for pt in vertices]
szs = [pt[2] for pt in vertices]

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.scatter(cxs, cys, czs, c="r")
#ax.scatter(sxs, sys, szs, c="b")
#plt.show()

#print(faces)
outtxt = "mtllib sample_material.mtl\n"
for pt in vertices:
    outtxt += f"v {pt[1]} {pt[2]} {pt[0]}\n"
outtxt += "g tube\n"
outtxt += "usemtl my_red\n"
for face in faces:
    outtxt += f"f {face[0]+1} {face[1]+1} {face[2]+1}\n"

with open("proto/tmp/sample_tube.obj", "w") as ofile:
    ofile.write(outtxt)
