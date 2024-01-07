import numpy as np
from matplotlib import colormaps as cmp
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from progressbar import progressbar

def generate_tube_surface_line(line_points, radius=0.1, num_segments=10):
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
    #print((num_points, num_segments))
    v0 = (num_points-1) * num_segments
    for j in range(1,num_segments-1):
        v1 = v0 + j
        v2 = v0 + ((j + 1) % num_segments)
        faces.extend([(v0, v1, v2)])

    return vertices, faces


def line2obj(ax, objfilename, mtlfilename, radius=0.1, num_segments=10):

    #print(ax.lines)
    #print(ax.lines[0].get_data_3d())
    #print(ax.lines[0].get_color())

    linenum = len(ax.lines)
    cnt_vertex = 0
    txt_vertices = ""
    txt_faces = ""
    txt_materials = ""
    cnt = 0
    for lines in progressbar(ax.lines):
        cnt += 1
        linename = f"line-{cnt}"
        mtlname = f"mtl-{cnt}"
        line_points = np.array(lines.get_data_3d(), dtype=float).T
        vertices, faces = generate_tube_surface_line(line_points, radius=radius, num_segments=num_segments)
        for pt in vertices:
            txt_vertices += f"v {pt[1]} {pt[2]} {pt[0]}\n"
        txt_faces += f"g {linename}\n"
        txt_faces += f"usemtl {mtlname}\n"
        for face in faces:
            txt_faces += f"f {face[0]+cnt_vertex+1} {face[1]+cnt_vertex+1} {face[2]+cnt_vertex+1}\n"
        cc = lines.get_color()
        txt_color = f"{cc[0]} {cc[1]} {cc[2]}"
        txt_materials += f"newmtl {mtlname}\n"
        txt_materials += f"Ka {txt_color}\n"
        txt_materials += f"Kd {txt_color}\n"
        txt_materials += f"Ks {txt_color}\n"
        cnt_vertex += len(vertices)

    with open(objfilename, "w") as ofile:
        ofile.write(f"mtllib {mtlfilename}\n")
        ofile.write(txt_vertices)
        ofile.write(txt_faces)
    with open(mtlfilename, "w") as ofile:
        ofile.write(txt_materials)
