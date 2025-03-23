from helper_funcs import *
import numpy as np
import trimesh
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output_path",type=str,help="path of the output folder")
parser.add_argument("--iteration",type=int,help="iteration to be visualized")
args = parser.parse_args()

path = args.output_path
it = args.iteration

hand = f"{path}/it{it}_hand.obj"
object = f"{path}/it{it}_object.obj"

obj_mesh = trimesh.load(object)
hand_mesh = trimesh.load(hand)

contact, interpenetrating_vertices, interpenetrating_indices = detect_contact_simple(hand_mesh=hand_mesh, obj_mesh=obj_mesh)

interpenetrating_vertices = np.array(interpenetrating_vertices)

hand_mesh.visual.vertex_colors[interpenetrating_indices] = [12, 7, 134, 255]

scene = trimesh.Scene()
scene.add_geometry(hand_mesh)
scene.add_geometry(obj_mesh)
scene.show()
