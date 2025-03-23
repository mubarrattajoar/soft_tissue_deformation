import numpy as np
from collections import deque
import igl

def vertices_within_k_degrees(mesh, vertex_index, k):
    visited = set()
    queue = deque([(vertex_index, 0)])

    within_k_degrees = set()

    while queue:
        current_vertex, current_distance = queue.popleft()
        visited.add(current_vertex)

        if current_distance > k:
            break

        within_k_degrees.add(current_vertex)

        neighbors = mesh.vertex_neighbors[current_vertex]

        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append((neighbor, current_distance + 1))

    return within_k_degrees


def calc_geodesic_distance(mesh, source_v1, source_v2):
    v1 = mesh.vertices[source_v1]
    v2 = mesh.vertices[source_v2]

    source_faces = np.where(mesh.faces == source_v1)
    target_faces = np.where(mesh.faces == source_v2)

    source_faces = source_faces[0]
    target_faces = target_faces[0]

    result = igl.exact_geodesic(v=np.array(mesh.vertices), f=np.array(mesh.faces),
                                vs=np.array([source_v1]), vt=np.array([source_v2]),
                                fs=source_faces, ft=target_faces)
    distances = result

    return distances[0]



def find_closest_vertex(mesh, query_point):
    distances = np.linalg.norm(np.asarray(mesh.vertices) - query_point, axis=1)
    closest_vertex_index = np.argmin(distances)
    closest_vertex = np.asarray(mesh.vertices)[closest_vertex_index]
    return closest_vertex.reshape(1, -1)


def find_vertex_normal(mesh, vertex_index):
    adjacent_faces = mesh.vertex_adjacency_graph[vertex_index]

    normal_sum = [0.0, 0.0, 0.0]
    for face_index in adjacent_faces:
        normal_sum += mesh.face_normals[face_index]
    average_normal = normal_sum / len(adjacent_faces)
    return average_normal


def detect_contact_simple( hand_mesh, obj_mesh):
    S, I, C = igl.signed_distance(hand_mesh.vertices, obj_mesh.vertices, obj_mesh.faces, return_normals=False)
    inside_sample_index = np.argwhere(S < 0.0)
    inside_samples = hand_mesh.vertices[inside_sample_index[:, 0], :]
    inside_samples_faces = hand_mesh.faces[inside_sample_index[:, 0], :]
    if len(inside_samples) == 0:
        return 0, [], []

    else:
        return 1, inside_samples.tolist(), inside_sample_index.reshape((-1)).tolist()