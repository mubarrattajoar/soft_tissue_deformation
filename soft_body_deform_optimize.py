import copy

import trimesh
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from helper_funcs import *
from model import *
import argparse

class MeshDeformer():
    def __init__(self,scene, exp_name, max_iter, with_diffuse = True, apply_laplacian = False):
        self.hand_mesh = None
        self.object_mesh = None
        self.hand_mesh_deformed = None
        self.object_mesh_deformed = None
        self.with_diffuse = with_diffuse
        self.apply_laplacian = apply_laplacian

        self.scene = scene
        self.exp_name = exp_name
        if not os.path.exists(f"results/{self.scene}/{self.exp_name}/"):
            os.makedirs(f"results/{self.scene}/{self.exp_name}/")

        self.iter = max_iter
        self.sampled_verts = None
        self.weights = {}
        # self.optimzer = torch.optim.Adam(self.hand_mesh.vertices.tolist(), lr=1e-3)

    def detect_contact_simple(self, hand_mesh, obj_mesh):
        S, I, C = igl.signed_distance(hand_mesh.vertices, obj_mesh.vertices, obj_mesh.faces, return_normals=False)
        inside_sample_index = np.argwhere(S < 0.0)
        inside_samples = hand_mesh.vertices[inside_sample_index[:, 0], :]
        inside_samples_faces = hand_mesh.faces[inside_sample_index[:, 0], :]
        if len(inside_samples) == 0:
            return 0, [], []

        else:
            return 1, inside_samples.tolist(), inside_sample_index.reshape((-1)).tolist()

    def energy_contact(self, hand_mesh, deformed_v_dict, sampled_verts, obj_mesh,lambda_cont = 0.2, with_diffuse  = False):
        S, I, C = igl.signed_distance(hand_mesh.vertices[sampled_verts], obj_mesh.vertices, obj_mesh.faces)
        inside_sample_index = np.argwhere(S < 0.0)
        inside_samples = hand_mesh.vertices[inside_sample_index[:, 0], :]
        inside_samples_faces = hand_mesh.faces[inside_sample_index[:, 0], :]
        if len(inside_samples) == 0:
            return 0.0
        else:
            total_loss = 0

            deepest_arg = np.argwhere(S == S.min())[0][0]
            deepest_vert = sampled_verts[int(deepest_arg)]
            deepest_vert_target_val = find_closest_vertex(mesh=obj_mesh,query_point=hand_mesh.vertices[deepest_vert])

            deepest_vert_pred_val = deformed_v_dict[deepest_vert]
            total_loss += torch.linalg.norm(deepest_vert_pred_val - torch.tensor(deepest_vert_target_val,dtype=torch.float32),axis=1)

            assert deepest_arg in inside_sample_index

            if with_diffuse == True:
                #diffusing deformation to surrounding regions
                surrounding_v_ids = sampled_verts.copy()
                surrounding_v_ids.remove(deepest_vert)
                for i,vert_s in enumerate(surrounding_v_ids):
                    offset = 0
                    for j,vert_d in enumerate(surrounding_v_ids):
                        if i != j:
                            geod = torch.tensor(calc_geodesic_distance(mesh=hand_mesh,source_v1=vert_s,source_v2=vert_d))
                            impact_factor = torch.tensor(S.min())*torch.exp(-1*lambda_cont*geod)
                            vert_d_normal = torch.tensor(find_vertex_normal(mesh=hand_mesh,vertex_index= vert_d))

                            offset +=  vert_d_normal*impact_factor

                    offset = offset/len(surrounding_v_ids)
                    vert_target_val = torch.tensor(hand_mesh.vertices[vert_s]) - offset
                    total_loss+= 0.02*torch.linalg.norm((deformed_v_dict[vert_s] - vert_target_val),axis = 1)

            return total_loss[0]


    def energy_contact_modified(self, hand_mesh, deformed_v_dict, sampled_verts, obj_mesh):
        S, I, C = igl.signed_distance(hand_mesh.vertices[sampled_verts], obj_mesh.vertices, obj_mesh.faces)
        inside_sample_index = np.argwhere(S < 0.0)
        inside_sample_index = np.array(sampled_verts)[
            inside_sample_index]  # index values changed to values as recorded in mesh
        inside_samples = hand_mesh.vertices[inside_sample_index]
        outside_sample_index = set(sampled_verts) - set(
            list(inside_sample_index.reshape(inside_sample_index.shape[0], )))
        hand_mesh_normals = igl.per_vertex_normals(hand_mesh.vertices, hand_mesh.faces)

        inside_samples_faces = hand_mesh.faces[inside_sample_index[:, 0], :]

        if len(inside_samples) == 0:
            return 0.0
        else:
            total_loss = 0

            for inside_vert_id in inside_sample_index:
                sampled_vert = inside_vert_id
                inside_vert_target_val = find_closest_vertex(mesh=obj_mesh,
                                                             query_point=hand_mesh.vertices[sampled_vert])
                inside_vert_pred_val = deformed_v_dict[sampled_vert[0]]
                total_loss += torch.linalg.norm(
                    inside_vert_pred_val - torch.tensor(inside_vert_target_val, dtype=torch.float32), axis=1)

            for outside_vert_id in outside_sample_index:
                outside_vert_normal = hand_mesh_normals[outside_vert_id]
                outside_vert_target = hand_mesh.vertices[outside_vert_id] + 0.1 * (
                            outside_vert_normal * np.abs(np.min(S)))
                outside_vert_pred_val = deformed_v_dict[outside_vert_id]
                total_loss += torch.linalg.norm(
                    outside_vert_pred_val - torch.tensor(outside_vert_target, dtype=torch.float32), axis=1)

            return total_loss[0]

    def energy_regularization(self,model,hand_mesh):
        total_loss = 0
        for vert in self.sampled_verts:
            for neighbour in hand_mesh.vertex_neighbors[vert]:
                if neighbour in  self.sampled_verts:
                    rot = model.rot_params[str(vert)+str("_R")].parameter*(torch.tensor(hand_mesh.vertices[neighbour]) - torch.tensor(hand_mesh.vertices[vert]))
                    offset_n = -torch.tensor(hand_mesh.vertices[neighbour]) - model.transl_params[str(neighbour)+"_t"].parameter
                    offsen_v = model.transl_params[str(vert)+"_t"].parameter + torch.tensor(hand_mesh.vertices[vert])
                    loss = self.weights[vert][neighbour]* torch.linalg.norm(rot + offset_n + offsen_v)
                    total_loss += loss

        return total_loss


    def energy_rigidity(self, model, hand_mesh):
        total_loss = 0
        for vert in self.sampled_verts:
            rot = model.rot_params[str(vert) + str("_R")].parameter
            for col in range(1, 3):
                total_loss+=torch.square(torch.dot(rot[:,col-1].T,rot[:,col]))
                for col2 in range(3):
                    total_loss+=torch.square(1-torch.dot(rot[:,col2].T,rot[:,col2]))
        return total_loss

    def energy_laplacian(self,hand_mesh,model):
        total_loss = 0
        for vert in self.sampled_verts:
            neighbours = hand_mesh.vertex_neighbors[vert]
            vert_pos = np.mean(hand_mesh.vertices[neighbours], axis = 0)
            pred_vert_pos = model.rot_params[str(vert)+str("_R")].parameter*(torch.tensor(hand_mesh.vertices[vert])) + model.transl_params[str(vert)+str("_t")].parameter

            total_loss += torch.linalg.norm(pred_vert_pos - torch.tensor(vert_pos))

        return total_loss

    def apply_laplacian_smoothing(self, hand_mesh):
        for vert in self.sampled_verts:
            neighbours = hand_mesh.vertex_neighbors[vert]
            vert_pos = np.mean(hand_mesh.vertices[neighbours], axis = 0)
            hand_mesh.vertices[vert] = vert_pos

        return hand_mesh


    def optimizer_loop(self,hand_mesh_path, object_mesh_path):
        self.hand_mesh = trimesh.load(hand_mesh_path)
        self.object_mesh = trimesh.load(object_mesh_path)

        self.hand_mesh.export(f"results/{self.scene}/{self.exp_name}/it{0}_hand.obj")
        self.object_mesh.export(f"results/{self.scene}/{self.exp_name}/it{0}_object.obj")
        self.undeformed_hand_mesh = copy.deepcopy(self.hand_mesh)

        self.contact, self.interpenetrating_vertices, self.interpenetrating_indices = self.detect_contact_simple(self.hand_mesh, self.object_mesh)
        self.interpenetrating_vertices = np.array(self.interpenetrating_vertices)
        self.sampled_verts = self.interpenetrating_indices.copy()

        if self.contact == 1:
            loss_data = []
            #sampling verts for optimization
            connected_neighbours_level = 2
            for vert in self.interpenetrating_indices:
                neighbour_nodes = vertices_within_k_degrees(mesh=self.hand_mesh, vertex_index=vert, k=connected_neighbours_level)
                self.sampled_verts.extend(list(neighbour_nodes))

            self.sampled_verts = list(set(self.sampled_verts))

            #calculate weights
            k = 4
            for vert in self.sampled_verts:
                v = self.hand_mesh.vertices[vert]
                v_neighbour_id= self.hand_mesh.vertex_neighbors[vert]
                v_neighbour = self.hand_mesh.vertices[v_neighbour_id]
                diff = np.linalg.norm(v - v_neighbour, axis=1)

                #calculate dist_max
                dist_max_id = vertices_within_k_degrees(mesh=self.hand_mesh, vertex_index=vert, k=k+1)
                dist = [calc_geodesic_distance(mesh=self.hand_mesh,source_v1=vert,source_v2=i) for i in dist_max_id]
                dist_max = max(dist)

                weight_v = np.square(1 - diff/dist_max)
                weight_v = weight_v/np.sum(weight_v)
                neighbour_w_dict = dict(zip(v_neighbour_id, weight_v))
                self.weights.update({vert: neighbour_w_dict})


            #initiate As, Ts
            model = DeformParams(self.weights, self.sampled_verts)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            for it in tqdm(range(self.iter)):
                #the loop

                deformed_verts, deformed_v_dict, deformed_mesh = model(mesh=self.hand_mesh)

                if self.with_diffuse:
                    contact_loss = self.energy_contact(hand_mesh=deformed_mesh,
                                                                deformed_v_dict=deformed_v_dict,
                                                                sampled_verts=self.sampled_verts,
                                                                obj_mesh=self.object_mesh,
                                                                with_diffuse=self.with_diffuse)

                else:
                    contact_loss = self.energy_contact_modified(hand_mesh=deformed_mesh, deformed_v_dict = deformed_v_dict,
                                           sampled_verts=self.sampled_verts, obj_mesh= self.object_mesh)

                loss = (5*contact_loss + 2*self.energy_regularization(model=model, hand_mesh= deformed_mesh) +
                        self.energy_rigidity(model = model, hand_mesh= deformed_mesh))

                if contact_loss== 0.0:
                    self.hand_mesh_deformed = deformed_mesh
                    deformed_mesh.export(f"results/{self.scene}/{self.exp_name}/it{it}_hand.obj")
                    self.object_mesh.export(f"results/{self.scene}/{self.exp_name}/it{it}_object.obj")
                    break

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                loss_data.append(loss.item())

                if apply_laplacian:
                    deformed_mesh = self.apply_laplacian_smoothing(deformed_mesh)

                if (it+1)%2 == 0:
                    deformed_mesh.export(f"results/{self.scene}/{self.exp_name}/it{it}_hand.obj")
                    self.object_mesh.export(f"results/{self.scene}/{self.exp_name}/it{it}_object.obj")

                plt.plot(loss_data)
                plt.savefig(f"results/{self.scene}/{self.exp_name}/loss.png")


    def visualize(self, order = "after"):

        scene = trimesh.Scene()
        self.hand_mesh.visual.vertex_colors[self.interpenetrating_indices] = [12, 7, 134, 255]

        if order == "before":
            scene.add_geometry(self.undeformed_hand_mesh)
            scene.add_geometry(self.object_mesh)

        elif order == "after":
            scene.add_geometry(self.hand_mesh_deformed)
            scene.add_geometry(self.object_mesh)

        scene.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--with_diffuse", type=str, help="True or False")
    parser.add_argument("--apply_laplacian", type=str, help="True or False")
    parser.add_argument("--scene", type=str, help="name of the scene")
    parser.add_argument("--exp", type=str, help="name of the experiment")
    parser.add_argument("--hand_path", type=str, help="path to the hand mesh")
    parser.add_argument("--obj_path", type=str, help="path to the obj mesh")
    parser.add_argument("--visualize", type=str, help="True or False")
    args = parser.parse_args()

    scene = args.scene
    exp_name = args.exp
    hand_mesh_path = args.hand_path
    obj_mesh_path = args.obj_path

    if args.with_diffuse == "True" or args.with_diffuse == "true":
        with_diffuse = True
    elif args.with_diffuse == "False" or args.with_diffuse == "false":
        with_diffuse = False

    if args.apply_laplacian == "True" or args.apply_laplacian == "true":
        apply_laplacian = True
    elif args.apply_laplacian == "False" or args.apply_laplacian == "false":
        apply_laplacian = False

    if args.visualize == "True" or args.visualize == "true":
        visualize = True
    elif args.visualize == "False" or args.visualize == "false":
        visualize = False


    deformer = MeshDeformer(scene = scene,exp_name = exp_name,max_iter=1000, with_diffuse=with_diffuse, apply_laplacian=apply_laplacian)
    deformer.optimizer_loop(hand_mesh_path, obj_mesh_path)

    if visualize:
        deformer.visualize(order="before")
        deformer.visualize(order="after")
