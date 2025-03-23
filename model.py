import torch
import torch.nn as nn
class RotationParameter(nn.Module):
    def __init__(self):
        super(RotationParameter, self).__init__()
        self.parameter = nn.Parameter(torch.eye(3))

class TranslationParameter(nn.Module):
    def __init__(self):
        super(TranslationParameter, self).__init__()
        self.parameter = nn.Parameter(torch.zeros(1, 3))

class DeformParams(nn.Module):
    def __init__(self, weight, sampled_verts):
        super(DeformParams, self).__init__()

        self.rot_params = nn.ModuleDict({str(sampled_verts[i])+str("_R"):RotationParameter() for i in range(len(sampled_verts))})
        self.transl_params = nn.ModuleDict({str(sampled_verts[i])+str("_t"): TranslationParameter() for i in range(len(sampled_verts))})
        self.weight = weight
        self.sampled_verts = sampled_verts

    def forward(self, mesh):
        new_verts = []
        for vert_id in self.sampled_verts:
            v = torch.tensor(mesh.vertices[vert_id],dtype=torch.float32)
            v_neighbour_ids = mesh.vertex_neighbors[vert_id]


            cum_transf = torch.zeros((1,3), dtype=torch.float32)
            for v_n_id in v_neighbour_ids:
                v_neighbour = torch.tensor(mesh.vertices[v_n_id],dtype=torch.float32)
                rotated = torch.matmul(self.rot_params[str(vert_id)+str("_R")].parameter,v - v_neighbour)
                translated = rotated + v_neighbour + self.transl_params[str(vert_id)+str("_t")].parameter
                transformed = translated * self.weight[vert_id][v_n_id]
                cum_transf += transformed

            new_verts.append(cum_transf)

            #replace vert with transformed vert
            mesh.vertices[vert_id] =  cum_transf.clone().detach().numpy()

        # return dict(zip(self.sampled_verts,new_verts))
        count = 0
        result = None
        for i in new_verts:
            if count == 0:
                count += 1
                result = i
            else:
                count += 1
                result = torch.cat((result,i),dim=0)


        return result, dict(zip(self.sampled_verts,new_verts)), mesh,