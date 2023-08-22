import numpy as np

def eulerToMatrix(r_eu):
    R_x = np.array([
        [np.cos(r_eu[0]), np.sin(r_eu[0]), 0],
        [-np.sin(r_eu[0]), np.cos(r_eu[0]), 0],
        [ 0, 0, 1]
    ])
    R_y = np.array([
        [1, 0, 0],
        [0, np.cos(r_eu[1]), -np.sin(r_eu[1])],
        [0, np.sin(r_eu[1]), np.cos(r_eu[1])]
    ])
    R_x = np.array([
        [np.cos(r_eu[2]), np.sin(r_eu[2]), 0],
        [-np.sin(r_eu[2]), np.cos(r_eu[2]), 0],
        [ 0, 0, 1]
    ])
    return R_x@R_y@R_x

def angleAndAxisToMatrix(u, theta):
    num_samples = theta.shape[0]
    cross_matrix = np.zeros((num_samples,3,3))
    cross_matrix[:,0,1] = -u[:,2]
    cross_matrix[:,0,2] = u[:,1]
    cross_matrix[:,1,2] = -u[:,0]
    cross_matrix[:,1,0] = u[:,2]
    cross_matrix[:,2,:] = -cross_matrix[:,:,2]

    R = np.cos(theta)[...,None,None]*np.eye(3)[None,...].repeat(num_samples,0) + np.sin(theta)[...,None,None]*cross_matrix + (1-np.cos(theta))[...,None,None]*np.einsum("bi,bj->bij",u,u)
    return R

def unproject_rays(K, h_rgb, w_rgb):
    umap = np.linspace(0.5, w_rgb-0.5, w_rgb)
    vmap = np.linspace(0.5, h_rgb-0.5, h_rgb)
    umap, vmap = np.meshgrid(umap, vmap, indexing='xy')
    points_2d = np.stack((umap, vmap, np.ones_like(umap)), -1)
    
    # Rays to concatenate with RGB image are unprojected with the same shape
    local_rays = np.einsum("ij,mnj -> mni",np.linalg.inv(K[:3,:3]),points_2d)
    local_rays = np.concatenate((local_rays,np.ones(local_rays.shape[:-1])[...,None]),axis=-1)

    return local_rays
################################
# Virtual cameras generation
###############################
def pointcloudCentroid(points_3d_camera, mask_depth, e_v, e_c):
    assert mask_depth.sum()>0, "Depth threshold is to high and there is no valid depth to unproject!"
    weights = mask_depth*1.0                        
    weights /= weights.sum((1,2)).reshape(-1,1,1)

    #c_i = ((points_3d_camera[...,:3] +e_v[:,None,None,:])*weights.unsqueeze(-1)).sum((1,2)) + e_c
    c_i = ((points_3d_camera[...,:3] +e_v[:,None,None,:])*weights[...,None]).sum((1,2)) + e_c
    return c_i

def rotateZtoCentroid(R, c_i):
    z = np.array([[0,0,1.0]])
    #u = torch.cross(z/z.norm(dim=-1)[...,None],c_i/c_i.norm(dim=-1)[...,None], dim=-1)
    u = np.cross(z/np.linalg.norm(z, axis=-1)[...,None],c_i/np.linalg.norm(c_i,axis=-1)[...,None], axis=-1)
    theta = np.arcsin(np.linalg.norm(u,axis=-1))
    return R @ angleAndAxisToMatrix(u/np.linalg.norm(u,axis=-1)[...,None], theta)

def generateVirtualCameras(input_T_world_camera, points_3d_camera, mask_depth, e_v, e_c):
    if isinstance(input_T_world_camera, np.ndarray):
        T_virtual = input_T_world_camera.copy()
    else:
        T_virtual = input_T_world_camera.clone()
    T_virtual[:,:3,3] += e_v 
    
    c_i = pointcloudCentroid(points_3d_camera, mask_depth, e_v, e_c)
    T_virtual[:,:3,:3] = rotateZtoCentroid(T_virtual[:,:3,:3], c_i)

    return T_virtual