import numpy as np
import pandas as pd
from numba import njit, prange
import multiprocessing as mp

# Kept under this name for the notebooks; the implementation lives in MLM.io (ASE-based).
from MLM.io import read_vasp  # noqa: E402,F401

@njit
def rotation_matrix(theta: float) -> np.ndarray:
    rad = np.deg2rad(theta)
    return np.array([[np.cos(rad), -np.sin(rad), 0.0],
                     [np.sin(rad), np.cos(rad), 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)
    
@njit
def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    norm_v1 = np.sqrt(np.dot(v1, v1))
    norm_v2 = np.sqrt(np.dot(v2, v2))
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0  # Handle zero-length vectors gracefully
    
    cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    
    # Clamp the cosine value to avoid numerical errors
    if cosine_angle < -1.0:
        cosine_angle = -1.0
    elif cosine_angle > 1.0:
        cosine_angle = 1.0
    
    angle_rad = np.arccos(cosine_angle)
    return np.rad2deg(angle_rad)  # Convert radians to degrees

@njit
def find_index_chunk(a1, a2, a3,
                     b1, b2, b3,
                     theta_chunk, combined_list,
                     dtol, ang_lat):
    final_selected_index = []
    for deg in theta_chunk:
        qualified_index = []
        rot_matrix = rotation_matrix(deg)
        b_1r = rot_matrix @ b1.T
        b_2r = rot_matrix @ b2.T
        b_3r = rot_matrix @ b3.T
        
        for n1 in combined_list:
            for n2 in combined_list:
                for m1 in combined_list:
                    for m2 in combined_list:
                        v1 = n1 * a1 + n2 * a2
                        v2 = m1 * b_1r + m2 * b_2r
                        vec_del = np.linalg.norm(v1 - v2)
                        if vec_del < dtol:
                            qualified_index.append((n1, n2, vec_del))
                            
        for i in range(len(qualified_index)):
            n_1, n_2, v_d = qualified_index[i][0], qualified_index[i][1], qualified_index[i][2]
            for j in range(len(qualified_index)):
                n_1p, n_2p = qualified_index[j][0], qualified_index[j][1]
                A1 = n_1 * a1 + n_2 * a2
                A2 = n_1p * a1 + n_2p * a2
                #norm_A1 = np.linalg.norm(A1)
                #norm_A2 = np.linalg.norm(A2)
                angle_A1_A2 = angle_between(A1,A2)
                delta_angle = np.abs(angle_A1_A2 - ang_lat)
                if delta_angle < 0.1:
                    final_selected_index.append((deg, n_1, n_2, n_1p, n_2p, v_d, delta_angle,
                                                  np.linalg.norm(A1+A2), A1, A2))
    
                          
    return final_selected_index

def find_index(a1, a2, a3,
               b1, b2, b3,
               theta_min, theta_max, theta_step,
               n_min, n_max,
               dtol, ang_lat,num_cores):
    a1 = a1 
    a2 = a2
    a3 = a3
    b1 = b1
    b2 = b2
    b3 = b3
    dtol = dtol
    ang_lat = ang_lat
    
    theta_array = np.arange(theta_min, theta_max, theta_step)
    #theta_array = np.array([i + offset for i in range(1, 31) for offset in np.arange(-0.1, 0.1, 0.02)])
    range_1 = list(range(-n_max, -n_min + 1))
    range_2 = list(range(n_min, n_max + 1))
    combined_list = np.array(range_1 + range_2)
    
    # Split theta_array into chunks based on the number of available CPU cores
    num_cores = num_cores
    chunk_size = len(theta_array) // num_cores
    theta_chunks = [theta_array[i:i + chunk_size] for i in range(0, len(theta_array), chunk_size)]
    
    # Use multiprocessing to execute find_index_chunk on each chunk
    with mp.Pool(processes=num_cores) as pool:
        results = pool.starmap(find_index_chunk, [(a1, a2, a3, b1, b2, b3, chunk, combined_list, dtol, ang_lat) for chunk in theta_chunks])
    
    # Combine results from all chunks
    qualified_index = [item for sublist in results for item in sublist]
    return qualified_index