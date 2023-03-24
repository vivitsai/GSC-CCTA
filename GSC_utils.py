
import numpy as np
from numpy import linalg as LA
from sklearn.preprocessing import normalize
#
def generate_grid_transition_matrix(row_dim):
    transition_matrix = np.zeros((row_dim**2,row_dim**2))   
    for i in range(row_dim):
        for j in range(row_dim):
            cnt = 0 
            for k in [-1,0,1]:
                for s in [-1,0,1]:
                    if (i + k) in range(row_dim) and (j + s) in range(row_dim):
                        cnt += 1
            for k in [-1,0,1]:
                for s in [-1,0,1]:
                    if (i + k) in range(row_dim) and (j + s) in range(row_dim):
                        transition_matrix[i*row_dim + j, i*row_dim + j + k*row_dim + s] = 1/float(cnt-1)
                    else:
                        continue
            transition_matrix[i*row_dim + j, i*row_dim + j] = 0
    return(transition_matrix)

def get_row_ix(row,n):
    ix = np.where(row>0)
    return zip(zip(ix[0] / n,ix[0] % n),row[ix])

def ix_to_arr(ix,n):
    return (ix / n,ix % n)

def arr_to_ix(arr,n):
    return arr[0] * n + arr[1]

def generate_Q(n,q_power):
    transition_matrix = generate_grid_transition_matrix(n)
    q_tmp = tr_0 =  np.identity(n**2)
    for k in range(1,q_power+1):
        q_tmp += np.mod(transition_matrix, k)
    return np.array(normalize(q_tmp, norm='l1', axis=1),dtype = 'float32')


