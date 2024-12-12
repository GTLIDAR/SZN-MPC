import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial import ConvexHull

def no_ped(pred_x, pred_y):
    A_poly_out = []
    b_poly_out = []

    
    g1 = np.array([0.125,0.0])
    g2 = np.array([0.1,0.25])

    G = np.c_[g1, g2, g1*0.1, g1*0.1, g1*0.1, g1*0.1, g1*0.1]

    L = np.linalg.norm(G, axis=0)
    A_poly_in = np.zeros([np.shape(G)[1],2])

    for i in range(np.shape(G)[1]):
        A_poly_in[i,0] =  -1/L[i] * G[1,i]
        A_poly_in[i,1] =  1/L[i] * G[0,i]
    

    A_poly = np.concatenate((A_poly_in,-A_poly_in),axis=0)
    centers = np.array([pred_x,pred_y])

    ATc = np.dot(A_poly, centers)
    ATG = np.dot(A_poly, G)
    # ATG = np.concatenate((ATG,ATG),axis=0)
    ones = np.ones([np.shape(G)[1],1])
   
    
    b_poly = ATc.reshape(14,1) + np.dot(np.abs(ATG),ones)
   
    
    A_poly_out.append(A_poly)
    b_poly_out.append(b_poly)
    
    
    return np.array(A_poly), np.array(b_poly), G


def plot_polytope(center, generator_vectors, xlim=(0, 16), ylim=(0, 8)):
    
    c = center
    
    
    n = generator_vectors.shape[1]  # Number of rows
  
    binary_combinations = list(itertools.product([1, -1], repeat=n))

    combinations_matrix = np.array(binary_combinations)
  
    vertices = np.dot(combinations_matrix,generator_vectors.T) + c
   
    hull = ConvexHull(vertices)
  
    plt.fill(vertices[hull.vertices,0], vertices[hull.vertices,1], 'k', alpha=0.2)
 

    return vertices


fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
cx = 6
cy = 9
A_poly_no_ped, b_poly_no_ped, G = no_ped(cx, cy)
# print(A_poly_no_ped.shape)
# print(b_poly_no_ped.shape)

plot_polytope(np.array([[cx,cy]]), G, xlim=(0, 16), ylim=(0, 8))
point=np.array([0,0.01])
dist = np.dot(A_poly_no_ped,point).reshape(14,1)-b_poly_no_ped
print(dist)
print(-np.max(dist))

plt.scatter(point[0], point[1])
plt.scatter(cx, cy)
border = Rectangle((-2,-2), 5, 5, fc = "None", ec="black" )
ax1.add_patch(border)
plt.show()