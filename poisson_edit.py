
import numpy as np
import cv2
from scipy.sparse import linalg,lil_matrix
from scipy.sparse import *


#Poisson edit process
def get_neighbor(point):
    x,y = point
    return [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
    
def sparse_mat(points):
    num = len(points)
    A = lil_matrix((num,num))
    for i,index in enumerate(points):
        A[i,i] = 4
        for p in get_neighbor(index):
            if p in points: 
                j = points.index(p)
                A[i,j] = -1
    A = lil_matrix.tocsc(A)
    return A

def on_boundary(point,points):
    for p in get_neighbor(point):
        if p not in points: return True
    return False
  

    
def calculate_vec(points,source,target):
    num = len(points)
    b = np.zeros(num)
    for i,index in enumerate(points):
        x,y = index
        b[i] = (4 * source[x,y])- 1*source[x+1, y] - 1*source[x-1, y]- 1*source[x, y+1] -  1*source[x, y-1]
        if on_boundary(index,points):
            for p in get_neighbor(index):
                if p not in points:
                    b[i] += target[p]
    return b


def poisson_edit_lib(source,target,mask):
    mask[mask != 0] = 1
    x, y = np.nonzero(mask)
    edit_point = list(zip(x, y))
    A = sparse_mat(edit_point)
    b = calculate_vec(edit_point,source,target)
    x = linalg.cg(A, b)
    composite = np.copy(target).astype(int)
        # Place new intensity on target at given index
    for i,index in enumerate(edit_point):
            composite[index] = x[0][i]
    return composite

from scipy import sparse as sp
from scipy.sparse.linalg import spsolve

def solve_sparse(A,b,iters,err_tol,option):
	def extract_diag(A):
	    N = np.size(A,1)
	    ii = np.arange(0,N)
	    return A[ii,ii]

	x = np.ones(len(b))
	converge = False
	#initial residual
	rsnorm = np.linalg.norm(b,np.inf)
	if rsnorm == 0:
	    x = b
	    error0 = 0
	    converge = True
	else:
		if np.linalg.norm(x,np.inf) == 0:
		    res = rsnorm
		    error0 = 1
		else:
		    r = b-A*x
		    res = np.linalg.norm(r,np.inf)
		    error0 = res/rsnorm
		    if error0 <= err_tol:
		        converge = True
		res0 = res

	if option.lower() == 'jacobi':
		w= 1
		M = sp.spdiags(extract_diag(A),0,np.size(A,0),np.size(A,1))
		N = M-A
	elif option.lower() == 'gauss':
		w = 1
		M = sp.tril(A,0)
		N = M-A
	elif option.lower() == 'sor':
		w= 1
		diaV = extract_diag(A)
		M = sp.spdiags(diaV,0,np.size(A,0),np.size(A,1))+w*sp.tril(A,-1)
		N = (1-w)*sp.spdiags(diaV,0,np.size(A,0),np.size(A,1))-w*sp.triu(A,1)
	if converge:
	    iter = 0
	    flag = 0
	    return x, error0, 0,flag

	iter = 0
	error = np.zeros(iters+1)
	error[0] = error0

	while iter < iters and error[iter]>err_tol:
	    iter = iter + 1
	    x = spsolve(M,N*x+w*b)
	    r = b-A*x
	    res = np.linalg.norm(r,np.inf)
	    error[iter] = res/res0
	    
	error = error[0:iter+1]

	if(error[iter]>err_tol):
	    flag = 1
	else:
	    flag = 0
	return x, error, iter, flag
def poisson_edit(source,target,mask,iters,method):
	mask[mask != 0] = 1
	x, y = np.nonzero(mask)
	edit_point = list(zip(x, y))
	A = sparse_mat(edit_point)
	b = calculate_vec(edit_point,source,target)
	u, error , iters, converge= solve_sparse(A, b,iters,1e-10,method)

	composite = np.copy(target).astype(int)
	    # Place new intensity on target at given index
	for i,index in enumerate(edit_point):
	        composite[index] = u[i]
	return composite   



