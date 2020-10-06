#!/usr/bin/env python
# coding: utf-8

# In[27]:


def re_m(L, U, P):
    return L @ U @ np.linalg.inv(P)



def diy_LU_v3(a):
    N = a.shape[0]
    U = a.copy()
    P = np.eye(N)
    L = np.eye(N)
    for k in range(0, N - 1):
        max_element = k + np.argmax(abs(U[k:, k]))
        if max_element != k:
            P_temp = np.eye(N)
            P_temp[k,k], P_temp[max_element,max_el] = 0, 0
            P_temp[k,max_element], P_temp[max_element,k] = 1, 1
            P = P_temp @ P
            L = P_temp @ L
            U = P_temp @ U
            lam = np.eye(N)
            gamma = U[k+1:, k] / U[k, k]
            lam[k+1:, k] = -gamma
            U = lam @ U
            L = lam @ L
            L = np.linalg.inv(L @ np.linalg.inv(P))
            return P, L, U
        np.set_printoptions(precision=3)
        P, L, U = diy_LU_v3(a1)
        print (P, "\n"*2, L, "\n"*2, U, "\n")
        print(L @ U - P @ a1, "\n")
        print (restore_matrix(L, U, P), "\n")
        print (a1)
        
        
        
        

        print (P, "\n"*2, L, "\n"*2, U, "\n")
        print(L @ U - P @ a, "\n")
        print (restore_m(L, U, P), "\n")
print (a)


# In[39]:


#5_первый способ
def woodbury_einsum(A, U, V, k):
    A_inv = np.diag(1./np.diag(A))
    tmp   = np.einsum('ab,bc,cd->ad',
                      V, A_inv, U,
                      optimize=['einsum_path', (1, 2), (0, 1)])
    B_inv = np.linalg.inv(np.eye(k) + tmp)
    tmp   = np.einsum('ab,bc,cd,de,ef->af',
                      A_inv, U, B_inv, V, A_inv,
                      optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    return A_inv - tmp
print(woodbury_einsum(A, U, V, k))


# In[37]:


#5_второй способ
def woodbury(A, U, V, k):
    A_inv = np.diag(1./np.diag(A))  #Быстрое матричное обращение диаг.
    B_inv = np.linalg.inv(np.eye(k) + V @ A_inv @ U)
    return A_inv - (A_inv @ U @ B_inv @ V @ A_inv)
n = 100000
p = 1000
k = 100
A = np.diag(np.random.randn(p))
U = np.random.randn(p, k)
V = U.T
M = U @ V + A
M_inv = woodbury(A, U, V, k)
assert np.allclose(M @ M_inv, np.eye(p))
print(woodbury(A, U, V, k))

