#!/usr/bin/env python
# coding: utf-8

# In[5]:


#3
import numpy as np
import matplotlib.pyplot as plt

A3 = np.array([[1,1], [1,1]])
u3, s3, vh3 = np.linalg.svd(A3)

plt.arrow(0,0,*(u3[:,0]*s3[0]), color='purple')
plt.arrow(0,0,*(u3[:,1]*s3[1]))
t3 = np.arange(1, 100, 0.1)

x3 = s3[0]* np.sin(1 * t3)
y3 = s3[1] * np.cos(1 * t3)
a3= np.array(u3[:,0])
fi3 = np.angle(a3[0]+1j*a3[1])
plt.plot(np.cos(fi3)*x3-np.sin(fi3)*y3, np.sin(fi3)*x3+np.cos(fi3)*y3, linewidth=0.6)
plt.xlim(-2,2)
plt.ylim(-2,2)


# In[6]:


v3=vh3.T.conj()
plt.arrow(0,0,*(v3[:,0]), color='purple')
plt.arrow(0,0,*v3[:,1])
t3 = np.arange(1, 100, 0.1)

x3 =  np.sin(1 * t3)
y3 =  np.cos(1 * t3)

plt.plot(x3, y3, linewidth=0.6)
plt.xlim(-2,2)
plt.ylim(-2,2)


# In[11]:



A1 = np.array([[3,0], [0,-2]])
u1, s1, vh1 = np.linalg.svd(A1)
plt.arrow(0,0,*(u1[:,0]*s1[0]), color='red')
plt.arrow(0,0,*(u1[:,1]*s1[1]))
t1 = np.arange(1, 100, 0.1)
v1=vh1.T.conj()

x1 = s1[0]* np.sin(1 * t1)
y1 = s1[1] * np.cos(1 * t1)
a1= np.array(u1[:,0])
fi1 = np.angle(a1[0]+1j*a1[1])
plt.plot(np.cos(fi1)*x1-np.sin(fi1)*y1, np.sin(fi1)*x1+np.cos(fi1)*y1, linewidth=0.6)
plt.xlim(-3,3)
plt.ylim(-3,3)


# In[9]:


plt.arrow(0,0,*(v1[:,0]), color='red')
plt.arrow(0,0,*v1[:,1])
t1 = np.arange(1, 100, 0.1)

x1 =  np.sin(1 * t1)
y1 =  np.cos(1 * t1)

plt.plot(x1, y1, linewidth=0.6)
plt.xlim(-2,2)
plt.ylim(-2,2)


# In[13]:


#2 
#т.к. нужно только по 2 вектора v1,v2,u1,u2 - рассматриваем матрицу не целиком
A2 = np.array([[0,2], [0,0]])
u2, s2, vh2 = np.linalg.svd(A2)
plt.arrow(0,0,*(u2[:,0]*s2[0]), color='red')
plt.arrow(0,0,*(u2[:,1]*s2[1]))
t2 = np.arange(1, 100, 0.1)

x2 = s2[0]* np.sin(1 * t2)
y2 = s2[1] * np.cos(1 * t2)
a2= np.array(u2[:,0])
fi2 = np.angle(a2[0]+1j*a2[1])
plt.plot(np.cos(fi2)*x2-np.sin(fi2)*y2, np.sin(fi2)*x2+np.cos(fi2)*y2, linewidth=0.6)
plt.xlim(-3,3)
plt.ylim(-3,3)


# In[15]:


A2 = np.array([[0,2], [0,0]])
u2, s2, vh2 = np.linalg.svd(A2)
print(u2,s2,vh2)


# In[17]:


v2=vh2.T.conj()
plt.arrow(0,0,*(v2[:,0]), color='red')
plt.arrow(0,0,*v2[:,1])
t2 = np.arange(1, 100, 0.1)

x2 =  np.sin(1 * t2)
y2 =  np.cos(1 * t2)

plt.plot(x2, y2, linewidth=0.6)
plt.xlim(-2,2)
plt.ylim(-2,2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




