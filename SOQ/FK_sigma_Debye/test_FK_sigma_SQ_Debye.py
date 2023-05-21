#!/usr/bin/env python
# coding: utf-8

# In[1]:


from FK import *
import os
import scipy.interpolate as interp


# In[5]:


n_x = 6
n_y = 6
n_layers = 11

r_ca = [0.426,0.459,0.473,0.496,0.511,0.515,0.522,0.533,0.535]
# r_ca = [0.426,0.511,0.535]

SQ_list = []
SQ_scaled_list = []
for i_r, r in enumerate(r_ca):
    e_SQ = np.array([[np.cos(x),np.sin(x)] for x in [0,np.pi/2]])
    e_TR = np.array([[np.cos(x),np.sin(x)] for x in [0,np.pi/3]])
    
    Ratio_ca = 1/r
    l_c = 1*np.sqrt((1+np.sqrt(3)/2)**2+0.5**2)*r
    l_a = 1*np.sqrt((1+np.sqrt(3)/2)**2+0.5**2)
    l_square = 1

    l_c_s = 2*np.pi/l_c
    l_a_s = 2*np.pi/l_a
    
    d_c = l_c*np.array([0,0,1])
    d_a1 = l_a*np.array([1,0,0])
    d_a2 = l_a*np.array([0,1,0])
    
    d_c_s = 2*np.pi*np.cross(d_a1, d_a2)/np.dot(np.cross(d_a1, d_a2),d_c)
    d_a1_s = 2*np.pi*np.cross(d_a2, d_c)/np.dot(np.cross(d_a2, d_c),d_a1)
    d_a2_s = 2*np.pi*np.cross(d_c, d_a1)/np.dot(np.cross(d_c, d_a1),d_a2)
    
    d_s = np.vstack([d_a1_s,d_a2_s,d_c_s])

    d_002 = 2*np.pi/np.linalg.norm(np.array([0,0,2])@d_s)
    Q_002 = np.linalg.norm(np.array([0,0,2])@d_s)
    
    d_410 = 2*np.pi/np.linalg.norm(np.array([4,1,0])@d_s)
    Q_410 = np.linalg.norm(np.array([4,1,0])@d_s)
    
    
    # Generate trajectory
    c_layer_FK = c_sigma(n_x,n_y,Ratio_ca=1/r)
    c_rod = stack_coords([shift_coords(c_layer_FK, np.array([0,0,l_c])*s) for s in range(n_layers)])

    l = np.sqrt((1+np.sqrt(3)/2)**2+0.5**2)
    bounds = np.array([[0,n_x*l],[0,n_y*l],[0,n_layers*l_c]])
    
    d_fluc=0.25
    c_rod_fluc = [c_rod[clr]+np.random.normal(0.0,d_fluc,size=[c_rod[clr].shape[0],c_rod[clr].shape[1]])*d_fluc 
                  for clr in range(3)]
    
    c_all = np.vstack(c_rod)
    # Evaluate scattering function
    Q = np.linspace(0.05,50,1000)
    SQ = scatter_histo(c_all.T, Q, p_sub=1.0, n_bins=1000)
    SQ_list.append(SQ)
    
    # Interpolation
    Q_scaled = np.linspace(0.04,16,400)
    f = interp.interp1d(Q*d_410,SQ)
    SQ_scaled = f(Q_scaled)
    SQ_scaled_list.append(SQ_scaled.tolist())
    
SQ_list = np.array(SQ_list)
SQ_scaled_list = np.array(SQ_scaled_list)


# In[6]:


np.savetxt('SQ_scaled.csv', SQ_scaled_list.T, delimiter=',')
np.savetxt('Q_scaled.csv', Q_scaled, delimiter=',')
np.savetxt('r_ca.csv', r_ca, delimiter=',')


# In[ ]:




