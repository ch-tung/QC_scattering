import numpy as np

def Create_dump(c,filename,boundary=False,periodic=False):
    n_type = len(c)
    n_particles = np.sum([len(coord) for coord in c])
    
    p_string = 'f '
    if periodic:
        p_string = 'pp '
    
    if type(boundary)==bool:
        x_max = np.max([max(coord[:,0]) for coord in c])
        x_min = np.min([min(coord[:,0]) for coord in c])
        y_max = np.max([max(coord[:,1]) for coord in c])
        y_min = np.min([min(coord[:,1]) for coord in c])
        z_max = np.max([max(coord[:,2]) for coord in c])
        z_min = np.min([min(coord[:,2]) for coord in c])
    else:
        x_max = boundary[0,1]
        x_min = boundary[0,0]
        y_max = boundary[1,1]
        y_min = boundary[1,0]
        z_max = boundary[2,1]
        z_min = boundary[2,0]
    
    with open(filename, 'w') as f:
        p_id = 0
        f.write('ITEM: TIMESTEP\n')
        f.write('{:d}\n'.format(0))
        f.write('ITEM: NUMBER OF ATOMS\n')
        f.write('{:d}\n'.format(n_particles))
        f.write('ITEM: BOX BOUNDS '+p_string+p_string+p_string+'\n')
        f.write('{} {}\n'.format(x_min, x_max))
        f.write('{} {}\n'.format(y_min, y_max))
        f.write('{} {}\n'.format(z_min, z_max))
        f.write('ITEM: ATOMS id type xu yu zu \n')
        for i_type in range(n_type):
            for i_p, coord in enumerate(c[i_type]):
                p_id+=1
                f.write('{:d} {:d} {} {} {}\n'.format(p_id, i_type, coord[0], coord[1], coord[2],))

# Unit cells
def unitcell_TR1(origin, orientation, ratio_ca=np.sqrt((1+np.sqrt(3)/2)**2+0.5**2)):
    '''
    Generate triangular unit cell
    origin: origin of unitcell, 3*1 array
    orientation: orientation of unitcell, 
                 represented by polar angle of the edge, float
    
    returns: cTR1 = [cTR1_wht,cTR1_blu,cTR1_ylw], lists of coordinates of the 3 types of particles
    '''
    e_SQ = np.array([[np.cos(x),np.sin(x)] for x in [0,np.pi/2]])
    e_TR = np.array([[np.cos(x),np.sin(x)] for x in [0,np.pi/3]])
    l_c = 1*np.sqrt((1+np.sqrt(3)/2)**2+0.5**2)/ratio_ca
    l_a = 1
        
    R = np.array([[np.cos(orientation),-np.sin(orientation),0],
                  [np.sin(orientation), np.cos(orientation),0],
                  [0,0,1]])

    # TR1
    # white sites
    cTR_wht1 = []
    for z in [1/4,3/4]:
        f_edge_w = np.array([[0,0],[0,1],[1,0]])
        c_xy = np.array([e_TR.T@f for f in f_edge_w])*l_a
        c_z = np.ones([len(f_edge_w),1])*l_c*z
        c = np.hstack((c_xy,c_z))
        cTR_wht1.append(c)
    cTR_wht1 = np.array(cTR_wht1)
    cTR_wht1 = cTR_wht1.reshape([2*len(f_edge_w),3])

    # blue sites
    cTR_blu1 = []
    for z in [0,1]:
        f_edge_b = np.array([[0,0.5],[0.5,0],[0.5,0.5]])
        c_xy = np.array([e_TR.T@f for f in f_edge_b])*l_a
        c_z = np.ones([len(f_edge_b),1])*l_c*z
        c = np.hstack((c_xy,c_z))
        cTR_blu1.append(c)
    cTR_blu1 = np.array(cTR_blu1)
    cTR_blu1 = cTR_blu1.reshape([2*len(f_edge_b),3])

    # yellow sites
    cTR_ylw1 = []
    for z in [1/2]:
        f_edge_y = (np.array([[0,0.5]])+np.array([[0.5,0]])+np.array([[0.5,0.5]]))/3
        c_xy = np.array([e_TR.T@f for f in f_edge_y])*l_a
        c_z = np.ones([len(f_edge_y),1])*l_c*z
        c = np.hstack((c_xy,c_z))
        cTR_ylw1.append(c)
    cTR_ylw1 = np.array(cTR_ylw1)
    cTR_ylw1 = cTR_ylw1.reshape([1*len(f_edge_y),3])
    
    cTR_wht1 = np.array([R@c for c in cTR_wht1])+origin
    cTR_blu1 = np.array([R@c for c in cTR_blu1])+origin
    cTR_ylw1 = np.array([R@c for c in cTR_ylw1])+origin 
    
    cTR1 = [cTR_wht1,cTR_blu1,cTR_ylw1]
    
    return cTR1

def unitcell_TR2(origin, orientation,ratio_ca=np.sqrt((1+np.sqrt(3)/2)**2+0.5**2)):
    '''
    Generate triangular unit cell
    origin: origin of unitcell, 3*1 array
    orientation: orientation of unitcell, 
                 represented by polar angle of the edge, float
    
    returns: cTR1 = [cTR1_wht,cTR1_blu,cTR1_ylw], lists of coordinates of the 3 types of particles
    '''
    e_SQ = np.array([[np.cos(x),np.sin(x)] for x in [0,np.pi/2]])
    e_TR = np.array([[np.cos(x),np.sin(x)] for x in [0,np.pi/3]])
    l_c = 1*np.sqrt((1+np.sqrt(3)/2)**2+0.5**2)/ratio_ca
    l_a = 1
        
    R = np.array([[np.cos(orientation),-np.sin(orientation),0],
                  [np.sin(orientation), np.cos(orientation),0],
                  [0,0,1]])

    # TR2
    # white sites
    cTR_wht2 = []
    for z in [1/4,3/4]:
        f_edge_w = np.array([[0,0],[0,1],[1,0]])
        c_xy = np.array([e_TR.T@f for f in f_edge_w])*l_a
        c_z = np.ones([len(f_edge_w),1])*l_c*z
        c = np.hstack((c_xy,c_z))
        cTR_wht2.append(c)
    cTR_wht2 = np.array(cTR_wht2)
    cTR_wht2 = cTR_wht2.reshape([2*len(f_edge_w),3])

    # blue sites
    cTR_blu2 = []
    for z in [0,1]:
        f_edge_b = (np.array([[0,0.5]])+np.array([[0.5,0]])+np.array([[0.5,0.5]]))/3
        c_xy = np.array([e_TR.T@f for f in f_edge_b])*l_a
        c_z = np.ones([len(f_edge_b),1])*l_c*z
        c = np.hstack((c_xy,c_z))
        cTR_blu2.append(c)
    cTR_blu2 = np.array(cTR_blu2)
    cTR_blu2 = cTR_blu2.reshape([2*len(f_edge_b),3])

    # yellow sites
    cTR_ylw2 = []
    for z in [1/2]:
        f_edge_y = np.array([[0,0.5],[0.5,0],[0.5,0.5]])
        c_xy = np.array([e_TR.T@f for f in f_edge_y])*l_a
        c_z = np.ones([len(f_edge_y),1])*l_c*z
        c = np.hstack((c_xy,c_z))
        cTR_ylw2.append(c)
    cTR_ylw2 = np.array(cTR_ylw2)
    cTR_ylw2 = cTR_ylw2.reshape([1*len(f_edge_y),3])
    
    cTR_wht2 = np.array([R@c for c in cTR_wht2])+origin
    cTR_blu2 = np.array([R@c for c in cTR_blu2])+origin
    cTR_ylw2 = np.array([R@c for c in cTR_ylw2])+origin
    
    cTR2 = [cTR_wht2,cTR_blu2,cTR_ylw2]
    
    return cTR2

def unitcell_SQ(origin, orientation, ratio_ca=np.sqrt((1+np.sqrt(3)/2)**2+0.5**2)):
    '''
    Generate rhombus unit cell
    origin: origin of unitcell, 3*1 array
    orientation: orientation of unitcell, 
                 represented by polar angle of the long diagonl, float
    
    returns: cH = [cH_wht,cH_blu,cH_ylw], lists of coordinates of the 3 types of particles
    '''
    e_SQ = np.array([[np.cos(x),np.sin(x)] for x in [0,np.pi/2]])
    e_TR = np.array([[np.cos(x),np.sin(x)] for x in [0,np.pi/3]])
    l_c = 1*np.sqrt((1+np.sqrt(3)/2)**2+0.5**2)/ratio_ca
    l_a = 1
        
    R = np.array([[np.cos(orientation),-np.sin(orientation),0],
                  [np.sin(orientation), np.cos(orientation),0],
                  [0,0,1]])

    # place atoms
    # white sites
    cSQ_wht = []
    for z in [1/4,3/4]:
        f_edge_w = np.array([[0,0],[0,1],[1,0],[1,1]])
        c_xy = np.array([e_SQ.T@f for f in f_edge_w])*l_a
        c_z = np.ones([len(f_edge_w),1])*l_c*z
        c = np.hstack((c_xy,c_z))
        cSQ_wht.append(c)
    cSQ_wht = np.array(cSQ_wht)
    cSQ_wht = cSQ_wht.reshape([2*len(f_edge_w),3])

    # blue sites
    cSQ_blu = []
    for z in [0,1]:
        f_edge_b = np.array([[0.5,0],[0.25,0.5],[0.75,0.5],[0.5,1]])
        c_xy = np.array([e_SQ.T@f for f in f_edge_b])*l_a
        c_z = np.ones([len(f_edge_b),1])*l_c*z
        c = np.hstack((c_xy,c_z))
        cSQ_blu.append(c)
    cSQ_blu = np.array(cSQ_blu)
    cSQ_blu = cSQ_blu.reshape([2*len(f_edge_b),3])

    # yellow sites
    cSQ_ylw = []
    for z in [1/2]:
        f_edge_y = np.array([[0,0.5],[0.5,0.25],[0.5,0.75],[1,0.5]])
        c_xy = np.array([e_SQ.T@f for f in f_edge_y])*l_a
        c_z = np.ones([len(f_edge_y),1])*l_c*z
        c = np.hstack((c_xy,c_z))
        cSQ_ylw.append(c)
    cSQ_ylw = np.array(cSQ_ylw)
    cSQ_ylw = cSQ_ylw.reshape([1*len(f_edge_y),3])
    
    cSQ_wht = np.array([R@c for c in cSQ_wht])+origin
    cSQ_blu = np.array([R@c for c in cSQ_blu])+origin
    cSQ_ylw = np.array([R@c for c in cSQ_ylw])+origin
    
    cSQ = [cSQ_wht,cSQ_blu,cSQ_ylw]
    
    return cSQ

# operations on unit cells
def uniq_coords(coordinates,scale=50):
    id_coords = (np.round(coordinates*scale))
    id_coords_unique = (np.unique(id_coords,axis=0,return_index=True)[1])
    id_coords_unique.sort()
    coordinates_unique = coordinates[id_coords_unique,:]
    return coordinates_unique

def merge_coords(function,iterables):
    # Generate coordinates with the generating function according to the iterables 
    c_m_wht, c_m_blu, c_m_ylw = [np.vstack([function(i)[clr] for i in iterables]) for clr in range(3)]
    c_m_wht = uniq_coords(c_m_wht)
    c_m_blu = uniq_coords(c_m_blu)
    c_m_ylw = uniq_coords(c_m_ylw)
    return c_m_wht, c_m_blu, c_m_ylw

def stack_coords(coords):
    c_s_wht, c_s_blu, c_s_ylw = [np.vstack([c[clr] for c in coords]) for clr in range(3)]
    c_s_wht = uniq_coords(c_s_wht)
    c_s_blu = uniq_coords(c_s_blu)
    c_s_ylw = uniq_coords(c_s_ylw)
    return c_s_wht, c_s_blu, c_s_ylw

def shift_coords(coords,shift=np.array([0,0,0])):
    c_s_wht, c_s_blu, c_s_ylw = [coords[clr]+shift for clr in range(3)]
    return c_s_wht, c_s_blu, c_s_ylw

def c_sigma_unit(origin,Ratio_ca=1.9,PBC=False):
    c_SQ1 = unitcell_SQ([0,0,0],0.0,ratio_ca=Ratio_ca)
    c_SQ2 = unitcell_SQ([0,0,0]+np.array([1+np.sqrt(3)/2,0.5,0]),np.pi/3,ratio_ca=Ratio_ca)
    c_TR1_1 = unitcell_TR1([0,0,0]+np.array([1,1,0]),2/3*np.pi,ratio_ca=Ratio_ca)
    c_TR1_2 = unitcell_TR1([0,0,0]+np.array([1,1,0]),1/3*np.pi,ratio_ca=Ratio_ca)
    c_TR2_1 = unitcell_TR2([0,0,0]+np.array([1,1,0]),-np.pi/2,ratio_ca=Ratio_ca)
    c_TR2_2 = unitcell_TR2([0,0,0]+np.array([1,0,0]),-np.pi/6,ratio_ca=Ratio_ca)
    
    c_wht, c_blu, c_ylw = stack_coords([c_SQ1,c_SQ2,c_TR1_1,c_TR1_2,c_TR2_1,c_TR2_2])
    
    R_c = (1+np.sqrt(3)/2)/np.sqrt((1+np.sqrt(3)/2)**2+(0.5)**2)
    R_s = 0.5/np.sqrt((1+np.sqrt(3)/2)**2+(0.5)**2)
    R = np.array([[R_c,-R_s,0],
                  [R_s,R_c,0],
                  [0,0,1]])
    c_wht = np.array([R@c for c in c_wht])
    c_blu = np.array([R@c for c in c_blu])
    c_ylw = np.array([R@c for c in c_ylw])
    
    if PBC:
        sigma = 1e-6
        i_c_wht = (c_wht[:,0]>sigma) & (c_wht[:,1]<np.sqrt((1+np.sqrt(3)/2)**2+(0.5)**2)-sigma)
        c_wht = c_wht[i_c_wht]
        i_c_blu = (c_blu[:,0]>sigma) & (c_blu[:,1]<np.sqrt((1+np.sqrt(3)/2)**2+(0.5)**2)-sigma) & (c_blu[:,2]>sigma)
        c_blu = c_blu[i_c_blu]
        i_c_ylw = (c_ylw[:,0]>sigma) & (c_ylw[:,1]<np.sqrt((1+np.sqrt(3)/2)**2+(0.5)**2)-sigma)
        c_ylw = c_ylw[i_c_ylw]
        
    c_wht = c_wht + origin
    c_blu = c_blu + origin
    c_ylw = c_ylw + origin
    
    return c_wht, c_blu, c_ylw

def c_sigma(nx,ny,Ratio_ca=1.9):
    shift_x = np.array([np.sqrt((1+np.sqrt(3)/2)**2+0.5**2),0,0])
    shift_y = np.array([0,np.sqrt((1+np.sqrt(3)/2)**2+0.5**2),0])
    c_unit_cells = []
    for i in range(nx):
        for j in range(ny):
            origin_ij = shift_x*i+shift_y*j
            c_unit_cells.append(c_sigma_unit(origin_ij,Ratio_ca,PBC=True))
    c_wht, c_blu, c_ylw = stack_coords(c_unit_cells)
    
    return c_wht, c_blu, c_ylw


#----------------------------------------------------
# Bergman Acta Cryst. (1954). 7, 857 
# Unit cells
def unitcell_TR1_B(origin, orientation, ratio_ca=np.sqrt((1+np.sqrt(3)/2)**2+0.5**2)):
    '''
    Generate triangular unit cell
    origin: origin of unitcell, 3*1 array
    orientation: orientation of unitcell, 
                 represented by polar angle of the edge, float
    
    returns: cTR1 = [cTR1_wht,cTR1_blu,cTR1_ylw], lists of coordinates of the 3 types of particles
    '''
    e_SQ = np.array([[np.cos(x),np.sin(x)] for x in [0,np.pi/2]])
    e_TR = np.array([[np.cos(x),np.sin(x)] for x in [0,np.pi/3]])
    l_c = 1*np.sqrt((1+np.sqrt(3)/2)**2+0.5**2)/ratio_ca
    l_a = 1
        
    R = np.array([[np.cos(orientation),-np.sin(orientation),0],
                  [np.sin(orientation), np.cos(orientation),0],
                  [0,0,1]])

    # TR1
    R_c = (1+np.sqrt(3)/2)/np.sqrt((1+np.sqrt(3)/2)**2+(0.5)**2)
    R_s = 0.5/np.sqrt((1+np.sqrt(3)/2)**2+(0.5)**2)
    R_l = np.array([[R_c,-R_s],
                  [R_s,R_c]])
    v_C = R_l.T@np.array([7/15-11/60,2/15-11/60])*np.sqrt((1+np.sqrt(3)/2)**2+(0.5)**2)
    v_D = R_l.T@np.array([11/15-11/60,1/15-11/60])*np.sqrt((1+np.sqrt(3)/2)**2+(0.5)**2)

    R_T = np.array([[np.cos(np.pi*2/3),-np.sin(np.pi*2/3)],
                    [np.sin(np.pi*2/3),np.cos(np.pi*2/3)]])
    f_O = np.array([0.5,np.sqrt(3)/6])
    f_0 = np.array([0.5,0])
    f_1 = R_T@(f_0-f_O)+f_O
    f_2 = R_T@(f_1-f_O)+f_O
    # white sites
    cTR_wht1 = []
    for z in [1/4,3/4]:
        f_edge_w = np.array([[0,0],[0,1],[1,0]])
        c_xy = np.array([e_TR.T@f for f in f_edge_w])*l_a
        c_z = np.ones([len(f_edge_w),1])*l_c*z
        c = np.hstack((c_xy,c_z))
        cTR_wht1.append(c)
    cTR_wht1 = np.array(cTR_wht1)
    cTR_wht1 = cTR_wht1.reshape([2*len(f_edge_w),3])

    # blue sites
    cTR_blu1 = []
    for z in [0,1]:
        f_edge_b = np.array([f_0,f_1,f_2])
        c_xy = np.array([f for f in f_edge_b])*l_a
        c_z = np.ones([len(f_edge_b),1])*l_c*z
        c = np.hstack((c_xy,c_z))
        cTR_blu1.append(c)
    cTR_blu1 = np.array(cTR_blu1)
    cTR_blu1 = cTR_blu1.reshape([2*len(f_edge_b),3])

    # yellow sites
    cTR_ylw1 = []
    for z in [1/2]:
        f_edge_y = (np.array([[0,0.5]])+np.array([[0.5,0]])+np.array([[0.5,0.5]]))/3
        c_xy = np.array([e_TR.T@f for f in f_edge_y])*l_a
        c_z = np.ones([len(f_edge_y),1])*l_c*z
        c = np.hstack((c_xy,c_z))
        cTR_ylw1.append(c)
    cTR_ylw1 = np.array(cTR_ylw1)
    cTR_ylw1 = cTR_ylw1.reshape([1*len(f_edge_y),3])
    
    cTR_wht1 = np.array([R@c for c in cTR_wht1])+origin
    cTR_blu1 = np.array([R@c for c in cTR_blu1])+origin
    cTR_ylw1 = np.array([R@c for c in cTR_ylw1])+origin 
    
    cTR1 = [cTR_wht1,cTR_blu1,cTR_ylw1]
    
    return cTR1

def unitcell_TR2_B(origin, orientation,ratio_ca=np.sqrt((1+np.sqrt(3)/2)**2+0.5**2)):
    '''
    Generate triangular unit cell
    origin: origin of unitcell, 3*1 array
    orientation: orientation of unitcell, 
                 represented by polar angle of the edge, float
    
    returns: cTR1 = [cTR1_wht,cTR1_blu,cTR1_ylw], lists of coordinates of the 3 types of particles
    '''
    e_SQ = np.array([[np.cos(x),np.sin(x)] for x in [0,np.pi/2]])
    e_TR = np.array([[np.cos(x),np.sin(x)] for x in [0,np.pi/3]])
    l_c = 1*np.sqrt((1+np.sqrt(3)/2)**2+0.5**2)/ratio_ca
    l_a = 1
        
    R = np.array([[np.cos(orientation),-np.sin(orientation),0],
                  [np.sin(orientation), np.cos(orientation),0],
                  [0,0,1]])

    # TR2
    R_c = (1+np.sqrt(3)/2)/np.sqrt((1+np.sqrt(3)/2)**2+(0.5)**2)
    R_s = 0.5/np.sqrt((1+np.sqrt(3)/2)**2+(0.5)**2)
    R_l = np.array([[R_c,-R_s],
                  [R_s,R_c]])
    v_C = R_l.T@np.array([7/15-11/60,2/15-11/60])*np.sqrt((1+np.sqrt(3)/2)**2+(0.5)**2)
    v_D = R_l.T@np.array([11/15-11/60,1/15-11/60])*np.sqrt((1+np.sqrt(3)/2)**2+(0.5)**2)

    R_T = np.array([[np.cos(np.pi*2/3),-np.sin(np.pi*2/3)],
                    [np.sin(np.pi*2/3),np.cos(np.pi*2/3)]])
    f_O = np.array([0.5,np.sqrt(3)/6])
    f_0 = np.array([0.5,0])
    f_1 = R_T@(f_0-f_O)+f_O
    f_2 = R_T@(f_1-f_O)+f_O
    # white sites
    cTR_wht2 = []
    for z in [1/4,3/4]:
        f_edge_w = np.array([[0,0],[0,1],[1,0]])
        c_xy = np.array([e_TR.T@f for f in f_edge_w])*l_a
        c_z = np.ones([len(f_edge_w),1])*l_c*z
        c = np.hstack((c_xy,c_z))
        cTR_wht2.append(c)
    cTR_wht2 = np.array(cTR_wht2)
    cTR_wht2 = cTR_wht2.reshape([2*len(f_edge_w),3])

    # blue sites
    cTR_blu2 = []
    for z in [0,1]:
        f_edge_b = (np.array([[0,0.5]])+np.array([[0.5,0]])+np.array([[0.5,0.5]]))/3
        c_xy = np.array([e_TR.T@f for f in f_edge_b])*l_a
        c_z = np.ones([len(f_edge_b),1])*l_c*z
        c = np.hstack((c_xy,c_z))
        cTR_blu2.append(c)
    cTR_blu2 = np.array(cTR_blu2)
    cTR_blu2 = cTR_blu2.reshape([2*len(f_edge_b),3])

    # yellow sites
    cTR_ylw2 = []
    for z in [1/2]:
        f_edge_y = np.array([f_0,f_1,f_2])
        c_xy = np.array([f for f in f_edge_y])*l_a
        c_z = np.ones([len(f_edge_y),1])*l_c*z
        c = np.hstack((c_xy,c_z))
        cTR_ylw2.append(c)
    cTR_ylw2 = np.array(cTR_ylw2)
    cTR_ylw2 = cTR_ylw2.reshape([1*len(f_edge_y),3])

    cTR_wht2 = np.array([R@c for c in cTR_wht2])+origin
    cTR_blu2 = np.array([R@c for c in cTR_blu2])+origin
    cTR_ylw2 = np.array([R@c for c in cTR_ylw2])+origin

    cTR2 = [cTR_wht2,cTR_blu2,cTR_ylw2]
    
    
    return cTR2

def unitcell_SQ_B(origin, orientation, ratio_ca=np.sqrt((1+np.sqrt(3)/2)**2+0.5**2)):
    '''
    Generate rhombus unit cell
    origin: origin of unitcell, 3*1 array
    orientation: orientation of unitcell, 
                 represented by polar angle of the long diagonl, float
    
    returns: cH = [cH_wht,cH_blu,cH_ylw], lists of coordinates of the 3 types of particles
    '''
    e_SQ = np.array([[np.cos(x),np.sin(x)] for x in [0,np.pi/2]])
    e_TR = np.array([[np.cos(x),np.sin(x)] for x in [0,np.pi/3]])
    l_c = 1*np.sqrt((1+np.sqrt(3)/2)**2+0.5**2)/ratio_ca
    l_a = 1
        
    R = np.array([[np.cos(orientation),-np.sin(orientation),0],
                  [np.sin(orientation), np.cos(orientation),0],
                  [0,0,1]])

    # place atoms
    R_c = (1+np.sqrt(3)/2)/np.sqrt((1+np.sqrt(3)/2)**2+(0.5)**2)
    R_s = 0.5/np.sqrt((1+np.sqrt(3)/2)**2+(0.5)**2)
    R_l = np.array([[R_c,-R_s],
                  [R_s,R_c]])
    v_C = R_l.T@np.array([7/15-11/60,2/15-11/60])*np.sqrt((1+np.sqrt(3)/2)**2+(0.5)**2)
    v_D = R_l.T@np.array([11/15-11/60,1/15-11/60])*np.sqrt((1+np.sqrt(3)/2)**2+(0.5)**2)
    # white sites
    cSQ_wht = []
    for z in [1/4,3/4]:
        f_edge_w = np.array([[0,0],[0,1],[1,0],[1,1]])
        c_xy = np.array([e_SQ.T@f for f in f_edge_w])*l_a
        c_z = np.ones([len(f_edge_w),1])*l_c*z
        c = np.hstack((c_xy,c_z))
        cSQ_wht.append(c)
    cSQ_wht = np.array(cSQ_wht)
    cSQ_wht = cSQ_wht.reshape([2*len(f_edge_w),3])

    # blue sites
    cSQ_blu = []
    for z in [0,1]:
        f_edge_b = np.array([[0.5,0],[-v_C[1],0.5],[1+v_C[1],0.5],[0.5,1]])
        c_xy = np.array([e_SQ.T@f for f in f_edge_b])*l_a
        c_z = np.ones([len(f_edge_b),1])*l_c*z
        c = np.hstack((c_xy,c_z))
        cSQ_blu.append(c)
    cSQ_blu = np.array(cSQ_blu)
    cSQ_blu = cSQ_blu.reshape([2*len(f_edge_b),3])

    # yellow sites
    cSQ_ylw = []
    for z in [1/2]:
        f_edge_y = np.array([[0,0.5],[0.5,-v_C[1]],[0.5,1+v_C[1]],[1,0.5]])
        c_xy = np.array([e_SQ.T@f for f in f_edge_y])*l_a
        c_z = np.ones([len(f_edge_y),1])*l_c*z
        c = np.hstack((c_xy,c_z))
        cSQ_ylw.append(c)
    cSQ_ylw = np.array(cSQ_ylw)
    cSQ_ylw = cSQ_ylw.reshape([1*len(f_edge_y),3])
    
    cSQ_wht = np.array([R@c for c in cSQ_wht])+origin
    cSQ_blu = np.array([R@c for c in cSQ_blu])+origin
    cSQ_ylw = np.array([R@c for c in cSQ_ylw])+origin
    
    cSQ = [cSQ_wht,cSQ_blu,cSQ_ylw]
    
    return cSQ

def c_sigma_unit_B(origin,Ratio_ca=1.9,PBC=False):
    c_SQ1 = unitcell_SQ_B([0,0,0],0.0,ratio_ca=Ratio_ca)
    c_SQ2 = unitcell_SQ_B([0,0,0]+np.array([1+np.sqrt(3)/2,0.5,0]),np.pi/3,ratio_ca=Ratio_ca)
    c_TR1_1 = unitcell_TR1_B([0,0,0]+np.array([1,1,0]),2/3*np.pi,ratio_ca=Ratio_ca)
    c_TR1_2 = unitcell_TR1_B([0,0,0]+np.array([1,1,0]),1/3*np.pi,ratio_ca=Ratio_ca)
    c_TR2_1 = unitcell_TR2_B([0,0,0]+np.array([1,1,0]),-np.pi/2,ratio_ca=Ratio_ca)
    c_TR2_2 = unitcell_TR2_B([0,0,0]+np.array([1,0,0]),-np.pi/6,ratio_ca=Ratio_ca)
    
    c_wht, c_blu, c_ylw = stack_coords([c_SQ1,c_SQ2,c_TR1_1,c_TR1_2,c_TR2_1,c_TR2_2])
    
    R_c = (1+np.sqrt(3)/2)/np.sqrt((1+np.sqrt(3)/2)**2+(0.5)**2)
    R_s = 0.5/np.sqrt((1+np.sqrt(3)/2)**2+(0.5)**2)
    R = np.array([[R_c,-R_s,0],
                  [R_s,R_c,0],
                  [0,0,1]])
    c_wht = np.array([R@c for c in c_wht])
    c_blu = np.array([R@c for c in c_blu])
    c_ylw = np.array([R@c for c in c_ylw])
    
    if PBC:
        sigma = 1e-6
        i_c_wht = (c_wht[:,0]>sigma) & (c_wht[:,1]<np.sqrt((1+np.sqrt(3)/2)**2+(0.5)**2)-sigma)
        c_wht = c_wht[i_c_wht]
        i_c_blu = (c_blu[:,0]>sigma) & (c_blu[:,1]<np.sqrt((1+np.sqrt(3)/2)**2+(0.5)**2)-sigma) & (c_blu[:,2]>sigma)
        c_blu = c_blu[i_c_blu]
        i_c_ylw = (c_ylw[:,0]>sigma) & (c_ylw[:,1]<np.sqrt((1+np.sqrt(3)/2)**2+(0.5)**2)-sigma)
        c_ylw = c_ylw[i_c_ylw]
        
    c_wht = c_wht + origin
    c_blu = c_blu + origin
    c_ylw = c_ylw + origin
    
    return c_wht, c_blu, c_ylw

def c_sigma_B(nx,ny,Ratio_ca=1.9):
    shift_x = np.array([np.sqrt((1+np.sqrt(3)/2)**2+0.5**2),0,0])
    shift_y = np.array([0,np.sqrt((1+np.sqrt(3)/2)**2+0.5**2),0])
    c_unit_cells = []
    for i in range(nx):
        for j in range(ny):
            origin_ij = shift_x*i+shift_y*j
            c_unit_cells.append(c_sigma_unit_B(origin_ij,Ratio_ca,PBC=True))
    c_wht, c_blu, c_ylw = stack_coords(c_unit_cells)
    
    return c_wht, c_blu, c_ylw

# -------------------------------------------------
# FK A15 phase

def c_A15_unit(origin,Ratio_ca=1.9,PBC=False):
    c_SQ1 = unitcell_SQ([0,0,0],0.0,ratio_ca=Ratio_ca)
    
    c_wht, c_blu, c_ylw = stack_coords([c_SQ1])
    
    R_c = 1
    R_s = 0
    R = np.array([[R_c,-R_s,0],
                  [R_s,R_c,0],
                  [0,0,1]])
    c_wht = np.array([R@c for c in c_wht])
    c_blu = np.array([R@c for c in c_blu])
    c_ylw = np.array([R@c for c in c_ylw])
    
    if PBC:
        sigma = 1e-6
        i_c_wht = (c_wht[:,0]>sigma) & (c_wht[:,1]<1-sigma)
        c_wht = c_wht[i_c_wht]
        i_c_blu = (c_blu[:,0]>sigma) & (c_blu[:,1]<1-sigma) & (c_blu[:,2]>sigma)
        c_blu = c_blu[i_c_blu]
        i_c_ylw = (c_ylw[:,0]>sigma) & (c_ylw[:,1]<1-sigma)
        c_ylw = c_ylw[i_c_ylw]
        
    c_wht = c_wht + origin
    c_blu = c_blu + origin
    c_ylw = c_ylw + origin
    
    return c_wht, c_blu, c_ylw

def c_A15(nx,ny,Ratio_ca=1.9):
    shift_x = np.array([1,0,0])
    shift_y = np.array([0,1,0])
    c_unit_cells = []
    for i in range(nx):
        for j in range(ny):
            origin_ij = shift_x*i+shift_y*j
            c_unit_cells.append(c_A15_unit(origin_ij,Ratio_ca,PBC=True))
    c_wht, c_blu, c_ylw = stack_coords(c_unit_cells)
    
    return c_wht, c_blu, c_ylw

# -------------------------------------------------
# FK Z phase

def c_Z_unit(origin,Ratio_ca=1.9,PBC=False):
    c_TR1_1 = unitcell_TR1([0,np.sqrt(3)/2,0],0,ratio_ca=Ratio_ca)
    c_TR1_2 = unitcell_TR1([0,np.sqrt(3)/2,0]+np.array([1,0,0]),1/3*np.pi,ratio_ca=Ratio_ca)
    c_TR1_3 = unitcell_TR1([0,np.sqrt(3)/2,0],1/3*np.pi,ratio_ca=Ratio_ca)
    c_TR1_4 = unitcell_TR1([0,np.sqrt(3)/2,0],-1/3*np.pi,ratio_ca=Ratio_ca)
    c_TR1_5 = unitcell_TR1([0,np.sqrt(3)/2,0],-2/3*np.pi,ratio_ca=Ratio_ca)
    c_TR1_6 = unitcell_TR1([0,np.sqrt(3)/2,0]+np.array([1,0,0]),-2/3*np.pi,ratio_ca=Ratio_ca)
    
    c_wht, c_blu, c_ylw = stack_coords([c_TR1_1,c_TR1_2,c_TR1_3,c_TR1_4,c_TR1_5,c_TR1_6])
    
    R_c = 1
    R_s = 0
    R = np.array([[R_c,-R_s,0],
                  [R_s,R_c,0],
                  [0,0,1]])
    c_wht = np.array([R@c for c in c_wht])
    c_blu = np.array([R@c for c in c_blu])
    c_ylw = np.array([R@c for c in c_ylw])
    
    if PBC:
        sigma = 1e-6
        i_c_wht = (c_wht[:,0]>sigma) & (c_wht[:,0]<1+sigma) & (c_wht[:,1]<np.sqrt(3)+sigma) & (c_wht[:,1]>sigma)
        c_wht = c_wht[i_c_wht]
        i_c_blu = (c_blu[:,0]>sigma) & (c_blu[:,0]<1+sigma) & (c_blu[:,1]<np.sqrt(3)+sigma) & (c_blu[:,1]>sigma) & (c_blu[:,2]>sigma)
        c_blu = c_blu[i_c_blu]
        i_c_ylw = (c_ylw[:,0]>sigma) & (c_ylw[:,0]<1+sigma) & (c_ylw[:,1]<np.sqrt(3)+sigma) & (c_ylw[:,1]>sigma)
        c_ylw = c_ylw[i_c_ylw]
        
    c_wht = c_wht + origin
    c_blu = c_blu + origin
    c_ylw = c_ylw + origin
    
    return c_wht, c_blu, c_ylw

def c_Z(nx,ny,Ratio_ca=1.9):
    shift_x = np.array([1,0,0])
    shift_y = np.array([0,1,0])*np.sqrt(3)
    c_unit_cells = []
    for i in range(nx):
        for j in range(ny):
            origin_ij = shift_x*i+shift_y*j
            c_unit_cells.append(c_Z_unit(origin_ij,Ratio_ca,PBC=True))
    c_wht, c_blu, c_ylw = stack_coords(c_unit_cells)
    
    return c_wht, c_blu, c_ylw

# =================================================
# tile
def tile_TR1(origin, orientation, ratio_ca=np.sqrt((1+np.sqrt(3)/2)**2+0.5**2)):
    '''
    Generate triangular unit cell
    origin: origin of unitcell, 3*1 array
    orientation: orientation of unitcell, 
                 represented by polar angle of the edge, float
    
    returns: c_tile_list , lists of coordinates of the triangular tile
    '''
    e_SQ = np.array([[np.cos(x),np.sin(x)] for x in [0,np.pi/2]])
    e_TR = np.array([[np.cos(x),np.sin(x)] for x in [0,np.pi/3]])
    l_c = 1*np.sqrt((1+np.sqrt(3)/2)**2+0.5**2)/ratio_ca
    l_a = 1
        
    R = np.array([[np.cos(orientation),-np.sin(orientation),0],
                  [np.sin(orientation), np.cos(orientation),0],
                  [0,0,1]])

    # TR1
    # white sites
    cTR_wht1 = []
    for z in [0]:
        f_edge_w = np.array([[0,0],[0,1],[1,0]])
        c_xy = np.array([e_TR.T@f for f in f_edge_w])*l_a
        c_z = np.ones([len(f_edge_w),1])*l_c*z
        c = np.hstack((c_xy,c_z))
        cTR_wht1.append(c)
    cTR_wht1 = np.array(cTR_wht1)
    cTR_wht1 = cTR_wht1.reshape([len(f_edge_w),3])
    
    cTR_wht1 = np.array([R@c + origin for c in cTR_wht1])
    
    c_tile_list = [cTR_wht1.tolist()]
    
    return c_tile_list

def tile_TR2(origin, orientation, ratio_ca=np.sqrt((1+np.sqrt(3)/2)**2+0.5**2)):
    '''
    Generate triangular unit cell
    origin: origin of unitcell, 3*1 array
    orientation: orientation of unitcell, 
                 represented by polar angle of the edge, float
    
    returns: c_tile_list , lists of coordinates of the triangular tile
    '''
    e_SQ = np.array([[np.cos(x),np.sin(x)] for x in [0,np.pi/2]])
    e_TR = np.array([[np.cos(x),np.sin(x)] for x in [0,np.pi/3]])
    l_c = 1*np.sqrt((1+np.sqrt(3)/2)**2+0.5**2)/ratio_ca
    l_a = 1
        
    R = np.array([[np.cos(orientation),-np.sin(orientation),0],
                  [np.sin(orientation), np.cos(orientation),0],
                  [0,0,1]])

    # TR1
    # white sites
    cTR_wht1 = []
    for z in [0]:
        f_edge_w = np.array([[0,0],[0,1],[1,0]])
        c_xy = np.array([e_TR.T@f for f in f_edge_w])*l_a
        c_z = np.ones([len(f_edge_w),1])*l_c*z
        c = np.hstack((c_xy,c_z))
        cTR_wht1.append(c)
    cTR_wht1 = np.array(cTR_wht1)
    cTR_wht1 = cTR_wht1.reshape([len(f_edge_w),3])
    
    cTR_wht1 = np.array([R@c + origin for c in cTR_wht1])
    
    c_tile_list = [cTR_wht1.tolist()]
    
    return c_tile_list

def tile_SQ(origin, orientation, ratio_ca=np.sqrt((1+np.sqrt(3)/2)**2+0.5**2)):
    '''
    Generate rhombus unit cell
    origin: origin of unitcell, 3*1 array
    orientation: orientation of unitcell, 
                 represented by polar angle of the long diagonl, float
    
    returns: c_tile_list , lists of coordinates of the triangular tile
    '''
    e_SQ = np.array([[np.cos(x),np.sin(x)] for x in [0,np.pi/2]])
    e_TR = np.array([[np.cos(x),np.sin(x)] for x in [0,np.pi/3]])
    l_c = 1*np.sqrt((1+np.sqrt(3)/2)**2+0.5**2)/ratio_ca
    l_a = 1
        
    R = np.array([[np.cos(orientation),-np.sin(orientation),0],
                  [np.sin(orientation), np.cos(orientation),0],
                  [0,0,1]])

    # place atoms
    # white sites
    cSQ_wht = []
    for z in [0]:
        f_edge_w = np.array([[0,0],[0,1],[1,1],[1,0]])
        c_xy = np.array([e_SQ.T@f for f in f_edge_w])*l_a
        c_z = np.ones([len(f_edge_w),1])*l_c*z
        c = np.hstack((c_xy,c_z))
        cSQ_wht.append(c)
    cSQ_wht = np.array(cSQ_wht)
    cSQ_wht = cSQ_wht.reshape([len(f_edge_w),3])
    
    cSQ_wht = np.array([R@c + origin for c in cSQ_wht])
    
    c_tile_list = [cSQ_wht.tolist()]
    
    return c_tile_list

def tile_sigma_unit(origin,Ratio_ca=1.9,PBC=False):
    c_SQ1 = tile_SQ([0,0,0],0.0,ratio_ca=Ratio_ca)
    c_SQ2 = tile_SQ([0,0,0]+np.array([1+np.sqrt(3)/2,0.5,0]),np.pi/3,ratio_ca=Ratio_ca)
    c_TR1_1 = tile_TR1([0,0,0]+np.array([1,1,0]),2/3*np.pi,ratio_ca=Ratio_ca)
    c_TR1_2 = tile_TR1([0,0,0]+np.array([1,1,0]),1/3*np.pi,ratio_ca=Ratio_ca)
    c_TR2_1 = tile_TR2([0,0,0]+np.array([1,1,0]),-np.pi/2,ratio_ca=Ratio_ca)
    c_TR2_2 = tile_TR2([0,0,0]+np.array([1,0,0]),-np.pi/6,ratio_ca=Ratio_ca)
    
    c_SQ = np.array(c_SQ1+c_SQ2)
    c_TR1 = np.array(c_TR1_1+c_TR1_2)
    c_TR2 = np.array(c_TR2_1+c_TR2_2)
    
    R_c = (1+np.sqrt(3)/2)/np.sqrt((1+np.sqrt(3)/2)**2+(0.5)**2)
    R_s = 0.5/np.sqrt((1+np.sqrt(3)/2)**2+(0.5)**2)
    R = np.array([[R_c,-R_s,0],
                  [R_s,R_c,0],
                  [0,0,1]])
    c_SQ = np.array([[R@c + origin for c in c_list] for c_list in c_SQ])
    c_TR1 = np.array([[R@c + origin for c in c_list] for c_list in c_TR1])
    c_TR2 = np.array([[R@c + origin for c in c_list] for c_list in c_TR2])
    

#     c_SQ = c_SQ
#     c_TR1 = c_TR1
#     c_TR2 = c_TR2
    
    return c_SQ.tolist(), c_TR1.tolist(), c_TR2.tolist()

def tile_sigma(nx,ny,Ratio_ca=1.9):
    shift_x = np.array([np.sqrt((1+np.sqrt(3)/2)**2+0.5**2),0,0])
    shift_y = np.array([0,np.sqrt((1+np.sqrt(3)/2)**2+0.5**2),0])
    tile_unit_cells = [[],[],[]]
    for i in range(nx):
        for j in range(ny):
            origin_ij = shift_x*i+shift_y*j
            for k in range(3):
                tile_sigma_unit_ij = tile_sigma_unit(origin_ij,Ratio_ca,PBC=True)
                tile_unit_cells[k]+=tile_sigma_unit_ij[k]
    
    return tile_unit_cells

def tile_A15_unit(origin,Ratio_ca=1.9,PBC=False):
    c_SQ = tile_SQ([0,0,0],0.0,ratio_ca=Ratio_ca)
    
    R_c = 1
    R_s = 0
    R = np.array([[R_c,-R_s,0],
                  [R_s,R_c,0],
                  [0,0,1]])
    c_SQ = np.array([[R@c + origin for c in c_list] for c_list in c_SQ])
    
    
    return c_SQ.tolist(), [], []

def tile_A15(nx,ny,Ratio_ca=1.9):
    shift_x = np.array([1,0,0])
    shift_y = np.array([0,1,0])
    tile_unit_cells = [[],[],[]]
    for i in range(nx):
        for j in range(ny):
            origin_ij = shift_x*i+shift_y*j
            for k in range(3):
                tile_unit_ij = tile_A15_unit(origin_ij,Ratio_ca,PBC=True)
                tile_unit_cells[k]+=tile_unit_ij[k]
                
    return tile_unit_cells

def tile_Z_unit(origin,Ratio_ca=1.9,PBC=False):
    c_TR1_1 = tile_TR1([0,np.sqrt(3)/2,0],0,ratio_ca=Ratio_ca)
    c_TR1_2 = tile_TR1([0,np.sqrt(3)/2,0]+np.array([1,0,0]),1/3*np.pi,ratio_ca=Ratio_ca)
    c_TR1_3 = tile_TR1([0,np.sqrt(3)/2,0],1/3*np.pi,ratio_ca=Ratio_ca)
    c_TR1_4 = tile_TR1([0,np.sqrt(3)/2,0],-1/3*np.pi,ratio_ca=Ratio_ca)
    c_TR1_5 = tile_TR1([0,np.sqrt(3)/2,0],-2/3*np.pi,ratio_ca=Ratio_ca)
    c_TR1_6 = tile_TR1([0,np.sqrt(3)/2,0]+np.array([1,0,0]),-2/3*np.pi,ratio_ca=Ratio_ca)
    
    c_TR1 = np.array(c_TR1_1+c_TR1_2+c_TR1_4+c_TR1_6)
    
    R_c = 1
    R_s = 0
    R = np.array([[R_c,-R_s,0],
                  [R_s,R_c,0],
                  [0,0,1]])
    
    c_TR1 = np.array([[R@c + origin for c in c_list] for c_list in c_TR1])
    
    return [], c_TR1.tolist(), []

def tile_Z(nx,ny,Ratio_ca=1.9):
    shift_x = np.array([1,0,0])
    shift_y = np.array([0,1,0])*np.sqrt(3)
    tile_unit_cells = [[],[],[]]
    for i in range(nx):
        for j in range(ny):
            origin_ij = shift_x*i+shift_y*j
            for k in range(3):
                tile_unit_ij = tile_Z_unit(origin_ij,Ratio_ca,PBC=True)
                tile_unit_cells[k]+=tile_unit_ij[k]
    
    return tile_unit_cells

# =====================================================
# Calculate S(Q)

def scatter_histo(c, qq, p_sub=1.0, n_bins=1000):
    """
    Calculate scattering function.

    Args:
        c: N by 3 particle trajectory
        qq: array
            wave vectors
        p_sub: amount of particles used to calculate S(Q)
        n_bins: amount of rdf histogram bins
    """

    N = c.shape[1]

    # two-point correlation
    n_list = int(N*p_sub)
    i_list = np.random.choice(np.arange(N), size=n_list)
    r_jk = c[:,i_list].T.reshape(n_list,1,3) - c[:,i_list].T.reshape(1,n_list,3)
    d_jk = np.sqrt(np.sum(r_jk**2,axis=2))

    d_max = np.max(d_jk)
    rr = np.linspace(d_max/n_bins,d_max,n_bins)

    index_r_jk = np.floor(d_jk/d_max*n_bins)
    np.fill_diagonal(index_r_jk,n_bins*2) # we are not calculating these pairs
    
    rho_r = np.zeros(n_bins)
    for i_r in range(n_bins):
        rho_r[i_r] = np.sum(index_r_jk==i_r)

    S_q = np.array([np.sum(rho_r*np.sin(rr*q)/(rr*q)) for q in qq])/N/2
    
    return S_q