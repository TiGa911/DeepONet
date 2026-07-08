import numpy as np
from scipy import interpolate
import geqdsk
#%%
def g_to_23(gfile, psi_norm, theta, psicut=0.995):
    """
    Calculte the flux surface mesh from gfile
    23 indicates the coordinate system defined in IMAS IDS
    equilibrium.time_slice.profiles_2d.grid_type
    """
# To be improved:
# 1. Near the axis, value of Jacobian is bad
    gg = geqdsk.load(gfile)
    #nbbbs    = gg['nbbbs']
    nw      = gg['nw']
    nh      = gg['nh']
    rdim    = gg['rdim']
    zdim    = gg['zdim']
    #rcentr  = gg['rcentr']
    rleft   = gg['rleft']
    zmid    = gg['zmid']
    rbbbs   = gg['bbbsrz'][:,0]  # R coordinates on separatrix 
    zbbbs   = gg['bbbsrz'][:,1]
    maxis_r = gg['rmaxis']
    maxis_z = gg['zmaxis']
    psib    = gg['sibry']
    psia    = gg['simag']
    psirz   = gg['psirz']
    fpol    = gg['fpol']
    #qpsi    = gg['qpsi']

    rr = np.linspace(rleft, rleft+rdim, nw)
    zz = np.linspace(zmid-zdim/2., zmid+zdim/2., nh)

    # Contruct an ellipse as initial plasma boudary
    bleft = np.min(rbbbs)
    btop  = np.max(zbbbs)
    bbot  = np.min(zbbbs)
    aa = (maxis_r - bleft) * 1.3
    bb = (btop - bbot)/2. * 1.3
    epoints = ellipse(aa, bb, theta)  
    epoints[0,:] = epoints[0,:] + maxis_r # Ellipse boundary points
    epoints[1,:] = epoints[1,:] + maxis_z

    # Method1
    ff = interpolate.RectBivariateSpline(rr, zz, psirz.T)
    # Method2, it works, extremely slow with s=0.
    #RR, ZZ = np.meshgrid(rr, zz)
    #tck = interpolate.bisplrep(RR, ZZ, psirz, s=1.e-6)
    # Method3, it works, but too slow. It is parallized.
    #RR, ZZ = np.meshgrid(rr, zz)
    #rbf = interpolate.Rbf( RR, ZZ, psirz, smooth=1.e-4 )    # Too slow

    # Real psi profile of gfile, the separatrix included, in Wb
    psi_g    = np.linspace(0.,1.,len(fpol)) * (psib - psia) + psia
    # Real psi on surfaces, after psicut, in Wb
    psi_surf = psi_norm * (psib - psia)*psicut + psia 

    ntheta = len(theta)
    npsi   = len(psi_surf)
    sep_r = np.zeros(ntheta)
    sep_z = np.zeros(ntheta)
    grid_r = np.zeros((npsi, ntheta))
    grid_z = np.zeros((npsi, ntheta))
    b_z   = np.zeros((npsi, ntheta))
    b_r   = np.zeros((npsi, ntheta))
    b_t   = np.zeros((npsi, ntheta))
    for nn in range(ntheta):
        # The vetor along the ray
        # First find the separatrix points
        r_t = np.linspace(maxis_r, epoints[0,nn], npsi)  # R(t)
        z_t = np.linspace(maxis_z, epoints[1,nn], npsi)
        psi_ray = np.zeros((npsi))
        for tt in range(npsi):             
            psi_ray[tt]  = ff(r_t[tt], z_t[tt])
        idt = np.argwhere(np.diff(np.sign(psi_ray - psib)) != 0)[0]
        ppp = np.linspace(psi_ray[idt-1][0], psi_ray[idt+1][0], npsi)
        ids = np.argwhere(np.diff(np.sign(ppp - psib)) != 0)
        sep_r[nn] = r_t[idt-1] + (r_t[idt+1] - r_t[idt-1]) * float(ids-1)/npsi
        sep_z[nn] = z_t[idt-1] + (z_t[idt+1] - z_t[idt-1]) * float(ids-1)/npsi

        # Then interpolate the surface points between axis and separatrix
        r_t = np.linspace(maxis_r, sep_r[nn], npsi)  # R(t)
        z_t = np.linspace(maxis_z, sep_z[nn], npsi)        
        psir_ray= np.zeros((npsi))   # dpsi/dR
        psiz_ray= np.zeros((npsi))   # dpsi/dz
        #psi_ray2 = np.zeros((npsi))
        for tt in range(npsi):             
            psi_ray[tt]  = ff(r_t[tt], z_t[tt])
            psir_ray[tt] = ff(r_t[tt], z_t[tt], dx=1) # dpsi/dR
            psiz_ray[tt] = ff(r_t[tt], z_t[tt], dy=1) # dpsi/dz
            #psi_ray2[tt] = interpolate.bisplev(r_t[tt], z_t[tt], tck)
        #psi_ray = rbf(r_t, z_t)
        # To avoid psi_surf[0]<psi_ray[0], that interp1d raises an error
        # psi_ray[0] = psi_surf[0]
        
        xxx    = interpolate.interp1d(psi_ray, r_t, kind='cubic', fill_value='extrapolate', assume_sorted='True')
        r_ray  = xxx(psi_surf)
        xxx    = interpolate.interp1d(psi_ray, z_t, kind='cubic', fill_value='extrapolate')
        z_ray  = xxx(psi_surf)
        xxx    = interpolate.interp1d(psi_ray, psiz_ray, kind='cubic', fill_value='extrapolate')
        b_r[:, nn]    = - xxx(psi_surf) /r_ray
        xxx    = interpolate.interp1d(psi_ray, psir_ray, kind='cubic', fill_value='extrapolate')
        b_z[:, nn]    = xxx(psi_surf) /r_ray
        xxx    = interpolate.interp1d(psi_g, fpol, kind='cubic')
        b_t[:, nn]    = xxx(psi_surf) /r_ray
        grid_r[:, nn] = r_ray
        grid_z[:, nn] = z_ray
    #
    grid_r[0,:] = maxis_r
    grid_z[0,:] = maxis_z

    b_theta = np.sqrt(b_r*b_r + b_z*b_z)

    xxx    = interpolate.RectBivariateSpline(psi_surf, theta, grid_r)
    drdpsi = xxx(psi_surf, theta, dx=1)
    drdthe = xxx(psi_surf, theta, dy=1)
    xxx    = interpolate.RectBivariateSpline(psi_surf, theta, grid_z)
    dzdpsi = xxx(psi_surf, theta, dx=1)
    dzdthe = xxx(psi_surf, theta, dy=1)    
    jacbn  = (drdpsi*dzdthe - drdthe*dzdpsi) * grid_r

    dvdpsi  = np.zeros(npsi)
    theta_c = theta[:-1] + np.diff(theta)
    dl      = np.zeros((npsi, ntheta-1))
    for ii in range(1,npsi):
        dr = np.diff(grid_r[ii,:])
        dz = np.diff(grid_z[ii,:])
        # Arc length
        dl[ii,:] = np.sqrt(dr*dr + dz*dz)
        xxx  = interpolate.interp1d(theta, b_theta[ii,:], kind='cubic')
        bp_c = xxx(theta_c)
        dvdpsi[ii] = np.sum(dl[ii,:]/bp_c) * 2. * np.pi
    dvdpsi[0] = dvdpsi[1] - (dvdpsi[2] - dvdpsi[1])

    vol  = np.zeros(npsi)
    dpsi = np.diff(psi_surf)
    for ii in range(1,npsi):
        vol[ii] = vol[ii-1] + 0.5*(dvdpsi[ii] + dvdpsi[ii-1]) * dpsi[ii-1]

    phi  = np.zeros(npsi)
    area = np.zeros(npsi)
    for ii in range(1,npsi):
        dphi  = 0.
        darea = 0.
        for jj in range(1, ntheta):
            x1 = grid_r[ii,jj]; x2 = grid_r[ii-1,jj]; x3 = grid_r[ii-1,jj-1]; x4 = grid_r[ii, jj-1]
            y1 = grid_z[ii,jj]; y2 = grid_z[ii-1,jj]; y3 = grid_z[ii-1,jj-1]; y4 = grid_z[ii, jj-1]
            ddarea = abs( 0.5 * ( x1*y2 + x2*y3  + x3*y4 + x4*y1 
                           - x2*y1 - x3*y2  - x4*y3 - x1*y4) )
            dphi   = dphi  + ddarea * 0.25 * (b_t[ii,jj] + b_t[ii, jj-1] + b_t[ii-1, jj] + b_t[ii-1, jj-1])
            darea  = darea + ddarea
        phi[ii]  = dphi + phi[ii-1]
        area[ii] = darea + area[ii-1]

    print('Normal end')
    return {
        'grid_r': grid_r,
        'grid_z': grid_z,
        'jacobian' : jacbn,
        'b_theta' : b_theta,
        'b_t'     : b_t,
        'drdpsi'  : drdpsi,
        'drdthe'  : drdthe,
        'dzdpsi'  : dzdpsi,
        'dzdthe'  : dzdthe,
        'dvdpsi'  : dvdpsi, # psi the the poloidal flux, not normalized
        'vol'     : vol,    # Plama volume
        'area'    : area,   # Cross section area
        'phi'     : phi,
        'dl'      : dl
    }

def ellipse(aa, bb, theta):
    """
    Find the points on a ellipse, for given theta
    """
    # first contruct a circle
    rmax = max(aa, bb)
    rmin = min(aa, bb) * 0.99
    xmax = rmax * np.cos(theta)  # circle points
    ymax = rmax * np.sin(theta)
    xmin = rmin * np.cos(theta)  # circle points
    ymin = rmin * np.sin(theta)
    
    # Along the ray, find the ellipse point
    nray = 101
    ntheta  = len(theta)
    epoints = np.zeros((2,ntheta))
    for ii in range(ntheta):
        xx = np.linspace(xmin[ii], xmax[ii], nray)
        yy = np.linspace(ymin[ii], ymax[ii], nray)
        zz = xx*xx/aa/aa + yy*yy/bb/bb
        ff = interpolate.interp1d(zz, xx, kind='cubic')
        gg = interpolate.interp1d(zz, yy, kind='cubic')
        epoints[0,ii] = ff(1.)
        epoints[1,ii] = gg(1.)
    return(epoints)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    psi_norm = np.linspace(0., 1., 81)
    theta    = np.linspace(0., 2.* np.pi, 127)
    grid = g_to_23('g038300.03900', psi_norm, theta)
    plt.plot(grid['grid_r'], grid['grid_z'])
    plt.plot(np.transpose(grid['grid_r']), np.transpose(grid['grid_z']))
    plt.axes().set_aspect('equal')
    plt.show()