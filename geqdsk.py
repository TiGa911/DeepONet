# -*- coding: utf-8 -*-
# To outline gfile from local disk or from the MDS+ server, output is a dict;
#  and write gfile to disk, the input is a dict.
# The names of the keys follow the description of geqdsk by Lao, except:
#       (rbbbs, zbbbs) -> bbbsrz
#       (rlim,  zlim)  -> limrz
# Originly written Yu Zhi, modified by Li GQ
#
# 2022.11.05
#  Add the function to outline geqdsk from MDS+ server of EAST

import numpy as np

__plugin_spec__ = {
    "name": "geqdsk",
    "filename_extension": "gfile",
    "filename_pattern": ["*.geqdsk", "*.gfile"]
}


def load(file, *args, **kwargs):
    """
    :param file: input file / file path
    :return: profile object
    """
    if type(file) is str:
        file = open(file, "r")

    # res = np.genfromtxt(file, dtype=np.dtype([
    #     ("head", 'S50'), ("idum", "i4"), ("nw", "i4"), ("nh", "i4")
    # ]))

    head = file.read(48)
    idum = int(file.read(4))
    nw = int(file.read(4))
    nh = int(file.read(4))
    file.readline()

    rdim = float(file.read(16))
    zdim = float(file.read(16))
    rcentr = float(file.read(16))
    rleft = float(file.read(16))
    zmid = float(file.read(16))
    file.readline()
    rmaxis = float(file.read(16))
    zmaxis = float(file.read(16))
    simag = float(file.read(16))
    sibry = float(file.read(16))
    bcentr = float(file.read(16))
    file.readline()
    current = float(file.read(16))
    simag = float(file.read(16))
    xdum = float(file.read(16))
    rmaxis = float(file.read(16))
    xdum = float(file.read(16))
    file.readline()

    zmaxis = float(file.read(16))
    xdum = float(file.read(16))
    sibry = float(file.read(16))
    xdum = float(file.read(16))
    xdum = float(file.read(16))
    file.readline()

    def _read_data(count, width=16):
        data = np.ndarray(shape=[count], dtype=float)

        for n in range(count):
            data[n] = float(file.read(width))
            if n >= count - 1 or ((n + 1) % 5 == 0):
                file.readline()
        return data

    #
    fpol = _read_data(nw)
    pres = _read_data(nw)
    ffprim = _read_data(nw)
    pprime = _read_data(nw)

    psirz = _read_data(nw * nh).reshape([nw, nh])

    qpsi = _read_data(nw)

    nbbbs = int(file.read(5))
    limitr = int(file.read(5))
    file.readline()

    bbbsrz = _read_data(nbbbs * 2).reshape([nbbbs, 2])
    limrz  = _read_data(limitr * 2).reshape([limitr, 2])
    file.close()
    r = np.linspace(rleft, rleft+rdim, nw)
    z = np.linspace(zmid-zdim/2., zmid+zdim/2., nh)
    return {
        "head": head,
        # "idum": idum,
        "nw": nw,
        "nh": nh,
        "rdim": rdim,
        "zdim": zdim,
        "rcentr": rcentr,
        "rleft": rleft,
        "zmid": zmid,
        "rmaxis": rmaxis,
        "zmaxis": zmaxis,
        "simag": simag,
        "sibry": sibry,
        "bcentr": bcentr,
        "current": current,
        # "simag": simag,
        # "rmaxis": rmaxis,
        # "zmaxis": zmaxis,
        # "sibry": sibry,
        "fpol": fpol,
        "pres": pres,
        "ffprim": ffprim,
        "pprime": pprime,
        "psirz": psirz,
        "qpsi": qpsi,
        "nbbbs": nbbbs,
        "bbbsrz": bbbsrz,
        "limitr": limitr,
        "limrz": limrz,
        "r": r,
        "z": z

    }

 

def RZmap(Rzdata, gg, rhoo_method=1):
    """
    Rzdata is a matrix with shape of m x n, m is the number of Rz points,
        n colums should be R, z, data, data_error, ...
    gg is the dictionary of gfile, should have r, z  or rdim, zdim, rleft, zmid.
    rhoo_method is the method option for rho > 1
        0:  Extrapolate rho from the rhon vs psin spline fitting
        1:  rhon = sqrt(psin)
        2:  rhon = r/a, where r=R-rmaxis, a=Rsep-rmaxis, at z=zmaxis plane 

    return a matrix with shape of m x n, the first two colums are psi and rho,
        the rows are sorted by monotonic psi
    """
    import numpy as np
    from scipy import interpolate

    mm = len(Rzdata[:, 0])

    psirz  = gg['psirz']
    sibry  = gg['sibry'] 
    simag  = gg['simag']
    rmaxis = gg['rmaxis']
    zmaxis = gg['zmaxis']
    qpsi   = gg['qpsi']
    rr     = gg['r']
    zz     = gg['z']

    psi_rho_data = Rzdata.copy()

    nw = len(qpsi)
    psin = np.linspace(0., 1., nw)
    phix = np.zeros(nw)
    tck  = interpolate.splrep(psin, qpsi)
    for ii in range(nw):
        phix[ii] = interpolate.splint(psin[0], psin[ii], tck)
    phin = (phix - phix[0])/(phix[-1] - phix[0]) # normalized toroidal flux
    rhon = np.sqrt(phin)
    tck_rhopsi = interpolate.splrep(psin, rhon)  # tck for pho vs psi

    psiY = np.zeros(mm)
    psirzn = (psirz - simag) / (sibry - simag)

    # Replace interp2d with RegularGridInterpolator
    ff = interpolate.RegularGridInterpolator((zz, rr), psirzn, method='linear', bounds_error=False, fill_value=None)
    for ii in range(mm):
        rrr = Rzdata[ii, 0]
        zzz = Rzdata[ii, 1]
        psiY[ii] = ff((zzz, rrr))
        psi_rho_data[:, 0] = psiY
    
    psi_rho_data = psi_rho_data[psi_rho_data[:, 0].argsort()]
    for ii in range(mm-1):
        # To avoid almost same psi value
        if psi_rho_data[ii+1, 0] - psi_rho_data[ii, 0] <= 0.:
            psi_rho_data[ii+1, 0] = psi_rho_data[ii+1, 0] + 0.0001
    psi_rho_data[:, 1] = interpolate.splev(psi_rho_data[:, 0], tck_rhopsi)

    # Set the rhon values for rhon > 1
    psiY = psi_rho_data[:, 0]
    if psiY[-1] > 1. and rhoo_method in [1, 2]:
        idx  = np.searchsorted(psiY, 1.)  # Find the index of psin=1
        if rhoo_method == 1:   
            rhoo = np.sqrt(psiY[idx:])
            psi_rho_data[idx:, 1] = rhoo
        if rhoo_method == 2:
            rrminor  = np.linspace(rmaxis, rr[-1], nw)
            # psi at the z=zmaxis plane
            psiza  = ff((zmaxis * np.ones_like(rrminor), rrminor))
            tck = interpolate.splrep(psiza, rrminor)
            xxx = np.insert(psiY[idx:], 0, 1.)  # Add a psin=1 point
            # Calculate the R value for psin >= 1.0 at the z=zmaxis plane
            rout = interpolate.splev(xxx, tck)
            rhoo = (rout[1:] - rmaxis) / (rout[0] - rmaxis)
            psi_rho_data[idx:, 1] = rhoo

    return psi_rho_data



def read_from_MDS(conn, timeid):
    """
    To avoid to connect the MDS+ server and open the tree again and again,
        This function assume the MDS+ server has been connected and the tree
        has been opened.
    conn: the the connction to 
    """
    geqdsk = {}
    for key in _gdatnames:
        # print(key)
        nx  = len(_gdatnames[key][1])
        if key in ['lim', 'limitr', 'r', 'z']:      # no time dependent
            tag = 'data(\\' + key + ')'
        else:
            tag = 'data(\\' + key + ')' + "[%s%s]"%( ''.join(['*,']*nx), timeid )
        geqdsk[_gdatnames[key][2]] = conn.get(tag).data()
    geqdsk['head'] = 'Read from EAST MDS+ server '
    geqdsk['nw'] = len( geqdsk['r'] )
    geqdsk['nh'] = len( geqdsk['z'] )
    geqdsk['rleft'] = geqdsk['r'][0]
    geqdsk['rdim']  = geqdsk['r'][-1] - geqdsk['r'][0]
    geqdsk['zmid']  = (geqdsk['z'][-1] + geqdsk['z'][0]) /2.
    geqdsk['zdim']  = geqdsk['z'][-1] - geqdsk['z'][0]
    geqdsk['rcentr'] = geqdsk['fpol'][-1] / geqdsk['bcentr']

    return geqdsk

def save(p, file, *args, **kwargs):
    """
    :param profile: object

    :param file: file path / file
    :return:
    """
    if type(file) is str:
        file = open(file, "w")

    nw = p["nw"]
    nh = p["nh"]

    file.write("%48s%4i%4i%4i\n" % (p["head"], 3, p["nw"], p["nh"]))
    file.write("%16.9e%16.9e%16.9e%16.9e%16.9e\n" %
               (p["rdim"], p["zdim"], p["rcentr"], p["rleft"], p["zmid"]))
    file.write("%16.9e%16.9e%16.9e%16.9e%16.9e\n" %
               (p["rmaxis"], p["zmaxis"], p["simag"], p["sibry"], p["bcentr"]))
    file.write("%16.9e%16.9e%16.9e%16.9e%16.9e\n" %
               (p["current"], p["simag"], 0, p["rmaxis"], 0))
    file.write("%16.9e%16.9e%16.9e%16.9e%16.9e\n" %
               (p["zmaxis"], 0, p["sibry"], 0, 0))

    def _write_data(d):
        count = len(d)
        for n in range(count):
            file.write("%16.9e" % d[n])
            if (n == count - 1) or ((n + 1) % 5 == 0):
                file.write('\n')

    _write_data(p["fpol"])
    _write_data(p["pres"])
    _write_data(p["ffprim"])
    _write_data(p["pprime"])
    _write_data(p["psirz"].reshape([nw * nh]))
    _write_data(p["qpsi"])
    file.write("%5i%5i\n" % (p["bbbsrz"].shape[0], p["limrz"].shape[0]))
    _write_data(p["bbbsrz"].reshape([p["bbbsrz"].size]))
    _write_data(p["limrz"].reshape([p["limrz"].size]))

    file.close()

    return


_gdatnames = {
#name on EAST MDS+: unit         , structure         , name in geqdsk dict
    'bcentr':[ 'T'               , []                , 'bcentr' ] ,
    'bdry'  :[ 'm'               , ['n','n']         , 'bbbsrz' ] , # s
    'EFIT_MFILE:cpasma':[ 'A'    , []                , 'current' ] ,
#    'dmion' :[ 'kg/m^3'          , ['psin']          ,  ] ,
#    'epoten':[ 'V'               , ['psin']          ,  ] ,
    'ffprim':[ '(Tm)^2/(Vs/rad)' , ['psin']          , 'ffprim' ] ,
    'fpol'  :[ 'Tm'              , ['psin']          , 'fpol' ] ,
    'gtime' :[ 'ms'              , []                , 'gtime' ] ,
    'lim'   :[ 'm'               , ['n']             , 'limrz' ] , # s
    'limitr':[ '(dimensionless)' , []                , 'limitr' ],
#    'mh'    :[ '(dimensionless)' , []                , 'nh' ],
#    'mw'    :[ '(dimensionless)' , []                , 'nw' ] ,
    'nbdry' :[ '(dimensionless)' , []                , 'nbbbs' ] ,
    'pprime':[ '(N/m^2)/(Vs/rad)', ['psin']          , 'pprime' ] ,
    'pres'  :[ 'Pa'              , ['psin']          , 'pres' ] ,
#    'pressw':[ 'N/m^2'           , ['psin']          ,  ] ,
    'psirz' :[ 'Vs/rad'          , ['rgrid','zgrid'] , 'psirz' ] ,
#    'pwprim':[ '(N/m^2)/(Vs/rad)', ['psin']          ,  ] ,
    'qpsi'  :[ '(dimensionless)' , ['psin']          , 'qpsi' ] ,
    'r'     :[ 'm'               , []                , 'r' ] , # s
#    'rgrid1':[ 'm'               , []                ,  ] , # s
#    'rhovn' :[ '(dimensionless)' , ['psin']          ,  ] ,
    'rmaxis':[ 'm'               , []                , 'rmaxis' ] ,
#    'rzero' :[ 'm'               , []                , 'rcentr' ] ,
#    'rvtor' :[ 'm'               , []                ,  ] ,
    'ssibry':[ 'Vs/rad'          , []                , 'sibry' ] ,
    'ssimag':[ 'Vs/rad'          , []                , 'simag' ] ,
#    'xdim'  :[ 'm'               , []                , 'rdim' ] ,
    'z'     :[ 'm'               , []                , 'z' ] , # s
#    'zdim'  :[ 'm'               , []                , 'zdim' ] ,
    'zmaxis':[ 'm'               , []                , 'zmaxis' ] ,
#    'zmid'  :[ 'm'               , []                , 'zmid' ] ,
}
