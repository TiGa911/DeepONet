# -*- coding: utf-8 -*-
# %%
import geqdsk
import numpy as np
# from MDSplus import Connection
from MDSplus.connection import Connection

MDSIP = '202.127.204.42'

def readmds(shot, time):
    # shot = 63948
    # shot = 71320
    # 71320 71326 73999
    # time = 6.0
    # status: 0 for error    1 for normal operation

    conn   = Connection(MDSIP)

    # === 本地 gfile 优先检查 ===
    # 如果当前目录下存在匹配的本地 gfile，直接加载，跳过 MDSplus 读取
    import glob as _glob
    _local_gfiles = _glob.glob(f'{shot}_*_gfile')
    _local_gfile = None
    _best_dt = float('inf')
    for _gf in _local_gfiles:
        try:
            _gf_time = float(_gf.replace(f'{shot}_', '').replace('_gfile', ''))
            _dt = abs(_gf_time - time)
            if _dt < _best_dt:
                _best_dt = _dt
                _local_gfile = _gf
        except ValueError:
            pass

    if _local_gfile is not None and _best_dt < 0.5:
        # 使用本地 gfile，跳过 MDSplus 读取
        real_time = float(_local_gfile.replace(f'{shot}_', '').replace('_gfile', ''))
        print(f'[本地 gfile] {_local_gfile} (Δt={_best_dt*1000:.1f}ms)')
        gg = geqdsk.load(_local_gfile)
        gg['head'] = f'{shot}_{time}'
        # 跳过 MDSplus gfile 读取，直接进入诊断数据读取
        pass
    else:
        # Read geqdsk from MDS+ server
        # TREE  = 'pefitrt_east'
        TREE  = 'efit_east'
        try:
            conn.openTree(TREE, shot)
        except Exception as e:
            raise ConnectionError(
                f'Shot {shot}: efit_east tree not available on MDSplus ({e}). '
                f'This shot has no EFIT equilibrium data — gfile cannot be read.'
            ) from e

        g_times = np.asarray(conn.get(r'data(\GTIME)').data(), dtype=np.float64).flatten()

        # === gfile 可用性检查 ===
        if len(g_times) == 0:
            conn.closeTree(TREE, shot)
            raise ValueError(
                f'Shot {shot} @ {time}s: efit_east tree has zero gfile time slices. '
                f'This shot has no EFIT equilibrium reconstruction data on MDSplus.'
            )

        dt = abs(g_times - time)
        timeid = np.argmin(dt)
        time_gap = dt[timeid]
        real_time = float(g_times[timeid])

        # 警告：gfile 时间与请求时间差距过大
        if time_gap > 0.5:
            print(
                f'⚠ Shot {shot}: gfile time {real_time:.3f}s is {time_gap*1000:.0f}ms '
                f'away from requested {time:.3f}s — equilibrium may not match diagnostics'
            )

        print(f'gfile: shot={shot} timeid={timeid} real_time={real_time:.4f}s (Δt={time_gap*1000:.1f}ms)')
        gg = geqdsk.read_from_MDS(conn, timeid)
        gg['head'] = f'{shot}_{time}'
        conn.closeTree(TREE, shot)

    # === gfile 自动保存，供 lower_view_final_nn.py::_find_gfile() 使用 ===
    gfile_time = float(real_time)
    gfile_name = f'{shot}_{gfile_time}_gfile'
    geqdsk.save(gg, gfile_name)

    # classify
    ne, Te, Ti, zeff= {}, {}, {}, {}

    # status default 1
    status = {}    
    status['TS_status'] = 1
    status['TXCS_status'] = 1
    status['Refl_status'] = 1
    # status['HRS_status'] = 1


    # # %%
    # # Te and ne from TS
    try:
        TREE = 'TS_EAST'
        conn.openTree(TREE, shot)
        TS_times = conn.get(r'dim_of(\Te_coreTS)').data()
        print('TS_times:',TS_times)
        timeid = np.argmin(abs(TS_times - time))
        print('timeid:',timeid)
        index  = '[*,' + str(timeid) + ']'
        print(index)
        R_TS     = conn.get(r'data(\R_coreTS)').data()
        print('R_TS:',R_TS)
        z_TS     = conn.get(r'data(\z_coreTS)').data()
        print('z_TS',z_TS)
        Te_TS    = conn.get(r'data(\Te_coreTS)' + index).data()   
        Te_TSerr = conn.get(r'data(\Te_coreTSerr)' + index).data()
        ne_TS    = conn.get(r'data(\ne_coreTS)' + index).data()  # in 1.e19
        ne_TSerr = conn.get(r'data(\ne_coreTSerr)' + index).data()
        if len(Te_TS) > len(R_TS):
            Te_TS = np.delete(Te_TS, 0)
        if len(Te_TSerr) > len(R_TS):
            Te_TSerr = np.delete(Te_TSerr, 0)
        if len(ne_TS) > len(R_TS):
            ne_TS = np.delete(ne_TS, 0)
        if len(ne_TSerr) > len(R_TS):
            ne_TSerr = np.delete(ne_TSerr, 0)
        # print('R_TS',R_TS)
        # print(R_TS.shape)
        # print('z_TS',z_TS)
        # print(z_TS.shape)
        # print('Te_TS',Te_TS)
        # print(Te_TS.shape)
        # print('Te_TSerr',Te_TSerr)
        # print(Te_TSerr.shape)
        # print('ne_TS',ne_TS)
        # print(ne_TS.shape)
        # print('ne_TSerr',ne_TSerr)
        # print(ne_TSerr.shape)
        TSdata =  np.transpose( [R_TS, z_TS, Te_TS, Te_TSerr, ne_TS, ne_TSerr] )
        # for i in TSdata:
        #     print(np.shape(i))
        #Tedata = np.loadtxt('Teprof_5076.txt')
        TS_mapped = geqdsk.RZmap(TSdata, gg, rhoo_method=1)  # 6列 psi,rho,te_data,err,ne,err
        # print(TS_mapped)
        # print(TS_mapped.shape)
        # TS
        Te['TS'] = {}
        Te['TS']['type'] = 'Te'
        Te['TS']['Rho'] = TS_mapped[:,1]
        Te['TS']['data'] = TS_mapped[:,2]

        ne['TS'] = {}
        ne['TS']['type'] = 'neTS'
        ne['TS']['Rho'] = TS_mapped[:,1]
        ne['TS']['data'] = TS_mapped[:,4]
        conn.closeTree(TREE, shot)
    except Exception as TS_EAST_ERROR:
        print('TS EAST ERROR')
        status['TS_status'] = 0
        pass

    # %%
    # Ti and Te from TXCS
    try:
        TREE = 'TXCS_EAST'
        conn.openTree(TREE, shot)
        TXCS_times = conn.get(r'dim_of(\Ti_TXCS)').data()
        timeid = np.argmin(abs(TXCS_times - time))
        index  = '[' + str(timeid) + ',*]'
        Ti_TXCS = np.array(conn.get(r'data(\Ti_TXCS)' + index).data() /1000.).flatten()
        print('Ti_TXCS',Ti_TXCS)
        R_TXCS  = 1.9 #63948
        try:
            z_TXCS  = np.array(conn.get(r'data(\z_TXCS)').data() /100.).flatten()
            print('z_TXCS',z_TXCS)
        except Exception:
            # Fallback: generate linear z array from -0.5 to +0.5 cm across channels
            n_ch = len(Ti_TXCS)
            z_TXCS = np.linspace(-0.5, 0.5, n_ch) / 100.  # in meters
            print('z_TXCS (fallback linear)', z_TXCS[:3], '...', z_TXCS[-3:])
        # R_TXCS  = conn.get(r'data(\R_TXCS)').data()
        print('R_TXCS',R_TXCS)

        # Te_TXCS may have different dimensions or be absent; use NaN fallback
        try:
            Te_TXCS = np.array(conn.get(r'data(\Te_TXCS)' + index).data() /1000.).flatten()
        except Exception:
            Te_TXCS = np.full_like(Ti_TXCS, np.nan)

        # Align z_TXCS with Ti_TXCS channels
        if len(z_TXCS) != len(Ti_TXCS):
            if len(z_TXCS) == 1:
                z_TXCS = np.full(len(Ti_TXCS), z_TXCS[0])
            else:
                z_TXCS = np.resize(z_TXCS, len(Ti_TXCS))
        if len(Te_TXCS) != len(Ti_TXCS):
            Te_TXCS = np.full(len(Ti_TXCS), np.nan)

        not_nan_idx = np.isfinite(Ti_TXCS)
        if not_nan_idx.sum() < 1:
            conn.closeTree(TREE, shot)
            status['TXCS_status'] = 0
            # Skip TXCS mapping but continue with remaining diagnostics
            # Use a flag to skip TXCS processing below
            raise ValueError('No valid Ti data at this time slice')

        Ti_TXCS = Ti_TXCS[not_nan_idx]
        Te_TXCS = Te_TXCS[not_nan_idx]
        z_TXCS  = z_TXCS[not_nan_idx]
        R_TXCS  = np.ones(len(z_TXCS)) * R_TXCS

        TXCSdata = np.transpose( [R_TXCS, z_TXCS, Ti_TXCS, Te_TXCS] )
        TXCS_mapped = geqdsk.RZmap(TXCSdata, gg, rhoo_method=1)
        #TXCS
        Ti['TXCS'] = {}
        Ti['type'] = 'Ti'
        Ti['TXCS']['Rho'] = TXCS_mapped[:, 1]
        Ti['TXCS']['data'] = TXCS_mapped[:, 2]

        Te['TXCS'] = {}
        Te['TXCS']['type'] = 'TeTXCS'
        Te['TXCS']['Rho'] = TXCS_mapped[:, 1]
        Te['TXCS']['data'] = TXCS_mapped[:, 3]
        conn.closeTree(TREE, shot)
    except Exception as TXCS_EAST_ERROR:
        print('TXCS_EAST ERROR:', str(TXCS_EAST_ERROR)[:80])
        status['TXCS_status'] = 0
        pass
    
    
    
    try:
        conn.openTree('Brem_EAST', shot)
        #有效Z
        Z_eff_times = conn.get(r'dim_of(\Zeff_ave)').data()
        print('Z_eff_times',Z_eff_times)
        timeid = np.argmin(abs(TXCS_times - time))
        index  = '[' + str(timeid) + ',*]'
        R_Brem = conn.get(r'data(\Zeff_ave)').data()
        print('R_Brem', R_Brem)
        Z_Brem = conn.get(r'data(\Zeff_ave)').data()
        Z_eff = conn.get(r'data(\Zeff_ave)').data()        
        ZEFF = Z_eff[timeid]
        print('ZEFF',ZEFF)
        Zeff_aveErr = conn.get(r'data(\Zeff_aveErr)').data()
        ZEFF_ERROR = Zeff_aveErr[timeid]
        print('Z_eff_err',ZEFF_ERROR)    
        timeid = np.argmin(abs(Z_eff_times - time))
        z_eff_value = Z_eff[timeid]
        conn.closeTree('Brem_EAST', shot)
        print('zeff_Brem')
        TXCSdata = np.transpose( [R_Brem, Z_Brem, Z_eff, Zeff_aveErr])
        TXCS_mapped = geqdsk.RZmap(TXCSdata, gg, rhoo_method=1)
        Ti['Brem'] = {}
        Ti['type'] = 'Ti'
        Ti['Brem']['Rho'] = TXCS_mapped[:, 1]
        Ti['Brem']['data'] = TXCS_mapped[:, 2]
        Te['Brem'] = {}
        Te['Brem']['type'] = 'TeTXCS'
        Te['Brem']['Rho'] = TXCS_mapped[:, 1]
        Te['Brem']['data'] = TXCS_mapped[:, 3]

    except Exception as ZEFFerror:
        z_eff_value = 2.2
        print('zeff_error')
        zeff_array = np.full(shape=51, fill_value=2.5)
        pass

    # %% H98 from energy_east
    # 读取能量约束因子 H98，用于 H/L 模式判定
    # 阈值：H98 >= 0.7 为 H-mode（与 fitting_mtanh.judge_plasma_mode 一致）
    try:
        conn.openTree('energy_east', shot)
        H98_times = conn.get(r'dim_of(\H98_MHD)').data()
        H98_data = conn.get(r'data(\H98_MHD)').data()
        h98_timeid = np.argmin(abs(H98_times - time))
        h98_value = float(np.array(H98_data).flatten()[h98_timeid])
        conn.closeTree('energy_east', shot)
        print('H98_value:', h98_value)
    except Exception as H98_ERROR:
        print('H98_ERROR:', H98_ERROR)
        h98_value = np.nan
        H98_times = np.array([])
        H98_data = np.array([])

    # %%
    # CXRS
    # TREE = 'CXRS_EAST'
    # conn.openTree(TREE, shot)
    # CXRS_times = conn.get(r'dim_of(\Ti_CXRS_T)').data()
    # timeid = np.argmin(abs(CXRS_times - time))
    # index  = '[' + str(timeid) + ',*]'
    # R_CXRS_T  = conn.get(r'data(\R_CXRS_T)').data()
    # Z_CXRS_T  = conn.get(r'data(\Z_CXRS_T)').data()
    # Ti_CXRS_T = conn.get(r'data(\Ti_CXRS_T)' + index).data() /1000.
    # Ti_CXRS_Terr = conn.get(r'data(\Ti_CXRS_Terr)' + index).data() /1000
    # Ti0_CXRS_T = conn.get(r'data(\Ti0_CXRS_T)' + index).data() 
    # Ti0_CXRS_Ter = conn.get(r'data(\Ti0_CXRS_Ter)' + index).data()
    # Vt0_CXRS_T = conn.get(r'data(\Vt0_CXRS_T)' + index).data()
    # Vt0_CXRS_Ter = conn.get(r'data(\Vt0_CXRS_Ter)' + index).data()
    # Vt_CXRS_T = conn.get(r'data(\Vt_CXRS_T)' + index).data()
    # Vt_CXRS_Terr = conn.get(r'data(\Vt_CXRS_Terr)' + index).data()  
    
    # CXRSdata = np.transpose( [R_CXRS, z_CXRS, Ti_CXRS, Te_CXRS] )
    # CXRS_mapped = geqdsk.RZmap(CXRSdata, gg, rhoo_method=1)
    # # CXRS
    # Ti['CXRS'] = {}
    # Ti['CXRS']['type'] = 'TiCXRS'
    # Ti['CXRS']['Rho'] = CXRS_mapped[:, 1]
    # Ti['CXRS']['data'] = CXRS_mapped[:, 2]
    # Te['CXRS'] = {}
    # Te['CXRS']['type'] = 'TeCXRS'
    # Te['CXRS']['Rho'] = CXRS_mapped[:, 1]
    # Te['CXRS']['data'] = CXRS_mapped[:, 3]
    # conn.closeTree(TREE, shot)
    
    # %%
    # HRS_EAST
    # try:
    #     TREE = 'HRS_EAST'
    #     conn.openTree(TREE, shot)
    #     CXRS_times = conn.get(r'dim_of(\Te_HRS)').data()
    #     timeid = np.argmin(abs(CXRS_times - time))
    #     index  = '[' + str(timeid) + ',*]'
    #     Te0_HRS = conn.get(r'data(\Te0_HRS)' + index).data() /1000.
    #     Te0_HRSerr = conn.get(r'data(\Te0_HRSerr)' + index).data() /1000.
    #     R_HRS = conn.get(r'data(\R_HRS)')
    #     Z_HRS = conn.get(r'data(\R_HRS)')
    #     Te_HRS = conn.get(r'data(\Te_HRS)' + index).data() /1000.
    #     Te_HRSerr = conn.get(r'data(\Te_HRSerr)' + index).data() /1000.
    #     HRSData = np.transpose([R_HRS, Z_HRS, Te_HRS, Te_HRSerr, Te0_HRS, Te0_HRSerr])
    #     HRS_mapped = geqdsk.RZmap(ReflData, gg, rhoo_method=1)
    
    #     ne['HRS'] = {}
    #     ne['HRS']['type'] = 'HRS'
    #     ne['HRS']['Rho'] = Refl_mapped[:,1]
    #     ne['HRS']['data'] = Refl_mapped[:,2]   
    # except Exception as HRS_ERROR:
    #     print('HRS_EAST_ERROR')
    #     status['HRS_status'] = 0
    #     pass
    
    # %%
    # ne from Reflectometer
    try:
        TREE = 'ReflJ_EAST'
        conn.openTree(TREE, shot)
        Refl_times = conn.get(r'dim_of(\ne_ReflJ)').data()
        timeid = np.argmin(abs(Refl_times - time))
        index = '[*,' + str(timeid) + ']'
        R_Refl  = conn.get(r'data(\R_ReflJ)' + index)
        z_Refl  = conn.get(r'data(\z_ReflJ)')
        z_Refl  = np.ones(len(R_Refl)) * z_Refl
        ne_Refl = conn.get(r'data(\ne_ReflJ)' + index)
        ReflData = np.transpose([R_Refl, z_Refl, ne_Refl])
        Refl_mapped = geqdsk.RZmap(ReflData, gg, rhoo_method=1)
        # Reflectometer
        ne['Refl'] = {}
        ne['Refl']['type'] = 'Refl'
        ne['Refl']['Rho'] = Refl_mapped[:,1]
        ne['Refl']['data'] = Refl_mapped[:,2]    
        conn.closeAllTrees()
    except Exception as Reflj_EAST_ERROR:
        print('Refl_EAST_ERROR')
        status['Refl_status'] = 0
        pass

    data = {'ne': ne, 'Te': Te, 'Ti': Ti, 'zeff': z_eff_value,
            'H98': {'value': h98_value, 'time': H98_times, 'data': H98_data}}
    status['H98'] = h98_value  # 方便调用方快速访问
    return data, status, real_time
    # return  TXCS_mapped, Refl_mapped