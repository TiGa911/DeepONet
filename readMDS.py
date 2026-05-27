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

    # Read geqdsk from MDS+ server
    # TREE  = 'pefitrt_east'
    TREE  = 'efit_east'
    conn.openTree(TREE, shot)
    g_times = conn.get(r'data(\GTIME)').data()  # All time slices for gfile on MDS+
    timeid  = np.argmin(abs(g_times - time))
    print('timeid:',g_times[timeid])
    real_time = g_times[timeid]
    gg          = geqdsk.read_from_MDS(conn, timeid)
    gg['head']  = f'{shot}_{time}'
    conn.closeTree(TREE, shot)

    # filename = f"{shot}_{time}_gfile"
    # open(filename, 'w').close()  # 创建空文件
    # print(f"已创建空文件: {filename}")
    # geqdsk.save(gg, '{}_{}_gfile'.format(shot,time))


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
    # try:
    TREE = 'TS_EAST'
    conn.openTree(TREE, shot)
    TS_times = conn.get(r'dim_of(\Te_coreTS)').data()
    print('TS_times:', TS_times)
    timeid = np.argmin(abs(TS_times - time))
    print('timeid:', timeid)
    index = '[*,' + str(timeid) + ']'
    print(index)
    
    # 读取所有数据
    R_TS = conn.get(r'data(\R_coreTS)').data()
    z_TS = conn.get(r'data(\z_coreTS)').data()
    Te_TS = conn.get(r'data(\Te_coreTS)' + index).data()   
    Te_TSerr = conn.get(r'data(\Te_coreTSerr)' + index).data()
    ne_TS = conn.get(r'data(\ne_coreTS)' + index).data()
    ne_TSerr = conn.get(r'data(\ne_coreTSerr)' + index).data()

    # 确保所有数据都是一维数组
    arrays_to_process = [R_TS, z_TS, Te_TS, Te_TSerr, ne_TS, ne_TSerr]
    array_names = ['R_TS', 'z_TS', 'Te_TS', 'Te_TSerr', 'ne_TS', 'ne_TSerr']
    
    # 调试信息
    for name, arr in zip(array_names, arrays_to_process):
        print(f"{name} shape: {np.array(arr).shape}, type: {type(arr)}")
    
    # 处理长度不一致的问题
    processed_arrays = []
    for arr in arrays_to_process:
        arr_flat = np.array(arr).flatten()
        processed_arrays.append(arr_flat)
    
    # 找到所有数组的最小长度
    lengths = [len(arr) for arr in processed_arrays]
    min_length = min(lengths)
    print(f"最小公共长度: {min_length}")
    
    # 截断所有数组到相同长度
    final_arrays = [arr[:min_length] for arr in processed_arrays]
    
    # 创建TSdata
    TSdata = np.column_stack(final_arrays)
    print(f"TSdata 最终形状: {TSdata.shape}")
    
    # 继续后续处理...
    TS_mapped = geqdsk.RZmap(TSdata, gg, rhoo_method=1)
    
    # TS
    Te['TS'] = {}
    Te['TS']['type'] = 'Te'
    Te['TS']['Rho'] = TS_mapped[:,1]
    Te['TS']['data'] = TS_mapped[:,2]
    print('TS_mapped[:,2]',TS_mapped[:,2])

    ne['TS'] = {}
    ne['TS']['type'] = 'neTS'
    ne['TS']['Rho'] = TS_mapped[:,1]
    ne['TS']['data'] = TS_mapped[:,4]
    
    conn.closeTree(TREE, shot)
    # except Exception as TS_EAST_ERROR:
    #     print('TS EAST ERROR:', TS_EAST_ERROR)
    #     status['TS_status'] = 0
    #     pass

    # %%
    # Ti and Te from TXCS
    try:
        TREE = 'TXCS_EAST'
        conn.openTree(TREE, shot)
        TXCS_times = conn.get(r'dim_of(\Ti_TXCS)').data()
        timeid = np.argmin(abs(TXCS_times - time))
        index  = '[' + str(timeid) + ',*]'
        Ti_TXCS = conn.get(r'data(\Ti_TXCS)' + index).data() /1000.
        print('Ti_TXCS',Ti_TXCS)
        R_TXCS  = 1.9 #63948
        z_TXCS  = conn.get(r'data(\z_TXCS)').data() /100.
        print('z_TXCS',z_TXCS)
        # R_TXCS  = conn.get(r'data(\R_TXCS)').data()
        print('R_TXCS',R_TXCS)
        Te_TXCS = conn.get(r'data(\Te_TXCS)' + index).data() /1000.

        not_nan_idx = np.isfinite(Ti_TXCS)
        Te_TXCS = Te_TXCS[not_nan_idx]
        Ti_TXCS = Ti_TXCS[not_nan_idx]

        #因为Ti基本没有数据所以这里是自己填充的
        z_TXCS  = np.transpose([z_TXCS])[not_nan_idx]
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
        print('TXCS_EAST ERROR')
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


    # %%
    # H98 from energy_east
    try:
        conn.openTree('energy_east', shot)
        H98_times = conn.get(r'dim_of(\H98_MHD)').data()
        H98_data = conn.get(r'data(\H98_MHD)').data()
        h98_timeid = np.argmin(abs(H98_times - time))
        h98_value = float(np.array(H98_data).flatten()[h98_timeid])
        print('h98_value',H98_data)
        print('h98_value',h98_timeid)
        print('h98_value',h98_value)
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
        print('ne:',ne_Refl)
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




    data = {
        'ne': ne,
        'Te': Te,
        'Ti': Ti,
        'zeff': z_eff_value,
        'H98': {
            'value': h98_value,
            'time': H98_times,
            'data': H98_data
        }
    }
    return data, status,real_time
    # return  TXCS_mapped, Refl_mapped