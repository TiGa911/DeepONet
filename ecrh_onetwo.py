import numpy as np
from Namelist3 import Namelist

def read_lhcd_data(filename):
    rho, power, current = [], [], []
    with open(filename, 'r') as f:
        next(f)  # 跳过表头
        for line in f:
            data = line.strip().split('\t')
            rho.append(float(data[0]))
            power.append(float(data[1]))
            current.append(float(data[2]))
    return np.array(rho), np.array(power), np.array(current)

def ecrh_onetwo(shot,time):
# def ecrh_toray_namelist2(shot,time):
# rho, power, current = read_lhcd_data("LHCD_data_{shot}_{time}.txt")
    rho, power, current = read_lhcd_data(f"LHCD_data_{shot}_{time}.txt")

    obj = Namelist()
    obj.read('inone_template')

    obj['NAMELIS2']['extcurrf_id']='lhw'
    obj['NAMELIS2']['extqerf_id']='lhw'
    obj['NAMELIS2']['extqirf_id']='lhw'

    obj['NAMELIS2']['extcurrf']=0
    obj['NAMELIS2']['extqerf']=0
    obj['NAMELIS2']['extqirf']=1

    obj['NAMELIS2']['extcurrf_amps']=42100.0
    obj['NAMELIS2']['extqerf_watts']=2e6
    obj['NAMELIS2']['extqirf_watts']=0.04

    obj['NAMELIS2']['extcurrf_curr']=list(current * 1e6 / 1e4) # 单环向谐波
    obj['NAMELIS2']['extqerf_qe']=list(power / 1e6)  # W/m³ → W/cm³
    obj['NAMELIS2']['extqirf_qi']=list(np.zeros_like(rho))

    obj['NAMELIS2']['extcurrf_rho']=list(rho)
    obj['NAMELIS2']['extqerf_rho']=list(rho)
    obj['NAMELIS2']['extqirf_rho']=list(rho)

    obj['NAMELIS2']['extcurrf_nj']=len(rho)    # 径向网格点数
    obj['NAMELIS2']['extqerf_nj']=len(rho)
    obj['NAMELIS2']['extqirf_nj']=len(rho)

    obj['NAMELIS2']['rfmode'] = 'ech'
    obj['NAMELIS2']['genraydat'] = ''
    obj['NAMELIS2']['rfon'] = -44.422
    obj['NAMELIS2']['rftime'] = 4400
    obj['NAMELIS2']['rfpow'] = 1000.7742


    obj['NAMELIS2']['freq'] = 140000000000.0  ###

    obj['NAMELIS2']['xec'] = 300 ##@@
    obj['NAMELIS2']['zec'] = -30 ##
    obj['NAMELIS2']['thetec'] = 80.0 ##
    obj['NAMELIS2']['phaiec'] = 200.0 ##

    obj['NAMELIS2']['irfcur'] = 1.0
    obj['NAMELIS2']['wrfo'] = 0.042
    obj['NAMELIS2']['nray'] = 30
    obj['NAMELIS2']['idamp'] = 2
    obj['NAMELIS2']['hlwec'] = 1.182
    obj['NAMELIS2']['ratwec'] = 1
    obj.write('inone_ecrh')
    print('ecrh_written')
    return

