import os, sys
print('__file__ =', repr(__file__))
print('abspath  =', repr(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
print('SCRIPT_DIR =', repr(SCRIPT_DIR))
print('ROOT_DIR   =', repr(ROOT_DIR))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SCRIPT_DIR)
print('profile_nn resolvable:', os.path.isdir(os.path.join(ROOT_DIR, 'profile_nn')))
print('cnn_deeponet exists:', os.path.exists(os.path.join(ROOT_DIR, 'profile_nn', 'cnn_deeponet.py')))
import profile_nn
print('profile_nn.__file__ =', repr(profile_nn.__file__))
import profile_nn.cnn_deeponet as cd
print('OK', cd.__file__)
