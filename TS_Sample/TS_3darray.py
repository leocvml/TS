import tensorflow as tf
import numpy as np
from matplotlib import *
import matplotlib.image as mp_image
import matplotlib.pyplot as plt
def Array3d():
    tensor_3d = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
    #[plane , row ,col]
    print("3d" , tensor_3d)
    print(tensor_3d[0,0,1])
    print(tensor_3d[1,0,1])


    
    
