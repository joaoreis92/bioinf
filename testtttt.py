import numpy as np
import pandas as pd

for i in range(1,int(input())): #More than 2 lines will result in 0 score. Do not leave a blank line also
    print((int(10**(i-4))*int(10**(i-3))*int(10**(i-2))*int(10**(i-1)))*i)