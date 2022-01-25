from time import sleep

import numpy as np
import matplotlib.pyplot as plt
from variables import global_variables

if __name__ == '__main__':
    while True:
        data = np.genfromtxt(global_variables['log_file'], delimiter=",", names=True)
        
        plt.plot(range(1, data['min'].shape[0]+1),
                 data['min'], '-o', label="Min")
        plt.plot(range(1, data['mean'].shape[0]+1),
                 data['mean'], '-o', label="Mean")
        plt.plot(range(1, data['max'].shape[0]+1),
                 data['max'], '-o', label="Max")
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.show()
        