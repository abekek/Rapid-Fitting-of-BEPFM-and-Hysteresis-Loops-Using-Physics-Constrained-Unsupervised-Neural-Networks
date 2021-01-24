"""
Created on Sun Jan 24 16:34:00 2021
@author: Alibek Kaliyev
"""

import h5py


# define a small function called 'print_tree' to look at the folder tree stucture
def print_tree(parent):
    print(parent.name)
    if isinstance(parent, h5py.Group):
        for child in parent:
            print_tree(parent[child])
