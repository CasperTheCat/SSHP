from ctypes import cdll
treelib = cdll.LoadLibrary('./libcpptree.so') # Linux only now...

class 3dTree(object):
    """
    Implementation of a 3D tree
    """
    def __init__(self):
        self.obj = cons;

    def insert(self, P, T, cd):
        treelib.tree_insert(self.obj, P, T, cd)

