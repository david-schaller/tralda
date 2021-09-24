# -*- coding: utf-8 -*-

"""
Tree Tools.
"""

from tralda.datastructures.Tree import Tree


__author__ = 'David Schaller'


def assert_leaf_sets_equal(trees):
    """Checks whether trees have the same set of unique leaf labels.
    
    Parameters
    ----------
    trees : sequence of Tree instances
    
    Returns
    -------
    set
        The common set of leaf labels; or False.
        
    Raises
    ------
    TypeError
        In case of input instances that are not of type Tree.
    RuntimeError
        If the sequence contains empty trees, the tree do not share the same
        set of leaves, or the leaf labels are not unique in one tree.
    AttributeError
        If the label attribute is not set for some leaf.
    """
    
    if not trees:
        raise RuntimeError('empty list of trees')
        
    L = None
        
    for T_i in trees:
        
        if not isinstance(T_i, Tree):
            raise TypeError("not a 'Tree' instance")
        
        L2 = set()
        for v in T_i.leaves():
            if v.label in L2:
                    raise RuntimeError('leaf labels not unique')
            else:
                L2.add(v.label)
            
        if not L:
            L = L2
            if len(L) == 0:
                raise RuntimeError('empty tree in tree list')
        elif L != L2:
            return False
    
    return L