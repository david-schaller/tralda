# -*- coding: utf-8 -*-

"""
Euler tour tree as AVL-tree.
    
Implementation of an Euler tour tree for dynamic graph algorithm. A version
of AVL-trees is used as balanced binary tree.

References:
    - Monika Rauch Henzinger, Valerie King. Randomized fully dynamic graph 
      algorithms with polylogarithmic time per operation. J. ACM 46(4) 
      July 1999. 502–536.
"""

from tralda.datastructures.DoublyLinkedList import DLList


__author__ = 'David Schaller'


class DGNode:
    """Dynamic graph node."""
    
    __slots__ = ['value', 'tree_edges', 'nontree_edges',
                 'active_occ', 'occurrences']
    
    def __init__(self, value, active_occ=None):
        
        self.value = value
        self.tree_edges = DLList()
        self.nontree_edges = DLList()
        self.active_occ = active_occ
        
        # list of occurrences (of this node) in the Euler tour
        self.occurrences = DLList()
        if active_occ:
            active_occ.occ = self.occurrences.append(active_occ)


class ETTreeNode:
    
    __slots__ = ('value', 'parent', 'left', 'right', 'prev_occ', 'next_occ',
                 'height', 'size', 'active', 'occ', 'ett')
    
    def __init__(self, value, prev_occ=None, next_occ=None,
                 active=False, ett=None):
        
        self.value = value
        self.parent = None
        self.left = None
        self.right = None
        
        self.prev_occ = prev_occ            # reference to previous occurence
        self.next_occ = next_occ            # reference to next occurence
        
        self.height = 1                     # height of the subtree --> for rebalancing
        self.size = 1 if active else 0      # stores number of active occurences in its subtree
        
        self.active = 1 if active else 0    # active occurrence?
        self.occ = None                     # reference to list entry in occurrences
        
        self.ett = None                     # root stores a reference to the ET tree
   
    
    def get_balance(self):
        if self.left and self.right:
            return self.left.height - self.right.height
        elif self.left:
            return self.left.height
        elif self.right:
            return -(self.right.height)
        else:
            return 0
    
    
    def get_root(self):
        current = self
        while current.parent:
            current = current.parent
        return current
    
    
    def __str__(self):
        return "Occ_" + str(self.value)
 

class ETTree:
    """Euler tour tree.
    
    Represents an Euler tour of a (spanning) tree as a balanced binary tree,
    which is here implemented as an AVL-tree.
    """
    
    __slots__ = ('root', 'nodedict', 'start', 'end', 'current_occ')
    
    def __init__(self, root=None, nodedict=None, start=None, end=None):
        
        self.root = root                            # root of the ET-Tree
        if self.root:
            self.root.ett = self
        if nodedict is not None:
            self.nodedict = nodedict                # hash of active Treenodes
        else:
            self.nodedict = dict()
        self.start = start
        self.end = end
    
    
    def __iter__(self):
        
        self.current_occ = self.start
        return self


    def __next__(self):
        
        if self.current_occ:
            x = self.current_occ
            self.current_occ = self.current_occ.next_occ
            return x
        else:
            raise StopIteration
      
    
    def __len__(self):
        
        if self.root:
            return self.root.size
        else:
            return 0
    
    
    def __contains__(self, item):
        
        if item not in self.nodedict:
            return False
        root = self.nodedict[item].active_occ.get_root()
        return root is self.root
    
    
    def get_size(self):
        
        if self.root:
            return self.root.size
        else:
            return 0
    
    
    def insert(self, value):
        
        if not self.root:
            self.root = ETTreeNode(value)
            return
        
        self._insert(value, self.root)
        self.root.ett = self
        
    
    def _insert(self, value, node):
        
        if value == node.value:
            return
        elif value < node.value:
            if node.left:
                self._insert(value, node.left)
            else:
                node.left = ETTreeNode(value, active=True)
                node.left.parent = node
                self.nodedict[value] = DGNode(value, active_occ=node.left)
                self.root = self.rebalance(node)
        else:
            if node.right:
                self._insert(value, node.right)
            else:
                node.right = ETTreeNode(value, active=True)
                node.right.parent = node
                self.nodedict[value] = DGNode(value, active_occ=node.right)
                self.root = self.rebalance(node)
    
    
    def delete_node(self, node, update_refs=False):
        
        if node.left and node.right:
            subst = self.smallest_in_subtree(node.right)    # replace by smallest
            to_rebalance = subst.parent                     # in right subtree
            if to_rebalance is node:
                to_rebalance = subst
            else:
                to_rebalance.left = subst.right
                if subst.right:
                    subst.right.parent = to_rebalance
                subst.right = node.right
                node.right.parent = subst
            subst.left = node.left
            node.left.parent = subst
            par = node.parent
            subst.parent = par
            if par:
                if node is par.left:
                    par.left = subst
                elif node is par.right:
                    par.right = subst
            else:
                self.root = subst
            self.root = self.rebalance(to_rebalance)
        elif node.left:
            par = node.parent
            node.left.parent = par
            if par:
                if node is par.left:
                    par.left = node.left
                elif node is par.right:
                    par.right = node.left
                self.root = self.rebalance(par)
            else:
                self.root = node.left
        elif node.right:
            par = node.parent
            node.right.parent = par
            if par:
                if node is par.left:
                    par.left = node.right
                elif node is par.right:
                    par.right = node.right
                self.root = self.rebalance(par)
            else:
                self.root = node.right
        else:
            par = node.parent
            if par:
                if node is par.left:
                    par.left = None
                elif node is par.right:
                    par.right = None
                self.root = self.rebalance(par)
            else:
                self.root = None
        self.root.ett = self
        node.parent, node.left, node.right = None, None, None   
        
        # update of prev./next references and active occurrence
        if update_refs:
            if node.prev_occ:                           # update occ. links
                node.prev_occ.next_occ = node.next_occ
            if node.next_occ:
                node.next_occ.prev_occ = node.prev_occ
            if node is self.start:
                self.start = node.next_occ
            if node is self.end:
                self.end = node.prev_occ
            
            active = self.nodedict[node.value]
            active.occurrences.remove_node(node.occ)
            if node.active:                             # was it the active occurrence?
                node.active = 0
                replace = active.occurrences[0]         # (arbitrary) new active node
                if replace:
                    replace.active = 1
                    self.nodedict[node.value].active_occ = replace
                    self.update_height_size(replace, propagate=True)
                else:                                   # no replacement found
                    print("could not find a replacement node "\
                          "for '{}'".format(node))
                    del self.nodedict[node.value]
            
    
    @staticmethod
    def initialize_from_tree(tree, nodedict=None):
        """Initialize the ET tree from an Euler tour of a rooted tree."""
        
        if not tree:
            return
        
        ett = ETTree(nodedict=nodedict)
        previous = None
        for occ in tree.euler_generator():
            if occ in ett.nodedict:
                if ett.nodedict[occ].active_occ:
                    ett_node = ETTreeNode(occ, active=False, prev_occ=previous)
                    ett_node.occ = ett.nodedict[occ].occurrences.append(ett_node)
                else:
                    ett_node = ETTreeNode(occ, active=True, prev_occ=previous)
                    ett.nodedict[occ].active_occ = ett_node
                    ett_node.occ = ett.nodedict[occ].occurrences.append(ett_node)
            else:
                ett_node = ETTreeNode(occ, active=True, prev_occ=previous)
                ett.nodedict[occ] = DGNode(occ, active_occ=ett_node)
            if previous:
                previous.next_occ = ett_node
            else:
                ett.start = ett_node
            previous = ett_node
            ett.rightinsert(ett_node)
        ett.end = previous
        return ett

    
    def rightinsert(self, new_node):
        """Insert new element that is greater than all previous elements."""
        
        if not self.root:
            self.root = new_node
            self.root.ett = self
            return
        current_node = self.root
        while current_node.right:
            current_node = current_node.right
        current_node.right = new_node
        new_node.parent = current_node
        self.root = self.rebalance(current_node)
        self.root.ett = self
        self.end = new_node
    
    
    def rebalance(self, node, stop=None):
        
        while node:
            self.update_height_size(node)
            balance = node.get_balance()
            while abs(balance) > 1:
                if balance > 1:
                    if node.left.get_balance() >= 0:    # single rotation needed
                        self.rightrotate(node)
                    else:                               # double rotation needed
                        self.leftrotate(node.left)
                        self.rightrotate(node)
                else:
                    if node.right.get_balance() <= 0:   # single rotation needed
                        self.leftrotate(node)
                    else:                               # double rotation needed
                        self.rightrotate(node.right)
                        self.leftrotate(node)
                balance = node.get_balance()
            if (not node.parent) or (node.parent is stop):
                return node
            node = node.parent
            
            
    def rightrotate(self, y):
        
        x = y.left
        B = x.right
        if y.parent and (y is y.parent.right):
            y.parent.right = x
        elif y.parent and (y is y.parent.left):
            y.parent.left = x
        x.parent = y.parent
        y.parent = x
        x.right = y
        y.left = B
        if B:
            B.parent = y
        self.update_height_size(y)
        self.update_height_size(x)
        
        
    def leftrotate(self, x):
        
        y = x.right
        B = y.left
        if x.parent and (x is x.parent.right):
            x.parent.right = y
        elif x.parent and (x is x.parent.left):
            x.parent.left = y
        y.parent = x.parent
        x.parent = y
        y.left = x
        x.right = B
        if B:
            B.parent = x
        self.update_height_size(x)
        self.update_height_size(y)
        
    
    def update_height_size(self, node, propagate=False):
        
        if not node:
            return
        if node.left and node.right:
            node.height = 1 + max(node.left.height, node.right.height)
            node.size = node.active + node.left.size + node.right.size
        elif node.left:
            node.height = 1 + node.left.height
            node.size = node.active + node.left.size
        elif node.right:
            node.height = 1 + node.right.height
            node.size = node.active + node.right.size
        else:
            node.height = 1
            node.size = node.active
        if propagate:                       # update the path to the root
            self.update_height_size(node.parent, propagate=True)
    
    
    def get_root(self, value):
        
        if value not in self.nodedict:
            print("could not find node '{}'".format(value))
            return
        return self.nodedict[value].active_occ.get_root()
        
    
    def smallest_in_subtree(self, node):
        
        current = node
        while current.left:
            current = current.left
        return current
    
    
    def biggest_in_subtree(self, node):
        
        current = node
        while current.right:
            current = current.right
        return current
    
    
    # obsolete
    def get_first_smaller(self, node):
        
        first_smaller = None
        if node.left:
            current = node.left
            while current.right:
                current = current.right
            first_smaller = current
        elif node.parent:
            current = node
            while (current.parent and current.parent.left and 
                   current.parent.left == current):
                current = current.parent
            if (current.parent and current.parent.right and 
                current.parent.right == current):
                first_smaller = current.parent
        return first_smaller
    
    
    # obsolete
    def get_first_greater(self, node):
        
        first_greater = None
        if node.right:
            current = node.right
            while current.left:
                current = current.left
            first_greater = current
        elif node.parent:
            current = node
            while (current.parent and current.parent.right and 
                   current.parent.right == current):
                current = current.parent
            if (current.parent and current.parent.left and 
                current.parent.left == current):
                first_greater = current.parent
        return first_greater
    
    
    def smaller(self, node1, node2):
        """Determines whether occurence 'node1' is smaller than 'node2' with
        respect to the Euler tour."""
        
        if node1 is node2:
            return False
        path1, path2 = [node1], [node2]
        while node1.parent:
            path1.append(node1.parent)
            node1 = node1.parent
        while node2.parent:
            path2.append(node2.parent)
            node2 = node2.parent
        if path1[-1] is not path2[-1]:
            return False
        
        for i in range(-2,-min(len(path1),len(path2))-1,-1):
            if path1[i] is not path2[i]:
                if path1[i] is path1[i].parent.left:
                    return True
                else:
                    return False
        if len(path1) < len(path2):
            if path2[-len(path1)-1] is path1[0].left:
                return False
            else: 
                return True
        elif len(path2) < len(path1):
            if path1[-len(path2)-1] is path2[0].left:
                return True
            else:
                return False
    
    
    def lca(self, node1, node2):
        """Finds the last common ancestor of 2 nodes."""
        
        if node1 is node2:
            return node1
        path1, path2 = [node1], [node2]
        while node1.parent:
            path1.append(node1.parent)
            node1 = node1.parent
        while node2.parent:
            path2.append(node2.parent)
            node2 = node2.parent
        if path1[-1] is not path2[-1]:
            return None
        else:
            lca = path1[-1]
        for i in range(-2,-min(len(path1),len(path2))-1,-1):
            if path1[i] is path2[i]:
                lca = path1[i]
            else:
                break
        return lca
    
    
    def cut(self, node1_val, node2_val):
        
        # vertices equal or not in the tree
        if (node1_val == node2_val or (node1_val not in self.nodedict) or 
            (node2_val not in self.nodedict)):
            print("Could not find edge {" + str(node1_val) + "," + 
                  str(node2_val) + "} in the ET tree! (1)")
            return
        
        # which node is further from ET root --> v, the other node is u
        # --> determine u1, u2, v1, v2
        # search in the shorter occurence list as candidate for v
        u, v = self.nodedict[node1_val], self.nodedict[node2_val]
        if len(u.occurrences) < len(v.occurrences):
            v, u = u, v
        # now v.occurrences has less or equal elements
        
        u1, u2, v1, v2 = None, None, None, None
        for occ in v.occurrences:
            if occ.prev_occ and occ.prev_occ.value == u.value:
                u1, v1 = occ.prev_occ, occ
            if occ.next_occ and occ.next_occ.value == u.value:
                v2, u2 = occ, occ.next_occ
            if u1 and u2:
                break
        if not (u1 and u2):                     # search was not successful
            print("Could not find edge {" + str(node1_val) + "," + 
                  str(node2_val) + "} in the ET tree! (2)")
            return
        if u1 is u2 or self.smaller(u2, u1):    # order must be swapped
            u1, v1, v2, u2 = v2, u2, u1, v1
        
        # now u is enclosing v: ... (u1, v1) ... (v2, u2) ...
        # print(u1, v1, v2, u2)
        
        # trivial case: a single node is enclosed (v1 == v2):
        if v1 is v2:
            self.delete_node(v1)                # parameter update_refs is not
            self.update_height_size(v1)         # necessary here (see below)
            spliced_tree = v1
        # splicing-out of a whole interval:
        else:
            lca = self.lca(v1,v2)
            if not lca:
                print("Nodes v1 and v2 are not connencted!")
                return

            # left side handling
            rebalance1 = None
            subtree1 = v1.left
            subtree2 = v1
            if subtree1:
                v1.left = None
                rebalance1 = subtree1
                subtree1.parent = None
            
            # anything smaller between v1 and lca? --> walk from v1 to lca
            if lca is not v1:
                current = v1
                while current.parent and not (current.parent is lca):
                    if current is current.parent.right:
                        smaller = current.parent
                        subtree2 = smaller.right
                        subtree2.parent = None
                        smaller.right = subtree1
                        if subtree1:
                            subtree1.parent = smaller
                        else:
                            rebalance1 = smaller
                        greater = None
                        current = smaller
                        
                        # walk from smaller to lca to find next node
                        # greater than v1 (at least true for lca.left)
                        while current.parent:
                            if current is current.parent.left:
                                greater = current.parent
                                break
                            current = current.parent
                        if not greater:
                            print("Couldn't find node to attach subtree! (2)")
                            return
                        subtree1 = greater.left
                        subtree1.parent = None
                        greater.left = subtree2
                        subtree2.parent = greater
                        if subtree2.parent is lca:
                            break
                        else:
                            current = subtree2
                    else:
                        current = current.parent
            
            # right side handling
            rebalance4 = None
            subtree4 = v2.right
            subtree3 = v2
            if subtree4:
                v2.right = None
                rebalance4 = subtree4
                subtree4.parent = None
            
            # anything greater between v2 and lca? --> walk from v2 to lca
            if lca is not v2:
                current = v2
                while current.parent and not (current.parent is lca):
                    if current is current.parent.left:
                        greater = current.parent
                        subtree3 = greater.left
                        subtree3.parent = None
                        greater.left = subtree4
                        if subtree4:
                            subtree4.parent = greater
                        else:
                            rebalance4 = greater
                        smaller = None
                        current = greater
                        
                        # walk from greater to lca to find next node
                        # smaller than v2 (at least true for lca.right)
                        while current.parent:
                            if current is current.parent.right:
                                smaller = current.parent
                                break
                            current = current.parent
                        if not smaller:
                            print("Couldn't find node to attach subtree! (3)")
                            return
                        subtree4 = smaller.right
                        subtree4.parent = None
                        smaller.right = subtree3
                        subtree3.parent = smaller
                        if subtree3.parent is lca:
                            break
                        else:
                            current = subtree3
                    else:
                        current = current.parent
            
            # reattaching of the subtrees
            attachment_point = lca.parent
            if attachment_point and (attachment_point.left is lca):
                attach_left = True          # remember where to
            else:                           # attach the subtree
                attach_left = False
                
            # inner sequence (to be spliced out)
            lca.parent = None
            if lca is v1:
                spliced_tree = self.rebalance(v2)
            elif lca is v2:
                spliced_tree = self.rebalance(v1)
            else:
                self.rebalance(v2, stop=lca)
                spliced_tree = self.rebalance(v1)
            
            # fragments of outer sequence
            if subtree1 and subtree4:
                attachment_point1 = self.biggest_in_subtree(rebalance1)
                attachment_point1.right = subtree4
                subtree4.parent = attachment_point1
                subtree1 = self.rebalance(rebalance4)  # passes also rebalance1
            elif subtree1:
                subtree1 = self.rebalance(rebalance1)
            elif subtree4:
                subtree1 = self.rebalance(rebalance4)
            
            # re-ligate outer sequence fragments
            if attachment_point:
                if attach_left:
                    attachment_point.left = subtree1
                else:
                    attachment_point.right = subtree1
                if subtree1:
                    subtree1.parent = attachment_point
                self.root = self.rebalance(attachment_point)
                self.root.ett = self
            else:
                self.root = subtree1
                self.root.ett = self
                subtree1.parent = None

        # handling of the obsolete node u2 (possible active node and occ. links
        # are taken care of in function delete_node(x, update_refs=True) )
        self.delete_node(u2, update_refs=True)
        u1.next_occ = v2.next_occ                   # since u2 is already deleted
        if u1.next_occ:                             # might be None
            u1.next_occ.prev_occ = u1
        else:
            self.end = u1
        v1.prev_occ, v2.next_occ = None, None       # disconnect seq. [v1, v2]
        
        new_ett = ETTree(root=spliced_tree, nodedict=self.nodedict, start=v1, end=v2)
        new_ett.root.ett = new_ett
        
        return (self, new_ett)
    
    
    def reroot(self, value):
        """Reroot an Euler tour at a given value."""
        
        if self.get_root(value) is not self.root:
            print("Node ", value, "is not in this ET tree!")
            return
        old1 = self.smallest_in_subtree(self.root)
        
        # case 1: new root is the old root --> do nothing
        if value == old1.value:
            return self
        
        # case 2: rerooting is necessary
        new_root = self.nodedict[value].occurrences[0]
        if not new_root:
            print("Could not find any occurrence of the new root!")
            return
        second_occ = old1.next_occ                  # remember for later (re-ligation)
        self.delete_node(old1, update_refs=True)    # delete first occ. of the old root
        
        # construct two subsequences: 
        # subtree1 (..., new_root1) and subtree2 [new_root1, ...)
        subtree1 = new_root.left
        subtree2 = new_root
        rebalance1 = None
        if subtree1:
            subtree2.left = None
            subtree1.parent = None
            rebalance1 = subtree1
        # anything smaller on the path to the root?
        # --> walk from new_root to root
        current = new_root
        while current.parent:
            if current is current.parent.right:
                smaller = current.parent
                subtree2 = smaller.right
                subtree2.parent = None
                smaller.right = subtree1
                if subtree1:
                    subtree1.parent = smaller
                else:
                    subtree1 = smaller
                    rebalance1 = smaller
                greater = None
                current = smaller
                while current.parent:
                    if current is current.parent.left:
                        greater = current.parent
                        break
                    current = current.parent
                if greater:
                    subtree1 = greater.left
                    subtree1.parent = None
                    greater.left = subtree2
                    subtree2.parent = greater
                    subtree2 = greater
                    current = subtree2
                else:
                    break
            else:
                current = current.parent
        
        # re-ligate: subtree2 + subtree1 + new occ. of the new root
        subtree2 = self.rebalance(new_root)
        active = self.nodedict[new_root.value]
        new_occ = ETTreeNode(active.value)              # new occ. of the root
        new_occ.occ = active.occurrences.append(new_occ)
        if subtree1:
            subtree1 = self.rebalance(rebalance1)
            
            attachment_point1 = self.biggest_in_subtree(subtree1)
            attachment_point1.right = new_occ           # tree links
            new_occ.parent = attachment_point1
            new_occ.prev_occ = attachment_point1        # ET list links
            attachment_point1.next_occ = new_occ
            
            attachment_point2 = self.biggest_in_subtree(subtree2)
            attachment_point2.right = subtree1          # tree links
            subtree1.parent = attachment_point2
            second_occ.prev_occ = attachment_point2     # ET list links
            attachment_point2.next_occ = second_occ
        else:
            attachment_point = self.biggest_in_subtree(subtree2)
            attachment_point.right = new_occ            # tree links
            new_occ.parent = attachment_point
            new_occ.prev_occ = attachment_point         # ET list links
            attachment_point.next_occ = new_occ
            
        new_root.prev_occ = None
        self.root = self.rebalance(new_occ)
        self.root.ett = self
        self.start, self.end = new_root, new_occ
        
        return self
    
    
    @staticmethod
    def link(ett1, ett2, a, b):
        """Link two ET trees on two given nodes.
        
        Links 'ett1' and 'ett2' at node 'a' and 'b', respectively. Therefore,
        the tours are first rerooted and the merged (+ additional occurrence
        of either 'a' or 'b').
        """
        
        if not (isinstance(ett1, ETTree) and isinstance(ett2, ETTree)):
            raise TypeError("Parameters must be of type 'ETTree'!")
        ett1 = ett1.reroot(a)
        ett2 = ett2.reroot(b)
        if not (ett1 and ett2):
            print("Could not find nodes to link in the ET trees!")
            return
        if ett1.root.size < ett2.root.size:
            ett1, ett2 = ett2, ett1          # now ett2 is smaller
        ett1_last = ett1.end
        ett2_last = ett2.end
        ett2_first = ett2.start
        new_occ = ETTreeNode(ett1_last.value, prev_occ=ett2_last)
        new_occ.occ = ett1.nodedict[ett1_last.value].occurrences.append(new_occ)
        
        ett2_last.right = new_occ            # tree links
        new_occ.parent = ett2_last
        ett1_last.right = ett2.root
        ett2.root.parent = ett1_last
        
        ett2_last.next_occ = new_occ         # ET list links
        ett1_last.next_occ = ett2_first
        ett2_first.prev_occ = ett1_last
        
        new_root = ett1.rebalance(new_occ)
        new_ett = ETTree(root=new_root, nodedict=ett1.nodedict, 
                         start=ett1.start, end=new_occ)
        new_root.ett = new_ett
        
        return new_ett
                
    
    def to_newick(self):
        """Recursive Tree --> Newick (str) function.
        
        Intended for testing purpose.
        """
        
        def construct_newick(node):
            if not (node.left or node.right):
                return str(node.value)# + "-" + str(node.size)+ "-" + str(node.active)
            else:
                if node.left and node.right:
                    s = "(" + construct_newick(node.left) + "," + construct_newick(node.right) + ")"
                elif node.left:
                    s = "(" + construct_newick(node.left) + ",-)"
                elif node.right:
                    s = "(-," + construct_newick(node.right) + ")"
                else:
                    s = ""
                return s + str(node.value)# + "-" + str(node.size) + "-" + str(node.active)
            
        return construct_newick(self.root)
    
    
    def ET_to_list(self):
        """Build a list of the Euler tour.
        
        Intended for testing purpose. Should return the same result like
        the function ET_to_list_recursive().
        """
        
        result = []
        for occ in self:
            result.append(occ.value)
        return result
        
        
    def ET_to_list_recursive(self, node=None):
        """Build a list of the Euler tour (recursive).
        
        Intended for testing purpose.
        """
        
        if not node:
            s = self.ET_to_list_recursive(self.root)
        else:
            s = []
            if node.left:
                s += self.ET_to_list_recursive(node.left)
            s += [node.value]
            if node.right:
                s += self.ET_to_list_recursive(node.right)
        return s
    
    
    def check_integrity(self, node=None):
        """Recursive check of the following properties of the ET tree:
            
        - all children have a correct parent reference
        - the tree is balanced (AVL property)
        - the size (nr. of active occurences) is correct in all subtrees
        Intended for testing purpose.
        """
        
        if not node:
            if not self.root:
                print("Tree has no root!")
                return False
            else:
                return self.check_integrity(self.root)
        else:
            height_left, height_right = 0, 0
            size = node.active
            if node.left:
                if (node is not node.left.parent) or (not self.check_integrity(node.left)):
                    print("Check node:", node, "left")
                    return False
                height_left = node.left.height
                size += node.left.size
            if node.right:
                if (node is not node.right.parent) or (not self.check_integrity(node.right)):
                    print("Check node:", node, "right")
                    return False
                height_right = node.right.height
                size += node.right.size
            if abs(node.get_balance()) > 1:
                print("Node", node, "is unbalanced!")
                return False
            if node.height != 1 + max(height_left, height_right):
                print("Height of node", node, "is incorrect!")
                return False
            if node.size != size:
                print("Size of node", node, "is incorrect!")
                return False
            return True
