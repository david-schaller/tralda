# -*- coding: utf-8 -*-

"""
Heuristic for cograph editing.

References
----------
.. [1] Christophe Crespelle.
   Linear-Time Minimal Cograph Editing.
   Preprint published 2019.
"""

import random
import networkx as nx

from tralda.datastructures.Tree import Tree, TreeNode


__author__ = 'David Schaller'


def edit_to_cograph(G, run_number=10):
    """Heuristic algorithm for optimal cograph editing.
    
    Time complexity O(|V|^2).
    
    Parameters
    ----------
    G : nexworkx.Graph
        A graph.
    run_number : Number of editing runs from which the best editing result is
        returned.
    
    Returns
    -------
    Tree
        The cotree, i.e., a Tree instance with inner vertex labels 'series'
        and 'parallel', that corresponds to the best editing result.
    
    References
    ----------
    .. [1] Christophe Crespelle.
       Linear-Time Minimal Cograph Editing.
       Preprint published 2019.
    """
    
    ce = CographEditor(G)
    best_cotree = ce.cograph_edit()
    
    return best_cotree.to_cograph()


class CographEditor:
    """Heuristic for cograph editing running in O(|V|^2)..
    
    References
    ----------
    .. [1] Christophe Crespelle.
       Linear-Time Minimal Cograph Editing.
       Preprint published 2019.
    """
    
    def __init__(self, G):
        """Constructor of CographEditor class.
        
        Parameters
        ----------
        G : networkx.Graph
            A graph.
        """
        
        if not isinstance(G, nx.Graph):
            raise TypeError('not a NetworkX Graph')
        
        self.G = G
        self.V = [v for v in G.nodes()]
        
        self.cotrees = []
        self.costs = []
        
        self.best_T = None
        self.best_cost = float('inf')
        
    
    def cograph_edit(self, run_number=10):
        """Run the cograph editing.
        
        Parameters
        ----------
        run_number : Number of editing runs from which the best editing result is
            returned.
        
        Returns
        -------
        Tree
            The cotree, i.e., a Tree instance with inner vertex labels 'series'
            and 'parallel', that corresponds to the best editing result.
        """
        
        for i in range(run_number):
            
            T = Tree(None)
            self.aux_counter = {}
            already_in_T = set()
            leaf_map = {}
            total_cost = 0
            
            # shuffle the vertex list (beginning from second run)
            if i > 0:
                random.shuffle(self.V)
            
            # build starting tree for the first two vertices
            self._start_tree(T, already_in_T, leaf_map)
            
            # incrementally insert vertices while updating total cost
            if len(self.V) > 2:
                for x in self.V[2:]:
                    cost, x_node = self._insert(x, T, already_in_T, leaf_map)
                    
                    # update the number of leaves on the path to the root
                    current = x_node.parent
                    while current:
                        self.aux_counter[current] += 1
                        current = current.parent
                    
                    leaf_map[x] = x_node
                    already_in_T.add(x)
                    total_cost += cost
                
            self.cotrees.append(T)
            self.costs.append(total_cost)
            
            # update the best cost
            if total_cost < self.best_cost:
                self.best_T = T
                self.best_cost = total_cost
            
            # stop if orginal graph was a cograph, i.e. cost is 0
            if self.best_cost <= 0:
                break
            
        return self.best_T
        
    
    def _start_tree(self, T, already_in_T, leaf_map):
        
        if len(self.V) == 0:
            raise RuntimeError('empty graph in cograph editing')
            return
        
        elif len(self.V) == 1:
            T.root = TreeNode(label=self.V[0])
            return
        
        v1, v2 = self.V[0], self.V[1]
        already_in_T.update([v1, v2])
        
        R = TreeNode(label='series')
        self.aux_counter[R] = 2
        T.root = R
        
        if self.G.has_edge(v1, v2):
            v1_node = TreeNode(label=v1)
            self.aux_counter[v1_node] = 1
            v2_node = TreeNode(label=v2)
            self.aux_counter[v2_node] = 1
            R.add_child(v1_node)
            R.add_child(v2_node)
        else:
            N = TreeNode(label='parallel')
            self.aux_counter[N] = 2
            R.add_child(N)
            v1_node = TreeNode(label=v1)
            self.aux_counter[v1_node] = 1
            v2_node = TreeNode(label=v2)
            self.aux_counter[v2_node] = 1
            N.add_child(v1_node)
            N.add_child(v2_node)
            
        leaf_map[v1] = v1_node
        leaf_map[v2] = v2_node
    
        
    def _insert(self, x, T, already_in_T, leaf_map):
        
        # information for non-hollow nodes
        C_nh = {}                   # node u: list of non-hollow children
        C_nh_number = {}            # node u: number of non-hollow children
        
        Nx_number = {}              # node u: number of neighbors of x in V(u)
        completion_forced = {}      # node u: completion-forced or not
        full = {}                   # node u: full or not (i.e. mixed)
        deletion_forced = {}        # node u: deletion-forced or not
        
        Nx_counter = 0
        
        # ---- FIRST STEP ----
        
        # first traversal (C_nh)
        for v in self.G.neighbors(x):
            if v not in already_in_T:
                continue
            
            Nx_counter += 1
            
            v = leaf_map[v]
            C_nh[v]                = []
            C_nh_number[v]         = 0
            Nx_number[v]           = 1
            completion_forced[v]   = True
            full[v]                = True
            deletion_forced[v]     = False
            
            current = v
            while current.parent:
                if current.parent in C_nh:
                    C_nh[current.parent].append(current)
                    break                               # rest of the path is already done
                else:
                    C_nh[current.parent] = [current]
                current = current.parent                # continue path to root
                
        # all nodes in T are adjacent to x
        if self.aux_counter[T.root] == Nx_counter:
            R = T.root
            x_node = TreeNode(label=x)
            self.aux_counter[x_node] = 1
            R.add_child(x_node)
            leaf_map[x] = x_node
            return 0, x_node                            # cost is 0
        # no nodes in T are adjacent to x
        elif Nx_counter == 0:
            # d(R)=1
            if len(T.root.children) == 1:
                N = T.root.children[0]
                x_node = TreeNode(label=x)
                self.aux_counter[x_node] = 1
                N.add_child(x_node)
            else:
                R_old = T.root
                R_new = TreeNode(label='series')
                self.aux_counter[R_new] = self.aux_counter[R_old]
                N = TreeNode(label='parallel')
                self.aux_counter[N] = self.aux_counter[R_old]
                R_new.add_child(N)
                N.add_child(R_old)
                T.root = R_new
                
                x_node = TreeNode(label=x)
                self.aux_counter[x_node] = 1
                N.add_child(x_node)
            return 0, x_node                            # cost is 0
                
        # second traversal, postorder
        # (C_nh_number, Nx_number, completion_forced, mixed, deletion_forced)
        stack = [T.root]
        while stack:
            u = stack.pop()
            if not u.children:
                continue
            elif u not in C_nh_number:
                stack.append(u)
                C_nh_number[u] = len(C_nh[u])
                for nh_child in C_nh[u]:
                    stack.append(nh_child)
            else:
                completion_forced_counter = 0
                deletion_forced_counter = 0
                full_counter = 0
                Nx_number[u] = 0
                for nh_child in C_nh[u]:
                    Nx_number[u] += Nx_number[nh_child]
                    if completion_forced[nh_child]:
                        completion_forced_counter += 1
                    if deletion_forced[nh_child]:
                        deletion_forced_counter += 1
                    if full[nh_child]:
                        full_counter += 1
                full[u] = True if Nx_number[u] == self.aux_counter[u] else False
                
                # u completion-forced?
                if (full[u] or
                    (u.label == 'parallel' and len(u.children) == C_nh_number[u]) or
                    (u.label == 'series' and len(u.children) == completion_forced_counter)):
                    completion_forced[u] = True
                else:
                    completion_forced[u] = False
                    
                # u deletion-forced?
                if ((u.label == 'series' and full_counter == 0) or
                    (u.label == 'parallel' and C_nh_number[u] == deletion_forced_counter)):
                    deletion_forced[u] = True
                else:
                    deletion_forced[u] = False
                    
        # ---- SECOND STEP ----
        cost_above = {}                     # node u: cost above u
        C_mixed = {}                        # node u: mixed children of u
        C_full = {}                         # node u: full children of u
        
        stack = [T.root]
        while stack:        
            u = stack.pop()
            C_mixed[u] = []
            C_full[u] = []
            for nh_child in C_nh[u]:
                if not full[nh_child]:      # only mixed nodes are considered
                    stack.append(nh_child)
                    C_mixed[u].append(nh_child)
                else:
                    C_full[u].append(nh_child)
            v = u.parent
            if not v:                       # root has no parent
                cost_above[u] = 0
            elif u.parent.label == 'parallel':
                cost_above[u] = cost_above[v] + Nx_number[v] - Nx_number[u]
            else:
                cost_above[u] = cost_above[v] + ((self.aux_counter[v] - Nx_number[v]) - 
                                                 (self.aux_counter[u] - Nx_number[u]))
        
        mincost = {}
        
        for u in cost_above.keys():         # contains exactly the mixed nodes
            
            # apply Lemma 7 directly if u has exactly 2 children
            if len(u.children) == 2 and u.label == 'parallel':
                for i in range(2):
                    v, v2 = u.children[i], u.children[-1-i]
                    if v in completion_forced and completion_forced[v]:
                        Nx_v2 = Nx_number[v2] if (v2 in Nx_number) else 0
                        cost = cost_above[u] + (self.aux_counter[v] - Nx_number[v]) + Nx_v2
                        if (u not in mincost) or (cost < mincost[u][0]):
                            mincost[u] = (cost, [v])
            elif len(u.children) == 2 and u.label == 'series':
                for i in range(2):
                    v, v2 = u.children[i], u.children[-1-i]
                    if v not in deletion_forced or deletion_forced[v]:
                        Nx_v = Nx_number[v] if (v in Nx_number) else 0
                        Nx_v2 = Nx_number[v2] if (v2 in Nx_number) else 0
                        cost = cost_above[u] + Nx_v + (self.aux_counter[v2] - Nx_v2)
                        if (u not in mincost) or (cost < mincost[u][0]):
                            mincost[u] = (cost, [v2])
            
            # u has at least 3 children
            elif len(u.children) >= 3:
                red = set()
                blue = set()
                
                if u.label == 'parallel':                                   # u is a parallel node
                    # (step 0)
                    if C_nh_number[u] == 1:                                 # clean child (single non-hollow)
                        v = C_nh[u][0]                                      # v is the single nh child of u
                        if completion_forced[v]:
                            cost = cost_above[u] + (self.aux_counter[v] - Nx_number[v])
                            mincost[u] = (cost, [v])
                        continue                                            # otherwise u is not a minimal
                                                                            # insertion node (Lemma 8)
                    
                    # first step
                    for v in C_mixed[u]:
                        if Nx_number[v] >= self.aux_counter[v] - Nx_number[v]:
                            red.add(v)
                        else:
                            blue.add(v)
                    
                    # second step
                    if C_nh_number[u] == len(u.children) and not blue:
                        current_min, current_min_node = float('inf'), None
                        for v in red:
                            # Nx_number[v] - (self.aux_counter[v] - Nx_number[v])
                            diff = 2 * Nx_number[v] - self.aux_counter[v]
                            if diff < current_min:
                                current_min, current_min_node = diff, v
                        red.remove(current_min_node)
                        blue.add(current_min_node)
                        
                    # third step
                    nb_filled = len(C_full[u]) + len(red)
                    if nb_filled < 2:
                        for i in range(2 - nb_filled):
                            current_min, current_min_node = float('inf'), None
                            for v in blue:
                                # (self.aux_counter[v] - Nx_number[v]) - Nx_number[v]
                                diff = self.aux_counter[v] - 2 * Nx_number[v]
                                if diff < current_min:
                                    current_min, current_min_node = diff, v
                            blue.remove(current_min_node)
                            red.add(current_min_node)
                else:                                                       # u is a series node
                    # (step 0)
                    if len(u.children) - len(C_full[u]) == 1:               # clean child (single non-full)
                        v = None
                        for child in u.children:
                            if (child not in full) or (not full[child]):
                                v = child                                   # v is the single non-full child of u
                                break
                        if (v not in deletion_forced) or (deletion_forced[v]):
                            Nx_v = Nx_number[v] if v in Nx_number else 0
                            cost = cost_above[u] + Nx_v
                            mincost[u] = (cost, C_full[u])
                        continue                                            # otherwise u is not a minimal
                                                                            # insertion node (Lemma 8)
                                                                            
                    # first step
                    for v in C_mixed[u]:
                        if self.aux_counter[v] - Nx_number[v] >= Nx_number[v]:
                            blue.add(v)
                        else:
                            red.add(v)
                    
                    # second step
                    if not C_full[u] and not red:
                        current_min, current_min_node = float('inf'), None
                        for v in blue:
                            # (self.aux_counter[v] - Nx_number[v]) - Nx_number[v]
                            diff = self.aux_counter[v] - 2 * Nx_number[v]
                            if diff < current_min:
                                current_min, current_min_node = diff, v
                        blue.remove(current_min_node)
                        red.add(current_min_node)
                        
                    # third step
                    nb_hollowed = len(u.children) - C_nh_number[u] + len(blue)  # number of hollow or blue children
                    if nb_hollowed < 2:
                        for i in range(2 - nb_hollowed):
                            current_min, current_min_node = float('inf' ), None
                            for v in red:
                                # Nx_number[v] - (self.aux_counter[v] - Nx_number[v])
                                diff = 2 * Nx_number[v] - self.aux_counter[v]
                                if diff < current_min:
                                    current_min, current_min_node = diff, v
                            red.remove(current_min_node)
                            blue.add(current_min_node)
                    
                cost = cost_above[u]
                for v in red:
                    cost += (self.aux_counter[v] - Nx_number[v])
                for v in blue:
                    cost += Nx_number[v]
                    
                mincost[u] = (cost, C_full[u] + list(red))
        
        # determine a minimum cost settling
        insertion_mincost, settling_node = float('inf'), None
        for u in mincost.keys():
            if mincost[u][0] < insertion_mincost:
                insertion_mincost, settling_node = mincost[u][0], u
        
        # insert x into tree
        u = settling_node           # node under which x is inserted
        filled = mincost[u][1]      # children of u to be filled (marked nodes in LinearCographDetector)
        
        # parallel node where one child is to be filled
        if u.label == 'parallel' and len(filled) == 1:
            w = filled[0]
            if w.is_leaf():
                new_node = TreeNode(label='series')  # leaves w and x (x added later)
                self.aux_counter[new_node] = 1
                u.remove_child(w)
                u.add_child(new_node)
                new_node.add_child(w)
                
                x_node = TreeNode(label=x)
                self.aux_counter[x_node] = 1
                new_node.add_child(x_node)
            else:
                x_node = TreeNode(label=x)
                self.aux_counter[x_node] = 1
                w.add_child(x_node)
        
        # series node and only one child is not to be filled
        elif (u.label == 'series' and 
              len(u.children) - len(filled) == 1):
            set_A = set(filled)       # auxiliary set
            w = None
            for child in u.children:
                if child not in set_A:
                    w = child
                    break
            if w.is_leaf():
                new_node = TreeNode(label='parallel')  # leaves w and x (x added later)
                self.aux_counter[new_node] = 1
                u.remove_child(w)
                u.add_child(new_node)
                new_node.add_child(w)
                
                x_node = TreeNode(label=x)
                self.aux_counter[x_node] = 1
                new_node.add_child(x_node)
            else:
                x_node = TreeNode(label=x)
                self.aux_counter[x_node] = 1
                w.add_child(x_node)
        
        else:
            filled_aux_counter = 0
            y = TreeNode(label=u.label)
            for a in filled:
                u.remove_child(a)
                y.add_child(a)
                filled_aux_counter += self.aux_counter[a]
            self.aux_counter[y] = filled_aux_counter
                
            if u.label == 'parallel':
                new_node = TreeNode(label='series')
                self.aux_counter[new_node] = filled_aux_counter
                u.add_child(new_node)
                
                new_node.add_child(y)
                x_node = TreeNode(label=x)
                self.aux_counter[x_node] = 1
                new_node.add_child(x_node)
            else:
                par = u.parent
                if par is not None:             # u was the root of T
                    par.remove_child(u)
                    par.add_child(y)
                else:
                    T.root = y                  # y becomes the new root
                
                new_node = TreeNode(label='parallel')
                self.aux_counter[new_node] = 0
                y.add_child(new_node)
                new_node.add_child(u)
                x_node = TreeNode(label=x)
                self.aux_counter[x_node] = 1
                new_node.add_child(x_node)
                
                # update the leaf numbers
                self.aux_counter[y] = self.aux_counter[u]
                self.aux_counter[u] -= filled_aux_counter
                self.aux_counter[new_node] = self.aux_counter[u]

        return insertion_mincost, x_node