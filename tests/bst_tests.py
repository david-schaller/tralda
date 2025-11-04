from __future__ import annotations

import unittest

import numpy as np

from tralda.datastructures.bst.simple import BinarySearchTree
from tralda.datastructures.bst.avl import TreeSet


class TestTrees(unittest.TestCase):
    test_sequence = np.random.default_rng(seed=0).permutation(10000)
    to_remove = [99, 7890, 0, 567, 1234, 4567, 345, 9876]
    reference_after_removal = sorted(set(test_sequence) - set(to_remove))

    def _insertion_and_removal(self, tree_class):
        # new instance
        tree = tree_class()

        for x in self.test_sequence:
            tree.add(x)

        # copy tree to make sure copying works
        tree = tree.copy()

        for x in self.to_remove:
            tree.remove(x)

        self.assertEqual(len(tree), len(self.reference_after_removal))

        self.assertEqual(self.reference_after_removal, [x for x in tree])

    def test_simple_search_tree(self):
        self._insertion_and_removal(BinarySearchTree)

    def test_avl_tree(self):
        self._insertion_and_removal(TreeSet)


if __name__ == "__main__":
    unittest.main()
