from __future__ import annotations

import unittest
import os
import random

from tralda.datastructures import Tree


example_tree = (
    "((((14,15,(27,(29,30)28)18)10,(19,20)11,16)7,8,9,17)1,(4,5)2,3,((23,24)12,13,22)6,(25,26)21)0;"
)


class TestTrees(unittest.TestCase):
    def test_basic_properties(self):
        tree = Tree.parse_newick(example_tree)

        self.assertEqual(len(tree), 31)
        self.assertEqual(tree.height(), 6)

    def test_integrity(self):
        tree = Tree.random_tree(20)

        self.assertTrue(tree._assert_integrity())

    def test_traversal(self):
        tree = Tree.random_tree(20)
        pre = {v for v in tree.preorder()}
        post = {v for v in tree.postorder()}

        self.assertTrue(pre == post)

    def test_newick(self):
        tree = Tree.random_tree(20)

        for v in tree.preorder():
            v.dist = random.random()

        newick = tree.to_newick()

        tree2 = Tree.parse_newick(newick)

        self.assertTrue(tree.equal_topology(tree2))

        for v, v2 in zip(tree.preorder(), tree2.preorder()):
            self.assertEqual(v.label, v2.label)
            self.assertTrue(v.dist - v2.dist <= 1e-6)

    def test_serialization(self):
        tree = Tree.random_tree(50)

        tree.serialize("temp_serialized.pickle")
        tree.serialize("temp_serialized.json")

        tree_p = tree.load("temp_serialized.pickle")
        tree_j = tree.load("temp_serialized.json")

        os.remove("temp_serialized.pickle")
        os.remove("temp_serialized.json")

        # print('tree', tree.to_newick())
        # print('tree_p', tree_p.to_newick())
        # print('tree_j', tree_j.to_newick())

        self.assertTrue(tree.equal_topology(tree_p) and tree.equal_topology(tree_j))

    def test_print_tree(self):
        # indentation was set to 4
        reference = [
            "0",
            "├───1",
            "│   ├───7",
            "│   │   ├───10",
            "│   │   │   ├───14",
            "│   │   │   ├───15",
            "│   │   │   └───18",
            "│   │   │       ├───27",
            "│   │   │       └───28",
            "│   │   │           ├───29",
            "│   │   │           └───30",
            "│   │   ├───11",
            "│   │   │   ├───19",
            "│   │   │   └───20",
            "│   │   └───16",
            "│   ├───8",
            "│   ├───9",
            "│   └───17",
            "├───2",
            "│   ├───4",
            "│   └───5",
            "├───3",
            "├───6",
            "│   ├───12",
            "│   │   ├───23",
            "│   │   └───24",
            "│   ├───13",
            "│   └───22",
            "└───21",
            "    ├───25",
            "    └───26",
        ]

        tree = Tree.parse_newick(example_tree)

        self.assertEqual(tree._lines_for_print_tree(4), reference)


if __name__ == "__main__":
    unittest.main()
