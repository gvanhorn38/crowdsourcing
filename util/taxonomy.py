"""
MIT License

Copyright (c) 2017 Grant Van Horn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import absolute_import, division, print_function
from collections import OrderedDict

import numpy as np


class Node(object):
    """ A node of the taxonomy.
    """

    def __init__(self, key, data, is_root=False):
        """
        Args:
            `key` needs to be unique for this node
            `data` is a generic data storage object (e.g. a dict {}). It needs
            to be json serializable
        """
        self.key = key
        self.data = data if data is not None else {}
        self.is_root = is_root

        self.parent = None
        self.children = OrderedDict()  # {key : Node}

        # Set in `finalize`
        self.is_leaf = None
        self.level = None
        self.ancestors = None
        self.order = None

    def finalize(self):
        """ Compute useful attributes.
        """

        self.is_leaf = len(self.children) == 0

        # Compute our level
        if self.parent is None:
            self.level = 0
        else:
            self.level = self.parent.level + 1

        # Compute our ancestors
        ancestors = []
        parent = self.parent
        while parent is not None:
            ancestors.append(parent)
            parent = parent.parent
        self.ancestors = ancestors

    def add_child(self, node):
        self.children[node.key] = node
        node.parent = self
        node.order = len(self.children) - 1

    def jsonify_data(self):
        """ jsonify our data, specifically handling numpy arrays.
        """
        json_data = {}
        for key, value in self.data.iteritems():
            if isinstance(value, np.ndarray):
                value = value.tolist()
            json_data[key] = value
        return json_data

    def __eq__(self, other):
        return self.key == other.key

    def __ne__(self, other):
        return self.key != other.key


class Taxonomy(object):
    """ A static taxonomy structure.
    """

    def __init__(self):

        self.nodes = OrderedDict()  # {key : Node}
        self.root_node = None
        self.finalized = False

        # precomputed return values
        # Computed after we are finalized
        self.num_leaf_nodes = 0
        self.num_inner_nodes = 0
        self.node_level_map = None  # {key : {level : key}}
        self._leaf_nodes = None
        self._inner_nodes = None
        self._breadth_first_traversal = None

        self.max_depth = None

    def _make_level_map(self):
        """ Make a dict that maps a leaf node and a level value to the ancestor
        node of the leaf node at that level.
        """

        node_level_map = {}
        for node in self.breadth_first_traversal():
            node_level_map[(node.key, node.level)] = node
            for ancestor in node.ancestors:
                node_level_map[(node.key, ancestor.level)] = ancestor
        self.node_level_map = node_level_map

    def finalize(self):
        """ Compute useful attributes assuming all nodes have been added.
        """

        levels = []
        for node in self.breadth_first_traversal():
            node.finalize()
            levels.append(node.level)
        self.max_depth = max(levels)

        self._make_level_map()
        self.num_leaf_nodes = len(self.leaf_nodes())
        self.num_inner_nodes = len(self.inner_nodes())

        self.finalized = True

    def add_node_from_data(self, node_data):
        """ Create a node from the node data.
        """

        if self.finalized:
            raise ValueError("Taxonomy is already finalized.")

        node = Node(node_data['key'], node_data['data'])
        parent_key = node_data['parent']

        # Assign the parent
        if parent_key is not None:
            parent = self.nodes[parent_key]
            parent.add_child(node)
        else:
            node.is_root = True

        self.add_node(node)


    def add_node(self, node):
        if self.finalized:
            raise ValueError("Taxonomy has been finalized.")

        self.nodes[node.key] = node
        if node.is_root:
            if self.root_node is not None:
                raise ValueError("Root node already specified.")
            self.root_node = node

    def node_at_level_from_node(self, level, leaf_node):
        assert self.finalized, "Need to finalize the taxonomy."
        return self.node_level_map[(leaf_node.key, level)]

    def leaf_nodes(self):
        if self.finalized and self._leaf_nodes != None:
            nodes = self._leaf_nodes
        else:
            nodes = [node for node in self.breadth_first_traversal()
                     if node.is_leaf]
            self._leaf_nodes = nodes
        return nodes

    def inner_nodes(self):
        if self.finalized and self._inner_nodes != None:
            nodes = self._inner_nodes
        else:
            nodes = [node for node in self.breadth_first_traversal()
                     if not node.is_leaf]
            self._inner_nodes = nodes
        return nodes

    def breadth_first_traversal(self):
        """ Return a list of nodes in breadth first order.
        """
        if self.finalized and self._breadth_first_traversal:
            nodes = self._breadth_first_traversal
        else:
            nodes = []
            queue = [self.root_node]
            while len(queue):
                node = queue.pop(0)
                if not node.is_leaf:
                    queue += node.children.values()
                nodes.append(node)
            self._breadth_first_traversal = nodes
        return nodes

    def duplicate(self, duplicate_data=True):
        node_data = self.export(export_data=duplicate_data)
        dup = Taxonomy()
        dup.load(node_data)
        return dup

    def load(self, taxonomy_data):
        """ Load the following structure
        [{
          'key' : key
          'parent' : key
          'data' : {}
        }]
        """
        if self.finalized:
            raise ValueError("Taxonomy is already finalized.")

        for node_data in taxonomy_data:
            self.add_node_from_data(node_data)


    def export(self, export_data=True):
        """ Export the following structure
        [{
          'key' : key
          'parent' : key
          'data' : {}
        }]
        """
        node_data = []
        for node in self.breadth_first_traversal():
            node_data.append({
                'key': node.key,
                'parent': node.parent.key if not node.is_root else None,
                'data': node.jsonify_data() if export_data else {}
            })

        return node_data

    def stats(self):
        """ Print some stats about the taxonomy.
        """
        if not self.finalized:
            raise ValueError("Finalize the taxonomy first.")

        num_nodes = len(self.nodes)
        print("Number of nodes: %d" % (num_nodes,))
        num_i_nodes = len(self.inner_nodes())
        print("Number of inner nodes: %d" % (num_i_nodes,))
        num_l_nodes = len(self.leaf_nodes())
        print("Number of leaf nodes: %d" % (num_l_nodes,))

        cc = [len(n.children) for n in self.breadth_first_traversal()]
        cs = sum(cc)
        print("Sum of children: %d" % (cs,))
        css = sum([x ** 2 for x in cc])
        print("Sum^2 of children: %d" % (css,))

        leaf_depths = [node.level for node in self.leaf_nodes()]
        print("Max leaf depth: %d" % (max(leaf_depths),))
        print("Min leaf depth: %d" % (min(leaf_depths),))
        print("Avg leaf depth: %d" % (np.mean(leaf_depths),))
