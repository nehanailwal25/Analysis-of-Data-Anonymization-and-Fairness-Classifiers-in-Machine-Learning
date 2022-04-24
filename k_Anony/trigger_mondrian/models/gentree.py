# -*- coding: utf-8 -*-

# logic tree


class GenTree(object):

    def __init__(self, value=None, parent=None, isleaf=False):
        self.value = ''
        self.level = 0
        self.leaf_num = 0
        self.parent = []
        self.child = []
        self.cover = {}
        if value is not None:
            self.value = value
            self.cover[value] = self
        if parent is not None:
            self.parent = parent.parent[:]
            self.parent.insert(0, parent)
            parent.child.append(self)
            self.level = parent.level + 1
            for t in self.parent:
                t.cover[self.value] = self
                if isleaf:
                    t.leaf_num += 1

    def node(self, value):

        try:
            return self.cover[value]
        except KeyError:
            return None

    def __len__(self):

        return self.leaf_num
