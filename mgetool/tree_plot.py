#
import copy
import itertools
import warnings
from typing import Union, List, Tuple, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap


def _join(trees):
    self = trees[0]
    other = trees[1:]
    for i in other:
        self.append(i)
    return self


def sum_tree(_iterable, n_jobs=1):
    if n_jobs == 1:
        return _join(_iterable)
    else:
        import multiprocessing
        pool = multiprocessing.Pool(processes=n_jobs)
        from tqdm import tqdm

        step = len(_iterable) // n_jobs
        left = len(_iterable) % n_jobs
        if left > 0:
            n_jobs = n_jobs - 1
            steps = [(i * step, (i + 1) * step,) for i in range(n_jobs)]
            steps.append((-left, len(_iterable)))
        else:
            steps = [(i * step, (i + 1) * step,) for i in range(n_jobs)]

        res = []
        for stepi in tqdm(steps):
            iteri = _iterable[stepi[0]:stepi[1]]
            res.append(pool.apply(func=_join, args=(iteri,)))

        pool.close()
        pool.join()
        return _join(res)


class BuildTree:
    """
    For composition importance show.
    1. Each layer of color are decided by the predefined 'color_maps',
        and each branch color is decided by 'color_maps' and 'com_maps'.
    2. If to sort, the tree are sort by 'com_maps' in each layer.
    3. The site of each node are decided by super root.
    4. After set site (``set_site_all``), don't change the tree (add/append/delete any branch).

    Notes:
        The tree must end with the name in last com_maps, unless you change the 'color_maps' and 'com_maps'.
        (the tree start from the name in first com_maps is not enforced).
        due to the index to get message from 'color_maps' and 'com_maps' is inverted order.

    Examples:
    # Generate 1
    >>> com = {"l3": {"O": {"V": 0.1, "Mo": 0.2}, "S": {"Ti": 0.93, "W": 0.4}}}
    >>> tt0 = BuildTree.from_com_dict(com, sort=True)

    # Generate 2
    >>> tt1 = BuildTree.from_one_branch(branch=['l3', "S", "V"], w=1)

    # Cat 2 tree
    >>> tt2 = BuildTree(name="Cr", w=1.0, sub_tree=[tt0,tt1])

    #  Set point sites, color, and line automatriclly.
    >>> tt2.settle()
    or
    >>> tt2.set_site_all()
    >>> tt2.set_color_all()
    >>> res = tt2.get_line()

    >>> tt2.show(text=True, mark=True)
    >>> msg_dict = tt2.get_msg()
    >>> num_dict = tt2.get_num()
    >>> branch_dict = tt2.get_branch()

    #  Tune site manually.
    >>> tt2.manual_set_site(site=np.array([1.5,1]),num=1,add=False,total=False)
    >>> tt2.show(text=True, mark=True)
    """

    def __init__(self, name: Union[str, Sequence[str]], w: float = 1.0,
                 sub_tree: Union["BuildTree", Sequence["BuildTree"]] = None,
                 array: np.ndarray = None,
                 color_maps=None,
                 com_maps=None,
                 ):
        """

        Args:
            name: str, name of node.
            w: float, weight of this node.
            sub_tree: list of BuildTree, sub-tree.
        """

        if sub_tree is None:
            sub_tree = []
        elif isinstance(sub_tree, BuildTree):
            sub_tree = [sub_tree, ]

        if com_maps is None:
            com_maps = []

        if color_maps is None:
            color_maps = [plt.get_cmap("cool")] * len(com_maps)

        if com_maps:
            assert len(com_maps) == len(color_maps)

        self._max_depth = len(com_maps)

        self.sub_tree = sub_tree
        assert 0.0 <= w <= 1.0
        self.name = name if isinstance(name, str) else tuple(name)

        # 0,  1,  2,    3,         4, 5,          6, 7, 8, 9
        # x1, y1, mark, mark_size, w, line_style, R, G, B, alpha
        # w is weight, and line width.
        if isinstance(array, np.ndarray):
            array[(3, 4),] = w
        else:
            array = np.array([0.0, 0.0, 0.0, w, w, 1.0, 0.0, 0.0, 0.0, 1.0])
        self.array = array
        self.num_msg = None

        self.color_maps = color_maps
        self.com_maps = com_maps

    def __len__(self):
        """Sub-tree numbers."""
        return len(self.sub_tree)

    def __setitem__(self, key, value):
        self.sub_tree[key] = value

    def __getitem__(self, item):
        return self.sub_tree[item]

    def _get_sub_tree(self, name):
        for i in self.sub_tree:
            if i.name == name:
                return i
        raise KeyError(f"Not find {name}")

    @property
    def root(self) -> np.ndarray:
        """Root site."""
        return self.array[:2]

    @property
    def depth(self) -> int:
        """Depth the of tree, the leaf layer is 1, and the root layer is the large."""
        if len(self.sub_tree) == 0:
            return 1
        else:
            return max([i.depth for i in self.sub_tree]) + 1

    @property
    def layer(self) -> int:
        """The layer number of tree. used for sort composition and get color.
        (If the number of layer are small than _max_depth, the layer are start not 0)."""
        layer = self._max_depth - self.depth
        return layer

    @property
    def node(self) -> int:
        """Total nodes in this tree."""
        if len(self.sub_tree) == 0:
            return 1
        else:
            return sum([i.node for i in self.sub_tree]) + 1

    @property
    def element(self) -> List[str]:
        """All node names in tree."""
        if len(self.sub_tree) == 0:
            return [self.name, ]
        else:
            s = list(itertools.chain(*[i.msg for i in self.sub_tree]))
            s.append(self.name)
            return list(set(s))

    @property
    def leaf_node(self) -> int:
        """Total leaf nodes in this tree."""
        if len(self.sub_tree) == 0:
            return 1
        else:
            return sum([i.leaf_node for i in self.sub_tree])

    def __copy__(self):
        new = self.__class__(self.name, w=self.array[4], sub_tree=self.sub_tree)
        new.array = self.array
        return new

    def copy(self):
        return self.__copy__()

    def __deepcopy__(self, memodict: dict = None):
        if memodict is None:
            pass
        new = self.__class__(self.name, w=self.array[4], sub_tree=copy.deepcopy(self.sub_tree))
        new.array = self.array.copy()
        return new

    def __add__(self, other: "BuildTree") -> "BuildTree":
        """Add 2 tree to one."""
        new = self.copy()
        new.append(other=other)
        return new

    def gain(self, others: 'BuildTree'):
        self.name = others.name
        self.array = self.array
        self.sub_tree = others.sub_tree
        self.num_msg = None

    def append(self, other: 'BuildTree', sort: bool = False) -> None:
        """Merge tree to the first tree. """
        assert self.name == other.name
        temp = []
        for t2i in other.sub_tree:
            mk = 0
            for ti in self.sub_tree:
                if ti.name == t2i.name:
                    ti.append(t2i, sort=sort)
                    mk = 1
                    break
            if mk == 0:
                temp.append(t2i)

        self.sub_tree.extend(temp)

        if sort is True:
            self.sort()

    @staticmethod
    def sum_tree(_iterable: Sequence["BuildTree"], n_jobs=1) -> "BuildTree":
        return sum_tree(_iterable, n_jobs=n_jobs)

    def __repr__(self):
        return f"{self.name}(depth:{self.depth}, leafs:{self.leaf_node}, branch:{len(self.sub_tree)})"

    # 1. generate

    @classmethod
    def from_one_branch(cls, branch: list, w: float = 1) -> "BuildTree":
        """Generate one tree by composition sequence, need use append to join them.

        Examples:
        >>> tt1 = BuildTree.from_one_branch(branch=["C", ("Ta","Ti"), 'l3', "O", "V"], w=1)
        >>> tt2 = BuildTree.from_one_branch(branch=["C", ("Ta","Mo"), 'l3', "O", "V"], w=1)
        >>> tt1.append(tt2)
        """
        branch.reverse()
        cs = cls(branch[0], w=w, sub_tree=None)
        for ci in branch[1:]:
            cs = cls(name=ci, w=w, sub_tree=[cs, ])
        return cs

    @classmethod
    def from_branches(cls, branches: Sequence, ws: Sequence, n_jobs=1) -> "BuildTree":
        if n_jobs == 1:
            self = cls.from_one_branch(branch=list(branches[0]), w=ws[0])
            for b, w in zip(branches[1:], ws[1:]):
                t = cls.from_one_branch(branch=list(b), w=w)
                self.append(t)
            return self
        else:
            res = [cls.from_one_branch(branch=list(b), w=w) for b, w in zip(branches, ws)]
            return cls.sum_tree(res, n_jobs=n_jobs)

    @classmethod
    def from_com_dict(cls, com: dict, sort: bool = False) -> Union["BuildTree", List["BuildTree"]]:
        """Generate tree by composition dict.

        Examples:
        >>> com1 = {"l3":{"O": {"V": 0.1, "Mo": 0.2}, "S": {"Ti": 0.3, "W": 0.4}}}
        >>> BuildTree.from_com_dict(com1)
        l3(depth:3, leafs:4, branch:2)
        """
        subs_tree = []
        for k, val in com.items():
            if isinstance(val, (float, int)):
                subs_tree.append(cls(name=k, w=val, sub_tree=None))
            else:
                subs = cls.from_com_dict(val, sort=False)
                subs_tree.append(cls(name=k, sub_tree=subs))

        if len(subs_tree) == 1:
            ma = subs_tree[0]
            if sort:
                ma.sort()
            return ma
        else:
            mas = subs_tree
            if sort:
                [i.sort() for i in mas]
            return mas

    # Color

    def _get_color(self, names: Union[list, tuple, str], layer: int = 0) -> list:
        """
        Get color by name.

        Args:
            names:str
            layer:int

        Returns:
            list of 4 float. (R,G,B,alpha)

        """
        if isinstance(names, str):
            names = [names, ]
        if self.com_maps:
            try:
                com_dict = self.com_maps[layer]
                ratio = sum([com_dict[i] for i in names]) / len(names) / len(com_dict)
                if isinstance(self.color_maps[layer], Colormap):
                    return list(self.color_maps[layer](ratio))
                else:
                    return list(self.color_maps[layer])

            except KeyError:
                warnings.warn("Error to find color, \n"
                              "Reasion 1: The depth of tree is out of size of com_maps\n"
                              "Reasion 2: The tree's layer name not corresponding with each layer of com_maps\n"
                              "Suggest the size com_maps is the same with max depth of tree",
                              UserWarning)
                return [0.0, 0.0, 0.0, 1.0]

        else:
            return [0.0, 0.0, 0.0, 1.0]

    def get_color(self, auto_w=True) -> list:
        """Get color this layer."""

        c4 = self._get_color(self.name, layer=self.layer)
        self.array[-4:] = c4
        if auto_w and self.sub_tree:
            self.array[3] = np.max([i.array[3] for i in self.sub_tree])
            self.array[4] = np.mean([i.array[4] for i in self.sub_tree])
        self.array[9] = self.array[4]
        return c4

    def check(self):
        if self.com_maps:
            if self.depth == self._max_depth:
                pass
            elif self.depth < self._max_depth:
                print(f"The {self.depth} is less than {self._max_depth},"
                      f"The top {self._max_depth - self.depth} message of color_map and com_map would not use.")
                if self.name not in self.com_maps[self.layer]:
                    warnings.warn(f"Please make the tree's layer name is corresponding "
                                  f"with each layer from {self._max_depth - self.depth} of com_maps!!!.", UserWarning)

            elif self.depth > self._max_depth:
                raise KeyError(f"The {self.depth} is more than {self._max_depth},"
                               f"please redefined your color_map/com_map or check your tree.", )

    def set_color_all(self, auto_w=True, check=True) -> None:
        """Get colors of all layer."""
        if check:
            self.check()
        for sti in self.sub_tree:
            sti.set_color_all(auto_w=auto_w, check=False)
        self.get_color(auto_w=auto_w)

    # Sort

    def sort(self) -> None:
        """Sort names for better shown for top layer."""
        if self.com_maps:
            if self.sub_tree and self.depth > 1:
                rank = self.com_maps[self.layer + 1]
                l = len(rank) + 1

                def com(i):
                    if i.name in rank:
                        return rank[i.name]
                    elif isinstance(i.name, tuple):
                        return sum([rank[i] for i in i.name]) / len(i.name)
                    else:
                        print(f"{i.name} not in {rank},"
                              f"would sort as same.")
                        return l

                self.sub_tree.sort(key=com)

    def sort_all(self) -> None:
        """Sort names for better shown. not forced."""
        self.sort()
        for sti in self.sub_tree:
            sti.sort_all()

    # Site

    @staticmethod
    def get_sub_node_site_random(root: np.ndarray, r: float = 1.0, noise: bool = True) -> np.ndarray:
        """Get sub root by uniform."""
        angles = np.random.choice(-30, -150) / 180 * np.pi
        if noise:
            ll = (np.random.random() / 2 + 0.5) * r
        else:
            ll = r
        b2 = np.sin(angles) * ll + root[0, 1]
        a2 = np.cos(angles) * ll + root[0, 0]
        return np.array([a2, b2])

    @staticmethod
    def get_sub_nodes_site_sequence(root: np.ndarray, r: float = 1.0, n: int = 10,
                                    noise: bool = True, method="uniform", r_noise=1.3) -> List:
        """Get sub root by uniform."""

        if n == 0:
            return []
        elif n == 1:
            angle = np.random.random(1) * 100 + 40

        else:
            if method == "randint":
                angle = np.random.randint(30, 150, n)
                if noise:
                    angle = np.random.uniform(-10, 10, n) + angle
                else:
                    angle = np.random.uniform(-10, 10) + angle
            elif method == "bias":
                angle = np.linspace(60, 160, n)
                if noise:
                    angle = np.random.uniform(-10, 10, n) + angle
                else:
                    angle = np.random.uniform(-10, 10) + angle
            else:
                angle = np.linspace(30, 150, n)
                if noise:
                    angle = np.random.uniform(-10, 10, n) + angle
                else:
                    angle = np.random.uniform(-10, 10) + angle

        angle = angle / 180 * np.pi

        # print(angle)
        if method == "bias":
            if noise:
                ll = (r_noise * np.random.random(n) / 2 + 0.3) * r * (0.5 + (np.arange(0, n, step=1) / n)) * (
                            int(n / 10) / 5 + 1)
            else:
                ll = np.full(n, r)
        else:
            if noise:
                ll = (r_noise * np.random.random(n) / 2 + 0.3) * r * (int(n / 10) / 5 + 1)
            else:
                ll = np.full(n, r)

        b2 = np.sin(angle) * ll + root[1]
        a2 = np.cos(angle) * ll + root[0]
        return [np.array([i, j]) for i, j in zip(a2, b2)]

    def set_site_all(self, root: np.ndarray = None, method="bias") -> None:
        """Set the sites for plot.

        Args:
            root:np.ndarray, site of root.
        """

        if root is None:
            root = np.array([0.0, 0.0])
        else:
            root = self.root

        if self.depth == 1:
            return

        if hasattr(self, "rs"):
            r = self.rs[-self.depth]
            if self.depth == 2:
                noise = False
            else:
                noise = True
        else:
            if self.depth == 2:
                r = 1.0
                noise = False
            else:
                r = 1.0 * 2 ** self.depth / 4
                noise = True

        if self._max_depth != 0 and self.depth == self._max_depth:
            method = method
        else:
            method = "uniform"

        n = len(self.sub_tree)

        sub_roots = self.get_sub_nodes_site_sequence(root, n=n, r=r, noise=noise, method=method)
        for sti, rti in zip(self.sub_tree, sub_roots):
            sti.array[:2] = rti
            sti.set_site_all(rti, method=method)

    def cover(self, other: 'BuildTree') -> bool:
        """Check the site of two trees covered or not."""
        d = self.root - other.root
        if (d[0] ** 2 + 0.25 * d[1] ** 2) ** 0.5 < self.array[5] + self.array[5]:
            return True
        else:
            return False

    def move(self, move_x: float, move_y: float) -> None:
        """Move site of total tree sites."""
        self.array[0] = self.array[0] + move_x
        self.array[1] = self.array[1] + move_y

    # Num, Line, Branch, Message
    def _get_line(self):
        lines = []
        for ti in self.sub_tree:
            if isinstance(ti.name, tuple):
                ni = ",".join([str(i) for i in ti.name])
            else:
                ni = str(ti.name)
            msg1 = np.concatenate((self.root, ti.array, np.array(ti.layer).reshape(1, ), np.array([ni]),))
            lines.append(msg1)
            # self.num=num
            # num += 1
            lines.extend(ti._get_line())
        return lines

    def get_line(self) -> np.ndarray:
        """Get point couple for plot."""
        data = np.array(self._get_line())
        return data

    def _get_num(self, num: int = 1, ini: tuple = None) -> Tuple[dict, int]:
        if ini is None:
            ini = (0,)
            num_dict = {0: ini}
        else:
            num_dict = {}
        for k, ti in enumerate(self.sub_tree):
            ini2 = tuple(list(ini) + [k])
            num_dict.update({num: ini2})
            # self.num=num
            num += 1
            res, num = ti._get_num(num, ini=ini2)
            num_dict.update(res)
        return num_dict, num

    def get_num(self) -> Dict:
        """Get branch index number dict of tree.

        such as :
        {0: (0,),
         1: (0, 0),
         2: (0, 0, 0),
         3: (0, 0, 1),
         4: (0, 1),
         5: (0, 1, 0),
         6: (0, 1, 1)}
        """
        return self._get_num()[0]

    def _get_branch(self, num: int = 1, ini: tuple = None) -> Tuple[dict, int]:
        if ini is None:
            ini = (self.name,)
            num_dict = {0: ini}
        else:
            num_dict = {}
        for ti in self.sub_tree:
            ini2 = tuple(list(ini) + [ti.name])
            num_dict.update({num: ini2})
            # self.num = num
            num += 1
            res, num = ti._get_branch(num, ini=ini2)
            num_dict.update(res)
        return num_dict, num

    def get_branch(self) -> Dict:
        """Get branch name dict of tree.

        such as :
        {0: ('l3',),
         1: ('l3', 'O'),
         2: ('l3', 'O', 'V'),
         3: ('l3', 'O', 'Mo'),
         4: ('l3', 'S'),
         5: ('l3', 'S', 'Ti'),
         6: ('l3', 'S', 'W')}
         """
        return self._get_branch()[0]

    def _all_sub_nodes(self):
        res = []
        if self.sub_tree:
            for i in self.sub_tree:
                res.append(i)
                res.extend(i._all_sub_nodes())
        return res

    def get_all_sub_nodes(self, include_self=True):
        """get all sub root"""
        res = []
        if include_self:
            res.append(self)
        res.extend(self._all_sub_nodes())
        return res

    def get_nodes(self, num: int=None, branch: Union[Tuple, List] = None, method="node2leafs"):

        temp, res = self.get_node_linear(num=num, branch=branch)
        if method == "node":
            return [temp, ]
        elif method == "root2node":
            return res
        elif method == "node2leafs":
            return temp.get_all_sub_nodes(include_self=True)
        elif method == "all":
            res.extend(temp.get_all_sub_nodes(include_self=False))
            return res
        else:
            return []

    def get_node_linear(self, num: int = None, branch: Tuple = None) -> Tuple["BuildTree", List["BuildTree"]]:
        """get the node by the number or branch.
        see also: get_branch, get_num"""
        temp = self
        res = [temp, ]

        if num is not None:
            if num == 0:
                pass
            else:
                if not hasattr(self, "num_msg") or self.num_msg is None:
                    self.get_msg()
                for i in self.num_msg[num][0][1:]:
                    temp = temp[i]
                    res.append(temp)
        else:
            if len(branch) == 1 and branch[0] == self.name:
                pass
            else:
                for i in branch[1:]:
                    temp = temp._get_sub_tree(i)
                    res.append(temp)
        return temp, res

    def set_nodes_prop(self, name: str, value: any, num: int = None, branch: Tuple = None, method="node2leafs") -> None:
        """Set the property of the node in the branch, (the node could not be leaf node).
        if total, set the all node from the root to target node.
        see also: get_branch"""
        res = self.get_nodes(num=num, branch=branch, method=method)
        [setattr(i, name, value) for i in res]

    def set_nodes_w(self, w, num: int = None, branch: Tuple = None,
                        method="node", index=(3, 4, 9)) -> None:
        """Set the array property of the node in the branch, (the node could not be leaf node).
        if total, set the all node from the root to target node.
        see also: get_branch"""
        res = self.get_nodes(num=num, branch=branch, method=method)

        for resi in res:
            array = resi.array
            array[list(index)] = w
            resi.array = array


    def set_nodes_site(self, site: Union[np.ndarray, List[float]], num: int = None, branch: Tuple = None,
                       method="node2leafs",
                       add: bool = False) -> None:
        """Set the property of the last node in the branch.
        see also: get_branch"""
        if add is False:
            assert method != "node"

        res = self.get_nodes(num=num, branch=branch, method=method)

        if not add:
            for resi in res:
                resi.array[:2] = np.array(site)
        else:
            for resi in res:
                resi.array[:2] = resi.array[:2] + np.array(site)

    def _get_msg(self, num: int = 1, ini: Tuple = None) -> Tuple[dict, int]:
        if ini is None:
            ini = ((0,), (self.name,))
            num_dict = {0: ini}
        else:
            num_dict = {}

        for k, ti in enumerate(self.sub_tree):
            ini2 = (tuple(list(ini[0]) + [k]), tuple(list(ini[1]) + [ti.name,]))
            num_dict.update({num: ini2})
            # self.num = num
            num += 1
            res, num = ti._get_msg(num, ini=ini2)
            num_dict.update(res)

        self.num_msg = num_dict

        return num_dict, num

    def get_msg(self) -> Dict:
        """Get message dict.

        such as :
        {0: ((0,), ('l3',)),
         1: ((0, 0), ('l3', 'O')),
         2: ((0, 0, 0), ('l3', 'O', 'V')),
         3: ((0, 0, 1), ('l3', 'O', 'Mo')),
         4: ((0, 1), ('l3', 'S')),
         5: ((0, 1, 0), ('l3', 'S', 'Ti')),
         6: ((0, 1, 1), ('l3', 'S', 'W'))}
        """
        return self._get_msg()[0]

    # Plot

    def settle(self, method="uniform"):
        self.sort_all()
        self.set_site_all(root=None, method=method)
        self.set_color_all(auto_w=True)
        self.get_line()

    def get_plt(self, text: bool = False, mark: bool = False, linewidth: float = 1.0, text_site="center",
                text_msg="num", text_size=15, mark_size: float = 200, mark_size_method: str = "auto",
                mark_layer=(), mark_alpha="mean", text_layer=(), ) -> plt:
        """Plot tree.
        0,  1,  2,  3,  4,         5,         6, 7,          8, 9, 10, 11,    12,    13,
        x0, y0, x1, y1, mark_type, mark_size, w, line_style, R, G, B,  alpha, layer, name
        """

        def func_text(ri, ni):
            if text_msg == "num":
                return f"{ri}"
            elif text_msg == "num_name":
                return f"{ri}-{ni}"
            else:
                return f"{ni}"

        assert mark_size_method in ("auto", "cnt")
        res = self.get_line()
        rank = np.arange(res.shape[0]) + 1
        data = res[:, :-1].astype(float)
        names = res[:, -1]
        if mark_alpha == "max":
            ma = 5
        else:
            ma = 6

        mk = [".", "o", "^", "x", "+", "*", "s"]

        if 0 in mark_layer:
            if mark:
                if mark_size_method == "auto":
                    s = (0.01 + data[0, 5]) * mark_size
                else:
                    s = mark_size
                plt.scatter(self.root[0], self.root[1], c="k", s=s, marker=mk[int(data[0, 4])], alpha=data[0, ma]**2)
        if 0 in text_layer:
            if text:
                plt.text(self.root[0], self.root[1], func_text(0, self.name), size=text_size)

        for resi, ri, ni in zip(data, rank, names):
            la = int(resi[-1])

            c = resi[8:12]
            lw = resi[6]

            # leaf 2 layer
            if la == self._max_depth - 2:
                c = resi[(8, 9, 10, 5),]
                lw = resi[5]

            if la >= self._max_depth - 2:
                lw2 = 1.0
            else:
                lw2 = linewidth

            plt.plot(resi[(0, 2),], resi[(1, 3),], c=c, linewidth=1 * lw2 * lw)

            if la in mark_layer:
                if mark:
                    if mark_size_method == "auto":
                        s = (0.01 + resi[5]) * mark_size
                    else:
                        s = mark_size
                    plt.scatter(resi[2], resi[3], c="k", s=s, marker=mk[int(resi[4])], alpha=resi[ma]**2)
            if la in text_layer:
                if text and text_site != "center":
                    plt.text(resi[2], resi[3], func_text(ri, ni), size=text_size)
                elif text:
                    plt.text((resi[0] + resi[2]) / 2, (resi[1] + resi[3]) / 2, func_text(ri, ni), size=text_size)


        return plt

    def show(self, **kwargs) -> None:
        """Plot tree."""
        plt0 = self.get_plt(**kwargs)
        plt0.show()


if __name__ == "__main__":
    np.random.seed(4)

    # com = {"l3": {"O": {"V": 0.1, "Mo": 0.2}, "S": {"Ti": 0.93, "W": 0.4}}}
    # tt0 = BuildTree.from_com_dict(com)
    #
    # tt0.set_site_all(root=np.array([1.0, 1.0]))
    # tt0.set_color_all()
    # res = tt0.get_line()
    # plt = tt0.get_plt(text=True, mark=True, mark_size=500)
    # plt.show()
    # msg = tt0.get_msg()
    # tt0.get_branch()
    # num = tt0.get_num()
