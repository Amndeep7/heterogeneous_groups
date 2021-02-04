"""
Grouping algorithms for heterogeneous groups.

Implements the algorithms that generate heterogeneous groups from a given data set.

Example: `
    Grouper(
        [
            {"id": "id1", "n": 2, "c1": "me", "c2": "val1"},
            {"id": "id2", "n": 7, "c1": "you", "c2": "val2"},
        ],
        DataProperties(
            IdentifierProperty(Key("id")),
            [NumericProperty(Key("n"), Weight(0.1), ValuePair(0, 10))],
            [
                CategoricalProperty(Key("c1")),
                CategoricalProperty(
                    Key("c2"),
                    connections=[Connection("val1", "val2", Similarity(0.25))],
                ),
            ],
        ),
    )
`
"""

# this disable needs to stay until the next version of astroid/pylint
# pylint: disable=unsubscriptable-object

from __future__ import annotations

import math
from functools import reduce
from itertools import combinations
from typing import (
    TYPE_CHECKING,
    Hashable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
    Union,
    cast,
)

from .data_properties import DataProperties, DataProperty, Key

if TYPE_CHECKING:  # get pyright mostly off my back
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass


class Grouper:
    """
    Creates a heterogeneous group out of a set of data each item of which should
    contain an identifier property and 0 or more numeric and categorical properties.
    """

    def __init__(
        self, data: Sequence[Mapping[Hashable, object]], data_props: DataProperties
    ) -> None:

        self.identifier = data_props.identifier_prop

        if data_props.numeric_props is not None:
            self.nprops = data_props.numeric_props
        else:
            self.nprops = []

        if data_props.categorical_props is not None:
            self.cprops = data_props.categorical_props
        else:
            self.cprops = []

        self.data: Sequence[MutableMapping[Key, float | str]] = [
            {
                key: cast(Union[float, str], d[key])
                for key in [p.key for p in self.nprops]
                + [p.key for p in self.cprops]
                + [self.identifier.key]
            }
            for d in data
        ]

        if data_props.randomizer_prop is not None:
            self.nprops.extend([data_props.randomizer_prop])
            for d in self.data:  # pylint: disable=invalid-name
                d[data_props.randomizer_prop.key] = data_props.randomizer_prop.data()

    def scaled_grid(
        self,
    ) -> Mapping[Hashable, Mapping[DataProperty, float | str]]:
        """
        Makes something akin to the portfolio grid from the paper, namely a matrix of
        data item identifiers crossed by all the data properties.
        """
        grid: Mapping[Hashable, MutableMapping[DataProperty, float | str]] = {
            cast(Hashable, id): {}
            for id in [item[self.identifier.key] for item in self.data]
        }

        # categorical data is just added into the grid
        for item in self.data:
            for cprop in self.cprops:
                identifier = cast(Hashable, item[self.identifier.key])
                c_value = cast(str, item[cprop.key])
                grid[identifier][cprop] = c_value
        # numeric data needs processing
        for nprop in self.nprops:
            if nprop.measurement_bounds:
                mbl = nprop.measurement_bounds.lower
                mbu = nprop.measurement_bounds.upper
            else:
                mbl = min([cast(float, item[nprop.key]) for item in self.data])
                mbu = max([cast(float, item[nprop.key]) for item in self.data])
            for item in self.data:
                identifier = cast(Hashable, item[self.identifier.key])
                n_value = cast(float, item[nprop.key])
                grid[identifier][nprop] = (n_value - mbl) / (mbu - mbl) * (
                    nprop.scale_bounds.upper - nprop.scale_bounds.lower
                ) + nprop.scale_bounds.lower

        return grid

    def difference_matrix(self) -> Mapping[Hashable, MutableMapping[Hashable, float]]:
        """
        Makes something akin to the difference matrix from the paper: the cross product
        of all the data items with themselves so as to have a value for the difference
        between them as determined by the sum of the products of the lacks of
        similarity multiplied by the weights of the respective property.
        """
        grid = self.scaled_grid()
        matrix = {
            id1: {
                id2: 0.0
                for id2 in [
                    cast(Hashable, item[self.identifier.key]) for item in self.data
                ]
            }
            for id1 in [cast(Hashable, item[self.identifier.key]) for item in self.data]
        }
        for i in matrix.keys():
            for j in matrix[i].keys():
                for nprop in self.nprops:
                    matrix[i][j] += (
                        nprop.weight
                        * abs(cast(float, grid[i][nprop]) - cast(float, grid[j][nprop]))
                    ) / (nprop.scale_bounds.upper - nprop.scale_bounds.lower)
                for cprop in self.cprops:
                    matrix[i][j] += cprop.weight * (
                        # dissimilarity = 1 - similarity
                        1
                        - cprop.similarities[cast(str, grid[i][cprop])][
                            cast(str, grid[j][cprop])
                        ]
                        if cprop.similarities
                        and grid[i][cprop] in cprop.similarities
                        and grid[j][cprop]
                        in cprop.similarities[cast(str, grid[i][cprop])]
                        else 0
                        if grid[i][cprop] == grid[j][cprop]
                        else 1
                    )
                matrix[i][j] = matrix[i][j] / (len(self.nprops) + len(self.cprops))
        return matrix

    def group_algorithm_number(self, num_g: int) -> Mapping[int, Sequence[str]]:
        """
        Implementation of the heterogeneous grouping algorithm based off of a specified
        number of groups.  The algorithm did not specify what to do to prevent group
        sizes from exceeding ceil(num_items/num_g) which results in a bug where almost
        all the items consolidate into one massive group with the exception of however
        many needed to fill out the specified number of groups requirement, so I came
        up with a stopgap measure that caps group sizes at ceil(num_items/num_g) but it
        runs into a worst case when num_items/num_g has a non-zero remainder such that
        groups consistently hit that size leaving one group considerably undersized w/r
        to its brethren.
        """
        if num_g < 1:
            raise ValueError(
                f"Insufficient groups: must be greater than 1, but was actually {num_g}"
            )
        if num_g > len(self.data):
            raise ValueError(
                f"Number of groups ({num_g}) greater than number of items "
                f"({len(self.data)})"
            )

        matrix = self.difference_matrix()
        groups: MutableMapping[int, MutableSequence[str]] = dict(
            zip(
                range(len(self.data)),
                [[cast(str, item[self.identifier.key])] for item in self.data],
            )
        )

        while len(groups.keys()) > num_g and any(
            map(lambda row: len(matrix[row]), matrix)
        ):
            # get the identifers for the two items with the highest difference
            m_i, m_j = max(
                [
                    (
                        r,
                        reduce(
                            lambda acc, cur: acc
                            # this would've been a reasonable catch from pylint except
                            # for the fact that the reduce calls the lambda immediately
                            # instead of lazily waiting until the end, so the variable
                            # doesn't turn into just the last one
                            if matrix[r][acc]  # pylint: disable=cell-var-from-loop
                            > matrix[r][cur]  # pylint: disable=cell-var-from-loop
                            else cur,
                            matrix[r].keys(),
                        ),
                    )
                    for r in matrix
                ],
                key=lambda ij: matrix[ij[0]][ij[1]],
            )

            # get the group identifier for items i and j
            g_h: int = reduce(
                lambda acc, cur: cur if m_i in groups[cur] else acc,
                groups.keys(),
                list(groups.keys())[0],
            )
            g_k: int = reduce(
                lambda acc, cur: cur if m_j in groups[cur] else acc,
                groups.keys(),
                list(groups.keys())[0],
            )

            if g_h != g_k and (
                len(groups[g_h]) + len(groups[g_k])
                <= math.ceil(len(matrix.keys()) / num_g)
            ):
                groups[g_h].extend(groups[g_k])
                groups.pop(g_k)

            # since it's not a triangular matrix, have to delete from both sides
            matrix[m_i].pop(m_j)
            matrix[m_j].pop(m_i)

        return groups

    def group_algorithm_same_size_best_approximation(
        self, num_g: int
    ) -> Mapping[int, Sequence[Hashable]]:
        """
        My own novel (to me at least, maybe not to literature at large) heterogeneous
        grouping algorithm that ensures group sizes are as consistent as possible.

        Textual description of the algorithm is as follows:

        1) Identify a set of size num_g consisting of the items most homogenous to each
        other.  Assign each item in this set to their own group.

        2) While there are still ungrouped items:
            - Identify the ungrouped item-unassigned group pairing which has the largest
            cumulative difference (i.e. the sum of the differences between this item
            and every item in the group) and assign that item to that group.
            - While there are still groups that haven't been assigned to this round (and
            there are still ungrouped items): repeat the assignment process but only
            using groups that have not yet been assigned to in the current round.
        """
        if num_g < 1:
            raise ValueError(
                f"Insufficient groups: must be greater than 1, but was actually {num_g}"
            )
        if num_g > len(self.data):
            raise ValueError(
                f"Number of groups ({num_g}) greater than number of items "
                f"({len(self.data)})"
            )

        matrix = self.difference_matrix()

        homogeneous_group = min(
            combinations(matrix.keys(), num_g),
            key=lambda c: sum(matrix[i[0]][i[1]] for i in combinations(c, 2)),
        )
        groups: Mapping[int, MutableSequence[Hashable]] = {
            i: [val] for i, val in enumerate(homogeneous_group)
        }

        ungrouped = list(set(matrix.keys()) - set(homogeneous_group))
        while ungrouped:
            unassigned = list(groups.keys())

            def get_cum_diff(i_item: int, i_group: int) -> float:
                return sum(
                    matrix[ungrouped[i_item]][v] for v in groups[unassigned[i_group]]
                )

            while unassigned and ungrouped:
                l_item = 0
                l_group = 0
                l_cd = get_cum_diff(l_item, l_group)

                for i, _ in enumerate(ungrouped):
                    for j, _ in enumerate(unassigned):
                        cur_cd = get_cum_diff(i, j)
                        if cur_cd > l_cd:
                            l_item = i
                            l_group = j
                            l_cd = cur_cd

                groups[unassigned[l_group]].append(ungrouped[l_item])

                unassigned.pop(l_group)
                ungrouped.pop(l_item)

        return groups
