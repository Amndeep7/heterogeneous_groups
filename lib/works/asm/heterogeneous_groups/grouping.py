"""
Heterogeneous Groups
"""

# this disable needs to stay until the next version of astroid/pylint
# pylint: disable=unsubscriptable-object

from __future__ import annotations

import collections.abc
from functools import reduce
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Hashable,
    List,
    Mapping,
    MutableMapping,
    NewType,
    NoReturn,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)

if TYPE_CHECKING:  # get pyright mostly off my back
    from dataclasses import dataclass
else:
    from pydantic.dataclasses import dataclass


@dataclass(frozen=True)  # pylint: disable=used-before-assignment
class Boundaries:
    """
    Boundaries for floats.
    """

    lower: float
    upper: float
    lower_inclusive: bool = True
    upper_inclusive: bool = True


class RestrictedFloat(float):
    """
    Restricts float to be within specific boundaries.
    """

    def __new__(
        cls: Type[RestrictedFloat], value: float, b: Boundaries
    ) -> RestrictedFloat:
        ret = super().__new__(cls, value)  # type: ignore
        if (
            # pylint: disable=too-many-boolean-expressions
            (b.lower_inclusive and ret < b.lower)
            or (not b.lower_inclusive and ret <= b.lower)
            or (b.upper_inclusive and ret > b.upper)
            or (not b.upper_inclusive and ret >= b.upper)
        ):
            raise ValueError(
                f"{ret} not in {'[' if b.lower_inclusive else '{'}{b.lower}, "
                f"{b.upper}{']' if b.upper_inclusive else '}'}"
            )
        return cast(RestrictedFloat, ret)


class Weight(RestrictedFloat):
    """
    The weight for a given property.
    """

    def __new__(cls: Type[Weight], value: float) -> Weight:
        ret = super().__new__(cls, value, Boundaries(0, 1))
        return cast(Weight, ret)


class Similarity(RestrictedFloat):
    """
    The value of the similiarity between two given properties.
    """

    def __new__(cls: Type[Similarity], value: float) -> Similarity:
        ret = super().__new__(cls, value, Boundaries(0, 1))
        return cast(Similarity, ret)


Key = NewType("Key", str)


@dataclass(frozen=True)
class Property:
    """
    Property.
    """

    key: Key

    def __hash__(self) -> int:
        return hash(self.key)


@dataclass(frozen=True)
class IdentifierProperty(Property):
    """
    Identifier property.
    """

    def __hash__(self) -> int:  # pylint: disable=useless-super-delegation
        return super().__hash__()


@dataclass(frozen=True)
class DataProperty(Property):
    """
    Data property.
    """

    weight: Weight = Weight(1.0)

    def __hash__(self) -> int:  # pylint: disable=useless-super-delegation
        return super().__hash__()


@dataclass(frozen=True)
class NumericProperty(DataProperty):
    """
    Numeric property.
    """

    measurement_bounds: Optional[Boundaries] = None
    scale_bounds: Boundaries = Boundaries(0, 1)

    def __hash__(self) -> int:  # pylint: disable=useless-super-delegation
        return super().__hash__()


@dataclass(frozen=True)
class CategoricalProperty(DataProperty):
    """
    Categorical property.
    """

    similarities: Optional[Mapping[str, Mapping[str, Similarity]]] = None

    def __hash__(self) -> int:  # pylint: disable=useless-super-delegation
        return super().__hash__()


def incorrect_type_error(instance: Any) -> NoReturn:
    """
    The 'else' when pattern matching on a union type.
    """
    raise AssertionError(f"Incorrect type: {type(instance).__name__}")


class HeterogeneousGrouper:
    """Creates a heterogeneous group out of a set of data"""

    def __init__(
        self,
        data: Sequence[Mapping[Hashable, Any]],
        identifier_prop: IdentifierProperty,
        numeric_props: Optional[Sequence[NumericProperty]] = None,
        categorical_props: Optional[Sequence[CategoricalProperty]] = None,
    ) -> None:
        """
        ``data`` is the complete data set.  Each item in the data set should contain
        an identifier property and 0 or more numeric and categorical properties.  The
        only data retained is from the numeric and categorical properties.

        ``identifier_prop`` is the property that uniquely identifies an item from the
        data set.

        ``numeric_props`` is the list of properties that contain numeric data.  The
        values for all those properties should be floats.

        ``categorical_props`` is the list of properties that contain categorical data.
        The values for all those properties should be strings.
        """

        self.identifier = identifier_prop

        if isinstance(numeric_props, collections.abc.Sequence):
            self.nprops = numeric_props
        elif numeric_props is None:
            self.nprops = []
        else:
            incorrect_type_error(numeric_props)

        if isinstance(categorical_props, collections.abc.Sequence):
            self.cprops = categorical_props
        elif categorical_props is None:
            self.cprops = []
        else:
            incorrect_type_error(categorical_props)

        self.data: Sequence[Mapping[Key, Union[float, str]]] = [
            {
                key: d[key]
                for key in [p.key for p in self.nprops]
                + [p.key for p in self.cprops]
                + [self.identifier.key]
            }
            for d in data
        ]

    def scaled_grid(
        self,
    ) -> Mapping[Hashable, Mapping[DataProperty, Union[float, str]]]:
        """
        Makes something akin to the portfolio grid from the paper.
        """
        grid: Mapping[Hashable, MutableMapping[DataProperty, Union[float, str]]] = {
            cast(Hashable, id): {}
            for id in [item[self.identifier.key] for item in self.data]
        }

        # categorical data is just added into the grid
        for item in self.data:
            for cprop in self.cprops:
                grid[cast(Hashable, item[self.identifier.key])][cprop] = cast(
                    str, item[cprop.key]
                )

        # numeric data needs processing
        for nprop in self.nprops:
            if nprop.measurement_bounds:
                mbl = nprop.measurement_bounds.lower
                mbu = nprop.measurement_bounds.upper
            else:
                mbl = min([cast(float, item[nprop.key]) for item in self.data])
                mbu = max([cast(float, item[nprop.key]) for item in self.data])
            for item in self.data:
                grid[cast(Hashable, item[self.identifier.key])][nprop] = (
                    cast(float, item[nprop.key]) - mbl
                ) / (mbu - mbl) * (
                    nprop.scale_bounds.upper - nprop.scale_bounds.lower
                ) + nprop.scale_bounds.lower

        return grid

    def difference_matrix(self) -> Mapping[Hashable, MutableMapping[Hashable, float]]:
        """
        Makes something akin to the difference matrix from the paper.
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
                for hnp in self.nprops:
                    matrix[i][j] += (
                        hnp.weight
                        * abs(cast(float, grid[i][hnp]) - cast(float, grid[j][hnp]))
                    ) / (hnp.scale_bounds.upper - hnp.scale_bounds.lower)
                for hcp in self.cprops:
                    matrix[i][j] += hcp.weight * (
                        hcp.similarities[cast(str, grid[i][hcp])][
                            cast(str, grid[j][hcp])
                        ]
                        if hcp.similarities
                        and grid[i][hcp] in hcp.similarities
                        and grid[j][hcp] in hcp.similarities[cast(str, grid[i][hcp])]
                        else 0
                        if grid[i][hcp] == grid[j][hcp]
                        else 1
                    )
                matrix[i][j] = matrix[i][j] / (len(self.nprops) + len(self.cprops))
        return matrix

    def group_algorithm_number(self, g: int) -> Any:
        """
        Implementation of the heterogeneous clustering/grouping algorithm based off of a specified number of groups.
        """
        matrix = self.difference_matrix()
        c = dict(
            zip(
                range(len(self.data)),
                [[item[self.identifier.key]] for item in self.data],
            )
        )

        while len(c.keys()) > g and any(map(lambda row: len(matrix[row]), matrix)):
            # get the identifers for the two items with the highest difference
            m_i, m_j = max(
                [
                    (
                        r,
                        reduce(
                            lambda acc, cur: acc
                            if matrix[r][acc] > matrix[r][cur]
                            else cur,
                            matrix[r].keys(),
                        ),
                    )
                    for r in matrix
                ],
                key=lambda ij: matrix[ij[0]][ij[1]],
            )

            # get the cluster/group identifier for items i and j
            c_h = reduce(
                lambda acc, cur: cur if m_i in c[cur] else acc,
                c.keys(),
                list(c.keys())[0],
            )
            c_k = reduce(
                lambda acc, cur: cur if m_j in c[cur] else acc,
                c.keys(),
                list(c.keys())[0],
            )

            if c_h != c_k:
                c[c_h].extend(c[c_k])
                c.pop(c_k)

            matrix[m_i].pop(m_j)
            matrix[m_j].pop(m_i)

        return c


if __name__ == "__main__":
    mydata: List[Dict[Hashable, Any]] = [
        {"id": "a", "n1": 1, "n2": 2, "c1": "left", "c2": "ye", "c3": "re"},
        {"id": "b", "n1": 3, "n2": 4, "c1": "right", "c2": "yee", "c3": "ree"},
        {"id": "c", "n1": 1, "n2": 2, "c1": "left", "c2": "yee", "c3": "re"},
    ]
    idp = IdentifierProperty(Key("id"))
    nps = [
        NumericProperty(key=Key("n1"), measurement_bounds=Boundaries(0, 10)),
        NumericProperty(key=Key("n2")),
    ]
    cps = [
        CategoricalProperty(Key("c1")),
        CategoricalProperty(
            Key("c2"),
            similarities={
                "ye": {"yee": Similarity(0.5)},
                "yee": {"ye": Similarity(0.5)},
            },
        ),
        CategoricalProperty(Key("c3"), Weight(0.25)),
    ]
    grouper = HeterogeneousGrouper(mydata, idp, nps, cps)
    print(grouper.scaled_grid())
    print(grouper.difference_matrix())
    print(grouper.group_algorithm_number(2))

    print("================")

    mydata = [
        {"id": "a", "c1": "left"},
        {"id": "b", "c1": "right"},
        {"id": "c", "c1": "left"},
    ]
    idp = IdentifierProperty(Key("id"))
    nps = []
    cps = [CategoricalProperty(Key("c1"))]
    grouper = HeterogeneousGrouper(mydata, idp, nps, cps)
    print(grouper.scaled_grid())
    print(grouper.difference_matrix())
    print(grouper.group_algorithm_number(2))

    print("================")

    mydata = [
        {"id": "a", "c1": "left", "c2": "up"},
        {"id": "b", "c1": "right", "c2": "up"},
        {"id": "c", "c1": "left", "c2": "down"},
    ]
    idp = IdentifierProperty(Key("id"))
    nps = []
    cps = [CategoricalProperty(Key("c1")), CategoricalProperty(Key("c2"))]
    grouper = HeterogeneousGrouper(mydata, idp, nps, cps)
    print(grouper.scaled_grid())
    print(grouper.difference_matrix())
    print(grouper.group_algorithm_number(2))

    print("================")

    mydata = [
        {"id": "a", "c1": "left", "c2": "up"},
        {"id": "b", "c1": "right", "c2": "up"},
        {"id": "c", "c1": "left", "c2": "down"},
    ]
    idp = IdentifierProperty([5])
    nps = []
    cps = [CategoricalProperty(Key("c1")), CategoricalProperty(Key("c2"))]
    grouper = HeterogeneousGrouper(mydata, idp, nps, cps)
    print(grouper.scaled_grid())
    print(grouper.difference_matrix())
    print(grouper.group_algorithm_number(2))
