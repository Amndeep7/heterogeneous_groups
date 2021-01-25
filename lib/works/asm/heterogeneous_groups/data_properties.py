"""
Data properties for heterogeneous groups.

Specifies what is extracted from a given data set in order to assign semantic values
and create connections as appropriate.

Example:
    `DataProperties(
        IdentifierProperty(Key("id")),
        [NumericProperty(Key("n"), Weight(0.1), ValuePair(0, 10))],
        [
            CategoricalProperty(Key("c1")),
            CategoricalProperty(Key("c2"), [Connection("val1", "val2", 0.25)])
        ]
     )`
"""

# this disable needs to stay until the next version of astroid/pylint
# pylint: disable=unsubscriptable-object

from __future__ import annotations

from collections import defaultdict
from dataclasses import InitVar
from typing import (
    TYPE_CHECKING,
    Mapping,
    MutableMapping,
    NewType,
    Optional,
    Sequence,
    Type,
    cast,
)

from pydantic import root_validator

if TYPE_CHECKING:  # get pyright mostly off my back
    from dataclasses import dataclass  # pylint: disable=ungrouped-imports
else:
    from pydantic.dataclasses import dataclass


@dataclass(frozen=True)  # pylint: disable=used-before-assignment
class ValuePair:
    """
    A pair of numbers, one of which is less than or equal to the other.
    """

    lower: float
    upper: float

    @root_validator
    @classmethod
    def check_ordering(
        cls: Type[ValuePair], values: Mapping[str, float]
    ) -> Mapping[str, float]:
        """
        Validate that the values are ordered correctly.
        """
        lower = cast(float, values.get("lower"))
        upper = cast(float, values.get("upper"))
        if lower > upper:
            raise ValueError("Lower value cannot be greater than upper value", values)
        return values


@dataclass(frozen=True)
class Boundaries(ValuePair):
    """
    Uses a pair of numbers as the boundaries for a range.  There are flags to determine
    if the boundary edge is inclusive.
    """

    lower_inclusive: bool = True
    upper_inclusive: bool = True


class RestrictedFloat(float):
    """
    Restricts a number to be within specific boundaries.
    """

    def __new__(
        cls: Type[RestrictedFloat], value: float, b: Boundaries
    ) -> RestrictedFloat:
        ret = super().__new__(cls, value)
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
        return ret


class Weight(RestrictedFloat):
    """
    The weight for a given property: must be in [0, 1] where 0 is no weight and 1 is
    fully weighted.
    """

    def __new__(cls: Type[Weight], value: float) -> Weight:
        ret = super().__new__(cls, value, Boundaries(0, 1))
        return cast(Weight, ret)


class Similarity(RestrictedFloat):
    """
    The similiarity ratio between two values in a categorical property: must be in
    [0, 1] where 0 is no similarity and 1 is exactly the same.
    """

    def __new__(cls: Type[Similarity], value: float) -> Similarity:
        ret = super().__new__(cls, value, Boundaries(0, 1))
        return cast(Similarity, ret)


@dataclass(frozen=True)
class Connection:
    """
    The similarity between two values in a categorical property.
    """

    value_a: str
    value_b: str
    similarity: Similarity = Similarity(0)


Key = NewType("Key", str)
"""
A semantic wrapper to clearly distinguish keys from data values that happen to be
strings.
"""


@dataclass(frozen=True)
class Property:
    """
    A property from a data item which is identified by its key.
    """

    key: Key

    def __hash__(self) -> int:
        return hash(self.key)


@dataclass(frozen=True)
class IdentifierProperty(Property):
    """
    The property that uniquely identifies the data item from all others.
    """

    def __hash__(self) -> int:  # pylint: disable=useless-super-delegation
        return super().__hash__()


@dataclass(frozen=True)
class DataProperty(Property):
    """
    A property that contains data.  Also includes the weight of the property which
    factors into the determination of similarity between two data items.  By default, a
    property is fully weighted.
    """

    weight: Weight = Weight(1.0)

    def __hash__(self) -> int:  # pylint: disable=useless-super-delegation
        return super().__hash__()


@dataclass(frozen=True)
class NumericProperty(DataProperty):
    """
    A property that contains numeric data.  These values will be normalized to the
    given scale (default [0, 1]).  If measurement bounds are provided then those are
    what will align to the lower and upper values on the scale, otherwise the
    measurement bounds will be automatically determined such that the least value in
    the data set aligns to the lower value and the greatest value in the data set aligns
    to the upper value.
    """

    measurement_bounds: Optional[ValuePair] = None
    scale_bounds: ValuePair = ValuePair(0, 1)

    def __hash__(self) -> int:  # pylint: disable=useless-super-delegation
        return super().__hash__()


@dataclass(frozen=True)
class CategoricalProperty(DataProperty):
    """
    A property that contains categorical data.  It is possible for categorical
    properties to not be completely dissimilar from each other (ex. red and yellow have
    some similarity as they are both warm colors whereas yellow and blue are dissimilar
    due to blue being a cool color), so it is possible to define the similarity in
    these connections.  Instead of using the convenience parameter of `connections`, it
    is possible to directly provide the underlying construct of nested maps, but care
    must be taken to ensure that this construct is commutative
    (ex. sims[val1][val2] == sims[val2][val1]).
    """

    connections: InitVar[Optional[Sequence[Connection]]] = None
    similarities: Optional[Mapping[str, Mapping[str, Similarity]]] = None

    def __post_init_post_parse__(
        self, connections: Optional[Sequence[Connection]]
    ) -> None:
        if self.similarities is None and connections is not None:
            sims: Mapping[str, MutableMapping[str, Similarity]] = defaultdict(dict)
            for con in connections:
                sims[con.value_a][con.value_b] = con.similarity
                sims[con.value_b][con.value_a] = con.similarity
            super().__setattr__("similarities", sims)

    @root_validator
    @classmethod
    def check_mapping_commutative(
        cls: Type[CategoricalProperty], values: Mapping[str, object]
    ) -> Mapping[str, object]:
        """
        Validate that the mapping is commutative.
        """
        sims = cast(
            Optional[Mapping[str, Mapping[str, Similarity]]], values.get("similarities")
        )
        if sims is not None and not all(
            [all([r in sims[c].keys() for c in sims[r].keys()]) for r in sims.keys()]
        ):
            raise ValueError("Similarities must be commutative", sims)
        return values

    def __hash__(self) -> int:  # pylint: disable=useless-super-delegation
        return super().__hash__()


@dataclass(frozen=True)
class DataProperties:
    """
    A specification of all relevant properties from the data set.  Only properties
    specified here will be accounted for in the similarity calculations.
    """

    identifier_prop: IdentifierProperty
    numeric_props: Optional[Sequence[NumericProperty]] = None
    categorical_props: Optional[Sequence[CategoricalProperty]] = None
