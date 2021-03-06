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
from dataclasses import InitVar, field
from random import Random
from typing import (
    TYPE_CHECKING,
    Mapping,
    MutableMapping,
    MutableSequence,
    NewType,
    Optional,
    Sequence,
    Type,
    cast,
)

from pydantic import parse_obj_as, root_validator  # pylint: disable=no-name-in-module

# Pyright complains about members not existing on type `Dataclass` since it doesn't
# support Pydantic's wrapper variant.  Instead we pretend to import the normal dataclass
# function so as to get Pyright off my back, but that means when we use `config=` in the
# decorator, it still gets pissy so those lines require explicit "type: ignore"s.
if TYPE_CHECKING:
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
    The weight for a given property: must be in [-1, 1] where -1 is fully inversely
    weighted (i.e. it'll cause items with similar values for a property to come
    together), 0 signifies no weight (i.e. this property will have no impact on how
    items come together), and 1 is fully weighted (i.e. it'll cause items with
    dissimilar values for a property to come together).
    """

    def __new__(cls: Type[Weight], value: float) -> Weight:
        ret = super().__new__(cls, value, Boundaries(-1, 1))
        return cast(Weight, ret)


class Similarity(RestrictedFloat):
    """
    The similiarity ratio between two values in a categorical property: must be in
    [0, 1] where 0 is no similarity and 1 is exactly the same.
    """

    def __new__(cls: Type[Similarity], value: float) -> Similarity:
        ret = super().__new__(cls, value, Boundaries(0, 1))
        return cast(Similarity, ret)


class PydanticConfig:  # pylint: disable=too-few-public-methods
    """
    Helper class to allow custom types to be used in `Field`s without having to write
    the validator functions that Pydantic is otherwise looking for.
    """

    arbitrary_types_allowed = True


# pylint: disable=unexpected-keyword-arg
@dataclass(frozen=True, config=PydanticConfig)  # type: ignore
# pylint: enable=unexpected-keyword-arg
class Connection:
    """
    The similarity between two values in a categorical property.
    """

    value_a: str
    value_b: str
    similarity: Similarity = field(default_factory=lambda: Similarity(0))


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


# pylint: disable=unexpected-keyword-arg
@dataclass(frozen=True, config=PydanticConfig)  # type: ignore
# pylint: enable=unexpected-keyword-arg
class RandomizerProperty(NumericProperty):
    """
    A property that contains pseudo-randomly generated data within the
    measurement_bounds.  A random number generator with a specific seed or any other
    properties can be provided so long as it implements `uniform(a, b)`.
    """

    random: Random = field(default_factory=Random)

    def __post_init__(self) -> None:
        if not isinstance(self.random, Random):
            super().__setattr__("random", Random(self.random))

    def __hash__(self) -> int:  # pylint: disable=useless-super-delegation
        return super().__hash__()

    def data(self, bounds: Optional[ValuePair] = None) -> float:
        """
        Generates random data within inclusive ranges as specified by the first
        provided of `bounds`, `measurement_bounds`, or `scale_bounds`.
        """
        if bounds is not None:
            lower = bounds.lower
            upper = bounds.upper
        elif self.measurement_bounds is not None:
            lower = self.measurement_bounds.lower
            upper = self.measurement_bounds.upper
        else:
            lower = self.scale_bounds.lower
            upper = self.scale_bounds.upper
        return self.random.uniform(lower, upper)


@dataclass(frozen=True)
class CategoricalProperty(DataProperty):
    """
    A property that contains categorical data.  It is possible for categorical
    properties to not be completely dissimilar from each other (ex. red and yellow have
    some similarity as they are both warm colors whereas yellow and blue are dissimilar
    due to blue being a cool color), so it is possible to define the similarity in
    these connections.
    """

    connections: InitVar[Optional[Sequence[Connection]]] = None
    similarities: Optional[Mapping[str, MutableMapping[str, Similarity]]] = field(
        default=None, init=False
    )

    def __post_init_post_parse__(
        self, connections: Optional[Sequence[Connection]]
    ) -> None:
        if connections is not None:
            connections = parse_obj_as(list[Connection], connections)
            sims: Mapping[str, MutableMapping[str, Similarity]] = defaultdict(dict)
            for con in connections:
                sims[con.value_a][con.value_b] = con.similarity
                sims[con.value_b][con.value_a] = con.similarity
            super().__setattr__("similarities", sims)
        else:
            super().__setattr__("similarities", None)

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
    specified here will be accounted for in the similarity calculations.  All properties
    must be unique (i.e. have a unique key).  The exception to this is the randomizer
    property which will overwrite any data keyed with the same key to be random data,
    consequently it is not necessary to specify it in the data set at all.
    """

    identifier_prop: IdentifierProperty
    numeric_props: Optional[MutableSequence[NumericProperty]] = None
    categorical_props: Optional[Sequence[CategoricalProperty]] = None
    randomizer_prop: Optional[RandomizerProperty] = None

    @staticmethod
    def _get_properties(
        values: Mapping[str, Property | Sequence[Property] | None]
    ) -> MutableSequence[Property]:
        """
        Returns a list of all specified properties.
        """
        props: MutableSequence[Property] = (
            [cast(Property, values.get("identifier_prop"))]
            if values.get("identifier_prop") is not None
            else []
        )
        props.extend(
            cast(
                Sequence[Property],
                values.get("numeric_props")
                if values.get("numeric_props") is not None
                else [],
            )
        )
        props.extend(
            cast(
                Sequence[Property],
                values.get("categorical_props")
                if values.get("categorical_props") is not None
                else [],
            )
        )
        props.extend(
            cast(
                Sequence[Property],
                [values.get("randomizer_prop")]
                if values.get("randomizer_prop") is not None
                else [],
            )
        )
        return props

    @root_validator
    @classmethod
    def check_keys_unique(
        cls: Type[DataProperties],
        values: Mapping[str, Property | Sequence[Property] | None],
    ) -> Mapping[str, Property | Sequence[Property] | None]:
        """
        Validate that all the keys are unique.
        """
        props = cls._get_properties(values)
        keys = list(map(lambda p: p.key, props))
        if len(keys) != len(set(keys)):
            raise ValueError("All properties must have unique keys", values)
        return values

    @root_validator
    @classmethod
    def convert_untyped_to_typed(
        cls: Type[DataProperties],
        values: MutableMapping[str, Property | Sequence[Property] | None],
    ) -> Mapping[str, Property | Sequence[Property] | None]:
        """
        Convert all untyped parameters to their appropriate types - necessary since
        when pydantic parses the object, it leaves any non-pydantic classes alone,
        which means that the types backed by, for example, floats aren't converted.
        This is an issue when the types apply certain restrictions to the base class.
        Implementation detail: at the moment this is more of a validation step, but
        might turn into an actual parsing/conversion process if additional custom
        classes are added.
        """

        def convert_key(prop: Property) -> Property:
            object.__setattr__(prop, "key", Key(prop.key))
            return prop

        def convert_weight(prop: Property) -> Property:
            if isinstance(prop, DataProperty):
                object.__setattr__(prop, "weight", Weight(prop.weight))
            return prop

        def convert_similarity(prop: Property) -> Property:
            if isinstance(prop, CategoricalProperty) and prop.similarities:
                for key1 in prop.similarities:
                    for key2 in prop.similarities[key1]:
                        prop.similarities[key1][key2] = Similarity(
                            prop.similarities[key1][key2]
                        )
            return prop

        props = cls._get_properties(values)

        # since the underlying objects are the same, these maps also apply to values
        props = list(map(convert_key, props))
        props = list(map(convert_weight, props))
        props = list(map(convert_similarity, props))

        return values
