# pylint: disable=missing-module-docstring,missing-function-docstring,redefined-outer-name

from random import Random

import pytest
from works.asm.heterogeneous_groups.data_properties import (
    CategoricalProperty,
    Connection,
    DataProperties,
    IdentifierProperty,
    Key,
    NumericProperty,
    RandomizerProperty,
    Similarity,
    ValuePair,
    Weight,
)
from works.asm.heterogeneous_groups.grouping import Grouper


@pytest.fixture
def data_ten_percent_difference():
    return [
        {"id": "a", "n1": 1, "n2": 2, "c1": "left", "c2": "ye", "c3": "re"},
        {"id": "b", "n1": 3, "n2": 4, "c1": "right", "c2": "yee", "c3": "ree"},
        {"id": "c", "n1": 1, "n2": 2, "c1": "left", "c2": "yee", "c3": "re"},
    ]


@pytest.fixture
def dataprops_ten_percent_difference():
    idp = IdentifierProperty(Key("id"))
    nps = [
        NumericProperty(key=Key("n1"), measurement_bounds=ValuePair(0, 10)),
        NumericProperty(key=Key("n2")),
    ]
    cps = [
        CategoricalProperty(Key("c1")),
        CategoricalProperty(
            Key("c2"),
            connections=[Connection("ye", "yee", 0.5)] # type: ignore
        ),
        CategoricalProperty(Key("c3"), Weight(0.25)),
    ]
    return DataProperties(idp, nps, cps)


def test_ten_percent_scaled_grid(
    data_ten_percent_difference, dataprops_ten_percent_difference
):
    assert Grouper(
        data_ten_percent_difference, dataprops_ten_percent_difference
    ).scaled_grid() == {
        "a": {
            CategoricalProperty(Key("c1"), Weight(1.0)): "left",
            CategoricalProperty(
                Key("c2"),
                Weight(1.0),
                connections=[Connection("ye", "yee", 0.5)], # type: ignore
            ): "ye",
            CategoricalProperty(Key("c3"), Weight(0.25)): "re",
            NumericProperty(
                Key("n1"),
                Weight(1.0),
                measurement_bounds=ValuePair(0.0, 10.0),
                scale_bounds=ValuePair(0.0, 1.0),
            ): 0.1,
            NumericProperty(Key("n2"), Weight(1.0)): 0.0,
        },
        "b": {
            CategoricalProperty(Key("c1"), Weight(1.0)): "right",
            CategoricalProperty(
                Key("c2"),
                Weight(1.0),
                connections=[Connection("ye", "yee", 0.5)], # type: ignore
            ): "yee",
            CategoricalProperty(Key("c3"), Weight(0.25)): "ree",
            NumericProperty(
                Key("n1"), Weight(1.0), measurement_bounds=ValuePair(0.0, 10.0)
            ): 0.3,
            NumericProperty(Key("n2"), Weight(1.0)): 1.0,
        },
        "c": {
            CategoricalProperty(Key("c1"), Weight(1.0)): "left",
            CategoricalProperty(
                Key("c2"),
                Weight(1.0),
                connections=[Connection("ye", "yee", 0.5)], # type: ignore
            ): "yee",
            CategoricalProperty(Key("c3"), Weight(0.25)): "re",
            NumericProperty(
                Key("n1"), Weight(1.0), measurement_bounds=ValuePair(0.0, 10.0)
            ): 0.1,
            NumericProperty(Key("n2"), Weight(1.0)): 0.0,
        },
    }


def test_ten_percent_difference_matrix(
    data_ten_percent_difference, dataprops_ten_percent_difference
):
    assert Grouper(
        data_ten_percent_difference, dataprops_ten_percent_difference
    ).difference_matrix() == {
        "a": {"a": 0.0, "b": 0.5900000000000001, "c": 0.1},
        "b": {"a": 0.5900000000000001, "b": 0.0, "c": 0.49000000000000005},
        "c": {"a": 0.1, "b": 0.49000000000000005, "c": 0.0},
    }


def test_ten_percent_group_algorithm_number(
    data_ten_percent_difference, dataprops_ten_percent_difference
):
    assert Grouper(
        data_ten_percent_difference, dataprops_ten_percent_difference
    ).group_algorithm_number(2) == {0: ["a", "b"], 2: ["c"]}


def test_ten_percent_group_algorithm_same_size_best_approximation(
    data_ten_percent_difference, dataprops_ten_percent_difference
):
    assert Grouper(
        data_ten_percent_difference, dataprops_ten_percent_difference
    ).group_algorithm_same_size_best_approximation(2) == {0: ["a", "b"], 1: ["c"]}


@pytest.fixture
def data_b_unique():
    return [
        {"id": "a", "c1": "left"},
        {"id": "b", "c1": "right"},
        {"id": "c", "c1": "left"},
    ]


@pytest.fixture
def dataprops_b_unique():
    idp = IdentifierProperty(Key("id"))
    nps = []
    cps = [CategoricalProperty(Key("c1"))]
    return DataProperties(idp, nps, cps)


def test_b_unique_scaled_grid(data_b_unique, dataprops_b_unique):
    assert Grouper(data_b_unique, dataprops_b_unique).scaled_grid() == {
        "a": {CategoricalProperty(Key("c1"), Weight(1.0)): "left"},
        "b": {CategoricalProperty(Key("c1"), Weight(1.0)): "right"},
        "c": {CategoricalProperty(Key("c1"), Weight(1.0)): "left"},
    }


def test_b_unique_matrix(data_b_unique, dataprops_b_unique):
    assert Grouper(data_b_unique, dataprops_b_unique).difference_matrix() == {
        "a": {"a": 0.0, "b": 1.0, "c": 0.0},
        "b": {"a": 1.0, "b": 0.0, "c": 1.0},
        "c": {"a": 0.0, "b": 1.0, "c": 0.0},
    }


def test_b_unique_group_algorithm_number(data_b_unique, dataprops_b_unique):
    assert Grouper(data_b_unique, dataprops_b_unique).group_algorithm_number(2) == {
        0: ["a", "b"],
        2: ["c"],
    }


def test_b_unique_group_algorithm_same_size_best_approximation(
    data_b_unique, dataprops_b_unique
):
    assert Grouper(
        data_b_unique, dataprops_b_unique
    ).group_algorithm_same_size_best_approximation(2) == {
        0: ["a", "b"],
        1: ["c"],
    }


@pytest.fixture
def data_a_split():
    return [
        {"id": "a", "c1": "left", "c2": "up"},
        {"id": "b", "c1": "right", "c2": "up"},
        {"id": "c", "c1": "left", "c2": "down"},
    ]


@pytest.fixture
def dataprops_a_split():
    idp = IdentifierProperty(Key("id"))
    nps = []
    cps = [CategoricalProperty(Key("c1")), CategoricalProperty(Key("c2"))]
    return DataProperties(idp, nps, cps)


def test_a_split_scaled_grid(data_a_split, dataprops_a_split):
    assert Grouper(data_a_split, dataprops_a_split).scaled_grid() == {
        "a": {
            CategoricalProperty(Key("c1"), Weight(1.0)): "left",
            CategoricalProperty(Key("c2"), Weight(1.0)): "up",
        },
        "b": {
            CategoricalProperty(Key("c1"), Weight(1.0)): "right",
            CategoricalProperty(Key("c2"), Weight(1.0)): "up",
        },
        "c": {
            CategoricalProperty(Key("c1"), Weight(1.0)): "left",
            CategoricalProperty(Key("c2"), Weight(1.0)): "down",
        },
    }


def test_a_split_matrix(data_a_split, dataprops_a_split):
    assert Grouper(data_a_split, dataprops_a_split).difference_matrix() == {
        "a": {"a": 0.0, "b": 0.5, "c": 0.5},
        "b": {"a": 0.5, "b": 0.0, "c": 1.0},
        "c": {"a": 0.5, "b": 1.0, "c": 0.0},
    }


def test_a_split_group_algorithm_number(data_a_split, dataprops_a_split):
    assert Grouper(data_a_split, dataprops_a_split).group_algorithm_number(2) == {
        0: ["a"],
        1: ["b", "c"],
    }


def test_a_split_group_algorithm_same_size_best_approximation(
    data_a_split, dataprops_a_split
):
    assert Grouper(
        data_a_split, dataprops_a_split
    ).group_algorithm_same_size_best_approximation(2) == {
        0: ["a"],
        1: ["b", "c"],
    }


@pytest.fixture
def data_example():
    return [
        {"id": "id1", "n": 2, "c1": "me", "c2": "val1"},
        {"id": "id2", "n": 7, "c1": "you", "c2": "val2"},
    ]


@pytest.fixture
def dataprops_example():
    idp = IdentifierProperty(Key("id"))
    nps = [NumericProperty(Key("n"), Weight(0.1), ValuePair(0, 10))]
    cps = [
        CategoricalProperty(Key("c1")),
        CategoricalProperty(
            Key("c2"),
            connections=[Connection("val1", "val2", Similarity(0.25))], # type: ignore
        ),
    ]
    return DataProperties(idp, nps, cps)


def test_example_scaled_grid(data_example, dataprops_example):
    assert Grouper(data_example, dataprops_example).scaled_grid() == {
        "id1": {
            CategoricalProperty(Key("c1"), Weight(1.0)): "me",
            CategoricalProperty(
                Key("c2"),
                Weight(1.0),
                connections=[Connection("val1", "val2", Similarity(0.25))], # type: ignore
            ): "val1",
            NumericProperty(
                Key("n"), Weight(0.1), measurement_bounds=ValuePair(0.0, 10.0)
            ): 0.2,
        },
        "id2": {
            CategoricalProperty(Key("c1"), Weight(1.0)): "you",
            CategoricalProperty(
                Key("c2"),
                Weight(1.0),
                connections=[Connection("val1", "val2", Similarity(0.25))], # type: ignore
            ): "val2",
            NumericProperty(
                Key("n"), Weight(0.1), measurement_bounds=ValuePair(0.0, 10.0)
            ): 0.7,
        },
    }


def test_example_matrix(data_example, dataprops_example):
    assert Grouper(data_example, dataprops_example).difference_matrix() == {
        "id1": {"id1": 0.0, "id2": 0.6},
        "id2": {"id1": 0.6, "id2": 0.0},
    }


def test_example_group_algorithm_number(data_example, dataprops_example):
    assert Grouper(data_example, dataprops_example).group_algorithm_number(2) == {
        0: ["id1"],
        1: ["id2"],
    }


def test_example_group_algorithm_same_size_best_approximation(
    data_example, dataprops_example
):
    assert Grouper(
        data_example, dataprops_example
    ).group_algorithm_same_size_best_approximation(2) == {
        0: ["id1"],
        1: ["id2"],
    }


def test_keys_must_be_unique():
    with pytest.raises(ValueError):
        DataProperties(IdentifierProperty(Key("id")), [NumericProperty(Key("id"))])


@pytest.fixture
def data_random():
    return [{"id": c} for c in "abcdefghij"]


@pytest.fixture
def dataprops_random():
    rand = Random(0)
    return (
        DataProperties(
            IdentifierProperty(Key("id")),
            randomizer_prop=RandomizerProperty(
                Key("rand"), random=rand  # type: ignore
            ),
        ),
        rand,
    )


def test_random_scaled_grid(data_random, dataprops_random):
    assert Grouper(data_random, dataprops_random[0]).scaled_grid() == {
        "a": {
            RandomizerProperty(
                Key("rand"), random=dataprops_random[1]
            ): 1.0  # type: ignore
        },
        "b": {
            RandomizerProperty(
                Key("rand"), random=dataprops_random[1]
            ): 0.8523199056630058  # type: ignore
        },
        "c": {
            RandomizerProperty(
                Key("rand"), random=dataprops_random[1]
            ): 0.2760946577539797  # type: ignore
        },
        "d": {
            RandomizerProperty(
                Key("rand"), random=dataprops_random[1]
            ): 0.0  # type: ignore
        },
        "e": {
            RandomizerProperty(
                Key("rand"), random=dataprops_random[1]
            ): 0.4310090049507776  # type: ignore
        },
        "f": {
            RandomizerProperty(
                Key("rand"), random=dataprops_random[1]
            ): 0.24938704521990493  # type: ignore
        },
        "g": {
            RandomizerProperty(
                Key("rand"), random=dataprops_random[1]
            ): 0.8964598901654225  # type: ignore
        },
        "h": {
            RandomizerProperty(
                Key("rand"), random=dataprops_random[1]
            ): 0.07582508793269464  # type: ignore
        },
        "i": {
            RandomizerProperty(
                Key("rand"), random=dataprops_random[1]
            ): 0.37178190830673485  # type: ignore
        },
        "j": {
            RandomizerProperty(
                Key("rand"), random=dataprops_random[1]
            ): 0.5541630439756922  # type: ignore
        },
    }


def test_random_matrix(data_random, dataprops_random):
    assert Grouper(data_random, dataprops_random[0]).difference_matrix() == {
        "a": {
            "a": 0.0,
            "b": 0.14768009433699425,
            "c": 0.7239053422460203,
            "d": 1.0,
            "e": 0.5689909950492225,
            "f": 0.750612954780095,
            "g": 0.10354010983457751,
            "h": 0.9241749120673054,
            "i": 0.6282180916932651,
            "j": 0.44583695602430784,
        },
        "b": {
            "a": 0.14768009433699425,
            "b": 0.0,
            "c": 0.5762252479090261,
            "d": 0.8523199056630058,
            "e": 0.42131090071222815,
            "f": 0.6029328604431008,
            "g": 0.04413998450241674,
            "h": 0.7764948177303111,
            "i": 0.4805379973562709,
            "j": 0.2981568616873136,
        },
        "c": {
            "a": 0.7239053422460203,
            "b": 0.5762252479090261,
            "c": 0.0,
            "d": 0.2760946577539797,
            "e": 0.15491434719679792,
            "f": 0.02670761253407475,
            "g": 0.6203652324114428,
            "h": 0.20026956982128502,
            "i": 0.09568725055275518,
            "j": 0.2780683862217125,
        },
        "d": {
            "a": 1.0,
            "b": 0.8523199056630058,
            "c": 0.2760946577539797,
            "d": 0.0,
            "e": 0.4310090049507776,
            "f": 0.24938704521990493,
            "g": 0.8964598901654225,
            "h": 0.07582508793269464,
            "i": 0.37178190830673485,
            "j": 0.5541630439756922,
        },
        "e": {
            "a": 0.5689909950492225,
            "b": 0.42131090071222815,
            "c": 0.15491434719679792,
            "d": 0.4310090049507776,
            "e": 0.0,
            "f": 0.18162195973087267,
            "g": 0.4654508852146449,
            "h": 0.35518391701808294,
            "i": 0.05922709664404274,
            "j": 0.12315403902491456,
        },
        "f": {
            "a": 0.750612954780095,
            "b": 0.6029328604431008,
            "c": 0.02670761253407475,
            "d": 0.24938704521990493,
            "e": 0.18162195973087267,
            "f": 0.0,
            "g": 0.6470728449455175,
            "h": 0.1735619572872103,
            "i": 0.12239486308682992,
            "j": 0.3047759987557872,
        },
        "g": {
            "a": 0.10354010983457751,
            "b": 0.04413998450241674,
            "c": 0.6203652324114428,
            "d": 0.8964598901654225,
            "e": 0.4654508852146449,
            "f": 0.6470728449455175,
            "g": 0.0,
            "h": 0.8206348022327279,
            "i": 0.5246779818586876,
            "j": 0.34229684618973033,
        },
        "h": {
            "a": 0.9241749120673054,
            "b": 0.7764948177303111,
            "c": 0.20026956982128502,
            "d": 0.07582508793269464,
            "e": 0.35518391701808294,
            "f": 0.1735619572872103,
            "g": 0.8206348022327279,
            "h": 0.0,
            "i": 0.2959568203740402,
            "j": 0.4783379560429975,
        },
        "i": {
            "a": 0.6282180916932651,
            "b": 0.4805379973562709,
            "c": 0.09568725055275518,
            "d": 0.37178190830673485,
            "e": 0.05922709664404274,
            "f": 0.12239486308682992,
            "g": 0.5246779818586876,
            "h": 0.2959568203740402,
            "i": 0.0,
            "j": 0.1823811356689573,
        },
        "j": {
            "a": 0.44583695602430784,
            "b": 0.2981568616873136,
            "c": 0.2780683862217125,
            "d": 0.5541630439756922,
            "e": 0.12315403902491456,
            "f": 0.3047759987557872,
            "g": 0.34229684618973033,
            "h": 0.4783379560429975,
            "i": 0.1823811356689573,
            "j": 0.0,
        },
    }


def test_random_group_algorithm_same_size_best_approximation(
    data_random, dataprops_random
):
    assert Grouper(
        data_random, dataprops_random[0]
    ).group_algorithm_same_size_best_approximation(3) in [
        {0: ["c", "g", "j"], 1: ["f", "a", "d", "e"], 2: ["i", "b", "h"]},
        {0: ["c", "g", "e"], 1: ["f", "a", "d", "j"], 2: ["i", "b", "h"]},
    ]
