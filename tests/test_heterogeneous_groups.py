# pylint: disable=missing-module-docstring,missing-function-docstring,redefined-outer-name

import pytest
from works.asm.heterogeneous_groups.data_properties import (
    CategoricalProperty,
    Connection,
    DataProperties,
    IdentifierProperty,
    Key,
    NumericProperty,
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
            similarities={
                "ye": {"yee": Similarity(0.5)},
                "yee": {"ye": Similarity(0.5)},
            },
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
                similarities={
                    "ye": {"yee": Similarity(0.5)},
                    "yee": {"ye": Similarity(0.5)},
                },
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
                similarities={
                    "ye": {"yee": Similarity(0.5)},
                    "yee": {"ye": Similarity(0.5)},
                },
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
                similarities={
                    "ye": {"yee": Similarity(0.5)},
                    "yee": {"ye": Similarity(0.5)},
                },
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
            connections=[Connection("val1", "val2", Similarity(0.25))],
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
                similarities={
                    "val1": {"val2": Similarity(0.25)},
                    "val2": {"val1": Similarity(0.25)},
                },
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
                similarities={
                    "val1": {"val2": Similarity(0.25)},
                    "val2": {"val1": Similarity(0.25)},
                },
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
