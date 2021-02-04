"""
CLI for heterogeneous groups - accessible by calling `heterogeneous_groups`.
"""

# this disable needs to stay until the next version of astroid/pylint
# pylint: disable=unsubscriptable-object

from enum import Enum
from functools import partial
from importlib.metadata import version as app_version
from json import load
from pathlib import Path
from typing import NoReturn, Optional

import typer
from pydantic import parse_file_as  # pylint: disable=no-name-in-module

from .data_properties import DataProperties
from .grouping import Grouper

app = typer.Typer()


def version_callback(  # pylint: disable=useless-return
    value: bool,
) -> Optional[NoReturn]:
    """
    Processes the version option before the rest of the inputs.
    """
    if value:
        typer.echo(f"Heterogeneous groups v{app_version(__package__)}")
        raise typer.Exit()
    return None


class AlgorithmUserInput(str, Enum):
    """
    Mapping between the user input and an interface representing the algorithms.
    """

    NUMBER = "NUMBER"
    SAME_SIZE = "SAME_SIZE"


class Algorithm(Enum):
    """
    Mapping between the grouping algorithms and an interface representing the
    algorithms.
    """

    NUMBER = partial(Grouper.group_algorithm_number)
    SAME_SIZE = partial(Grouper.group_algorithm_same_size_best_approximation)


@app.command()
def main(  # pylint: disable=too-many-arguments
    path_to_data: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help=(
            "Path to a json blob consisting of a list wrapping around the data items,"
            " which should be objects."
        ),
    ),
    path_to_dataprops: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help=(
            "Path to the json blob consisting of the object that describes the "
            "keys/properties that will be extracted from the data items and used to "
            "determine the heterogeneous groups."
        ),
    ),
    algorithm: AlgorithmUserInput = typer.Option(
        AlgorithmUserInput.SAME_SIZE,
        "-a",
        "--algo",
        "--algorithm",
        case_sensitive=False,
        help=(
            "The heterogeneous grouping algorithm to be used on the dataset to form "
            "the groups."
        ),
    ),
    num_groups: int = typer.Option(
        1,
        "-n",
        "--num-groups",
        help="Number of groups to divide the items in the dataset into.",
    ),
    verbose: Optional[bool] = typer.Option(None, "-v", "--verbose"),
    version: Optional[bool] = typer.Option(  # pylint: disable=unused-argument
        None, "-V", "--version", callback=version_callback, is_eager=True
    ),
) -> None:
    """
    Separate a given dataset into heterogeneous groups.
    """
    with open(path_to_data, "r") as datafile:
        grouper = Grouper(
            load(datafile), parse_file_as(DataProperties, path_to_dataprops)
        )
        if verbose:
            typer.echo(grouper)
            typer.echo(grouper.data)
            typer.echo(grouper.scaled_grid())
            typer.echo(grouper.difference_matrix())
        typer.echo(Algorithm[algorithm.name].value(grouper, num_groups))


if __name__ == "__main__":
    app()
