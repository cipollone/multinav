# -*- coding: utf-8 -*-
#
# Copyright 2020 Roberto Cipollone, Marco Favorito
#
# ------------------------------
#
# This file is part of multinav.
#
# multinav is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# multinav is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with multinav.  If not, see <https://www.gnu.org/licenses/>.
#

"""Test fluents extraction."""

import pytest
from flloat.semantics import PLInterpretation

from multinav.envs import env_cont_sapientino
from multinav.envs.base import AbstractFluents


class Fluents1(AbstractFluents):
    """Class without fluent list."""

    def evaluate(self, obs, action):
        """Any evaluation function."""
        return {"is_one"} if obs == 1 else set()


class Fluents2(Fluents1):
    """Complete fluents extractor."""

    def __init__(self):
        """Initialize."""
        self.fluents = {"is_one", "is_two"}


def test_fluents_base():
    """Test the base class."""
    with pytest.raises(TypeError):
        fluents = Fluents1()
    fluents = Fluents2()

    assert fluents.evaluate(1, 0).issubset(fluents.fluents)


def test_fluents_cont_sapientino():
    """Test fluents extraction on cont-sapientino."""
    # NOTE: this test depends on gym-sapientino color order
    with pytest.raises(ValueError):
        fluents = env_cont_sapientino.Fluents({"not a color"})
    fluents = env_cont_sapientino.Fluents({"red", "blue"})  # with just 2 fluents

    assert fluents.evaluate(dict(beep=0, color=1), 0) == PLInterpretation(set())
    assert fluents.evaluate(dict(beep=1, color=1), 0) == PLInterpretation({"red"})
    assert fluents.evaluate(dict(beep=1, color=3), 0) == PLInterpretation({"blue"})
    with pytest.raises(RuntimeError):
        fluents.evaluate(dict(beep=1, color=2), 0)  # green not used
