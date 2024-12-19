# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

"""Provides common fixtures used across all unit and integration tests."""

import pytest
from click.testing import CliRunner


@pytest.fixture(scope="function")
def click_runner() -> CliRunner:
    """Provides a click Runner to test click-enhanced functions."""
    return CliRunner()
