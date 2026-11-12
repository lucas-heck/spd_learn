# Copyright (c) 2024-now SPD Learn Developers
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the logging utilities in :mod:`spd_learn.logging`."""

import logging
import sys
import types
import warnings

import pytest

from spd_learn import logging as spd_logging


@pytest.fixture(autouse=True)
def restore_package_logger():
    """Restore the package logger after each test."""
    logger = logging.getLogger(spd_logging.LOGGER_NAME)
    handlers = list(logger.handlers)
    level = logger.level
    propagate = logger.propagate
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def test_get_logger_names():
    assert spd_logging.get_logger().name == "spd_learn"
    assert spd_logging.get_logger("models").name == "spd_learn.models"
    assert spd_logging.get_logger("spd_learn.modules").name == "spd_learn.modules"


def test_import_installs_null_handler():
    logger = logging.getLogger("spd_learn")
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)


@pytest.mark.parametrize("level", ["DEBUG", "warning", logging.ERROR])
def test_set_log_level(level):
    spd_logging.set_log_level(level)
    expected = getattr(logging, level.upper()) if isinstance(level, str) else level
    assert logging.getLogger("spd_learn").level == expected


def test_disable_and_enable_logging():
    spd_logging.disable_logging()
    assert logging.getLogger("spd_learn").level == logging.CRITICAL + 1
    spd_logging.enable_logging("DEBUG")
    assert logging.getLogger("spd_learn").level == logging.DEBUG


@pytest.mark.parametrize("level", ["DEBUG", logging.ERROR])
def test_log_level_context_manager_restores(level):
    logger = logging.getLogger("spd_learn")
    logger.setLevel(logging.WARNING)
    expected = getattr(logging, level) if isinstance(level, str) else level
    with spd_logging.log_level(level):
        assert logger.level == expected
    assert logger.level == logging.WARNING


def test_log_level_context_manager_restores_on_error():
    logger = logging.getLogger("spd_learn")
    logger.setLevel(logging.WARNING)
    with pytest.raises(RuntimeError):
        with spd_logging.log_level("DEBUG"):
            raise RuntimeError
    assert logger.level == logging.WARNING


def test_configure_logging_standard_handler(monkeypatch):
    # Make ``from rich.logging import RichHandler`` fail.
    monkeypatch.setitem(sys.modules, "rich.logging", None)
    spd_logging.configure_logging(
        level="DEBUG", use_rich=True, format_string="%(message)s"
    )
    logger = logging.getLogger("spd_learn")
    assert logger.level == logging.DEBUG
    assert logger.propagate is False
    assert len(logger.handlers) == 1
    (handler,) = logger.handlers
    assert type(handler) is logging.StreamHandler
    assert handler.level == logging.DEBUG
    assert handler.formatter._fmt == "%(message)s"


def test_configure_logging_without_rich_uses_default_format():
    spd_logging.configure_logging(level=logging.INFO, use_rich=False)
    (handler,) = logging.getLogger("spd_learn").handlers
    assert type(handler) is logging.StreamHandler
    assert handler.formatter._fmt == spd_logging.DEFAULT_FORMAT


def test_configure_logging_with_rich(monkeypatch):
    class FakeRichHandler(logging.Handler):
        def __init__(self, level, show_path, show_time, rich_tracebacks, markup):
            super().__init__(level=level)
            self.options = (show_path, show_time, rich_tracebacks, markup)

    fake_module = types.ModuleType("rich.logging")
    fake_module.RichHandler = FakeRichHandler
    monkeypatch.setitem(sys.modules, "rich.logging", fake_module)

    spd_logging.configure_logging(level="WARNING", show_path=True, show_time=False)
    (handler,) = logging.getLogger("spd_learn").handlers
    assert isinstance(handler, FakeRichHandler)
    assert handler.level == logging.WARNING
    assert handler.options == (True, False, True, True)


def test_warn_once_emits_warning():
    with pytest.warns(RuntimeWarning, match="careful"):
        spd_logging.warn_once("careful", RuntimeWarning)


@pytest.mark.parametrize(
    "removal_version, expected",
    [("1.0.0", "It will be removed in version 1.0.0."), (None, None)],
)
def test_deprecation_helper_warns_once(removal_version, expected):
    helper = spd_logging.DeprecationHelper("old", "new", "0.2.0", removal_version)
    with pytest.warns(DeprecationWarning) as record:
        helper.warn()
    message = str(record[0].message)
    assert "'old' is deprecated since version 0.2.0. Use 'new' instead." in message
    if expected is None:
        assert "removed" not in message
    else:
        assert expected in message

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        helper.warn()  # second call is silent


def test_deprecated_decorator():
    @spd_logging.deprecated("old_func", "new_func", "0.2.0", "1.0.0")
    def old_func(x, y=1):
        """Docstring."""
        return x + y

    assert old_func.__name__ == "old_func"
    assert old_func.__doc__ == "Docstring."
    with pytest.warns(DeprecationWarning, match="old_func"):
        assert old_func(1, y=2) == 3
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert old_func(2) == 3
