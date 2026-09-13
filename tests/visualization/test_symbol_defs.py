"""Tests for tralda.visualization._symbol_defs (symbol resolution, linestyle resolution)."""

from __future__ import annotations

import pytest

from tralda.visualization._symbol_defs import (
    SYMBOL_DEF_BY_NAME,
    SymbolKind,
    get_symbol_def,
    resolve_linestyle,
    resolve_symbol,
)

# ===========================================================================
# resolve_symbol
# ===========================================================================


class TestResolveSymbol:
    @pytest.mark.parametrize(
        "name", ["circle", "square", "triangle-up", "star", "none", "dot", "cap"]
    )
    def test_canonical_name_returned_unchanged(self, name):
        assert resolve_symbol(name) == name

    @pytest.mark.parametrize(
        "mpl_marker,expected",
        [
            ("s", "square"),
            ("^", "triangle-up"),
            ("v", "triangle-down"),
            ("<", "triangle-left"),
            (">", "triangle-right"),
            ("*", "star"),
            ("D", "diamond"),
            ("p", "pentagon"),
            ("h", "hexagon"),
            ("P", "cross"),
            ("X", "x-mark"),
        ],
    )
    def test_unambiguous_mpl_marker_resolves(self, mpl_marker, expected):
        assert resolve_symbol(mpl_marker) == expected

    def test_ambiguous_mpl_marker_o_resolves_to_simple_circle(self):
        """'o' is shared by circle, dot, circle-dot, circle-inner-ring; SIMPLE wins."""
        assert resolve_symbol("o") == "circle"

    def test_ambiguous_mpl_marker_pipe_resolves_to_cap(self):
        assert resolve_symbol("|") == "cap"

    def test_ambiguous_plotly_marker_circle_resolves_to_simple_circle(self):
        assert resolve_symbol("circle") == "circle"

    def test_plotly_marker_line_ns_resolves_to_cap(self):
        assert resolve_symbol("line-ns") == "cap"

    def test_plotly_marker_circle_dot_resolves_to_circle_dot(self):
        assert resolve_symbol("circle-dot") == "circle-dot"

    def test_unknown_name_returned_unchanged(self):
        assert resolve_symbol("totally-unknown-symbol") == "totally-unknown-symbol"


# ===========================================================================
# get_symbol_def
# ===========================================================================


class TestGetSymbolDef:
    def test_none_returns_invisible(self):
        assert get_symbol_def(None) is SYMBOL_DEF_BY_NAME["none"]

    def test_empty_string_returns_invisible(self):
        assert get_symbol_def("") is SYMBOL_DEF_BY_NAME["none"]

    def test_unknown_name_returns_invisible(self):
        assert get_symbol_def("nonexistent") is SYMBOL_DEF_BY_NAME["none"]

    def test_known_name_returns_matching_def(self):
        assert get_symbol_def("star").name == "star"
        assert get_symbol_def("star").kind is SymbolKind.SIMPLE

    def test_resolves_alternate_notation(self):
        assert get_symbol_def("^").name == "triangle-up"

    @pytest.mark.parametrize(
        "name,kind",
        [
            ("none", SymbolKind.INVISIBLE),
            ("dot", SymbolKind.EDGE_STYLE),
            ("cap", SymbolKind.EDGE_STYLE),
            ("circle", SymbolKind.SIMPLE),
            ("circle-dot", SymbolKind.COMPOSITE),
            ("circle-inner-ring", SymbolKind.COMPOSITE),
        ],
    )
    def test_symbol_kind(self, name, kind):
        assert get_symbol_def(name).kind is kind


# ===========================================================================
# resolve_linestyle
# ===========================================================================


class TestResolveLinestyle:
    @pytest.mark.parametrize(
        "mpl,plotly",
        [("-", "solid"), ("--", "dash"), (":", "dot"), ("-.", "dashdot")],
    )
    def test_mpl_to_plotly(self, mpl, plotly):
        assert resolve_linestyle(mpl, target="plotly") == plotly

    @pytest.mark.parametrize(
        "mpl,plotly",
        [("-", "solid"), ("--", "dash"), (":", "dot"), ("-.", "dashdot")],
    )
    def test_plotly_to_mpl(self, mpl, plotly):
        assert resolve_linestyle(plotly, target="mpl") == mpl

    @pytest.mark.parametrize("plotly", ["solid", "dash", "dot", "dashdot"])
    def test_plotly_name_passthrough_when_target_is_plotly(self, plotly):
        assert resolve_linestyle(plotly, target="plotly") == plotly

    @pytest.mark.parametrize("mpl", ["-", "--", ":", "-."])
    def test_mpl_shorthand_passthrough_when_target_is_mpl(self, mpl):
        assert resolve_linestyle(mpl, target="mpl") == mpl

    def test_default_target_is_plotly(self):
        assert resolve_linestyle("--") == "dash"

    def test_unknown_linestyle_passthrough(self):
        assert resolve_linestyle("unknown-style", target="plotly") == "unknown-style"
        assert resolve_linestyle("unknown-style", target="mpl") == "unknown-style"

    @pytest.mark.parametrize("mpl", ["-", "--", ":", "-."])
    def test_round_trip_mpl_to_plotly_to_mpl(self, mpl):
        plotly_name = resolve_linestyle(mpl, target="plotly")
        assert resolve_linestyle(plotly_name, target="mpl") == mpl
