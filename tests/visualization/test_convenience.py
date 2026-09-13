"""Tests for tralda.visualization.plot_tree and Tree.plot (tralda.visualization._convenience)."""

from __future__ import annotations

import pytest

from tralda.visualization import plot_tree

# ===========================================================================
# Matplotlib backend
# ===========================================================================


class TestMatplotlibBackend:
    @pytest.fixture(autouse=True)
    def _setup(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        yield
        plt.close("all")

    def test_returns_fig_and_ax(self, small_tree):
        from matplotlib.axes import Axes
        from matplotlib.figure import Figure

        fig, ax = plot_tree(small_tree, backend="matplotlib")
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    @pytest.mark.parametrize("suffix", [".png", ".pdf", ".svg"])
    def test_saves_file_to_path(self, small_tree, tmp_path, suffix):
        path = tmp_path / f"tree{suffix}"
        plot_tree(small_tree, backend="matplotlib", path=str(path))
        assert path.exists()
        assert path.stat().st_size > 0

    def test_show_calls_figure_show(self, small_tree, monkeypatch):
        from matplotlib.figure import Figure

        calls = []
        monkeypatch.setattr(Figure, "show", lambda self, *a, **kw: calls.append(True))
        plot_tree(small_tree, backend="matplotlib", show=True)
        assert calls == [True]

    def test_forwards_layout_kwargs(self, small_tree):
        fig, _ax = plot_tree(small_tree, backend="matplotlib", layout_mode="vertical")
        assert isinstance(fig.get_size_inches()[0], float)

    def test_tree_plot_method_delegates(self, small_tree, tmp_path):
        path = tmp_path / "tree.png"
        _fig, _ax = small_tree.plot(backend="matplotlib", path=str(path))
        assert path.exists()


# ===========================================================================
# Plotly backend
# ===========================================================================


class TestPlotlyBackend:
    @pytest.fixture(autouse=True)
    def _setup(self):
        pytest.importorskip("plotly")

    def test_returns_go_figure(self, small_tree):
        import plotly.graph_objects as go

        fig = plot_tree(small_tree, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_saves_html_file(self, small_tree, tmp_path):
        path = tmp_path / "tree.html"
        plot_tree(small_tree, backend="plotly", path=str(path))
        assert path.exists()
        assert path.stat().st_size > 0

    def test_show_calls_figure_show(self, small_tree, monkeypatch):
        import plotly.graph_objects as go

        calls = []
        monkeypatch.setattr(go.Figure, "show", lambda self, *a, **kw: calls.append(True))
        plot_tree(small_tree, backend="plotly", show=True)
        assert calls == [True]

    def test_tree_plot_method_delegates(self, small_tree, tmp_path):
        path = tmp_path / "tree.html"
        small_tree.plot(backend="plotly", path=str(path))
        assert path.exists()

    def test_png_export_requires_kaleido_gives_clear_import_error(
        self, small_tree, monkeypatch, tmp_path
    ):
        import plotly.graph_objects as go

        def _fake_write_image(self, path, *a, **kw):
            raise ValueError('Image export using the "kaleido" engine requires the Kaleido package')

        monkeypatch.setattr(go.Figure, "write_image", _fake_write_image)
        with pytest.raises(ImportError, match=r"tralda\[plotly\]"):
            plot_tree(small_tree, backend="plotly", path=str(tmp_path / "tree.png"))

    def test_unrelated_write_image_error_is_not_swallowed(self, small_tree, monkeypatch, tmp_path):
        import plotly.graph_objects as go

        def _fake_write_image(self, path, *a, **kw):
            raise ValueError("some unrelated failure")

        monkeypatch.setattr(go.Figure, "write_image", _fake_write_image)
        with pytest.raises(ValueError, match="unrelated failure"):
            plot_tree(small_tree, backend="plotly", path=str(tmp_path / "tree.png"))

    def test_real_png_export_with_kaleido(self, small_tree, tmp_path):
        pytest.importorskip("kaleido")
        path = tmp_path / "tree.png"
        plot_tree(small_tree, backend="plotly", path=str(path))
        assert path.exists()
        assert path.stat().st_size > 0


# ===========================================================================
# Backend validation
# ===========================================================================


class TestInvalidBackend:
    def test_unknown_backend_raises_value_error(self, small_tree):
        with pytest.raises(ValueError, match="unknown backend"):
            plot_tree(small_tree, backend="bogus")

    def test_tree_plot_unknown_backend_raises_value_error(self, small_tree):
        with pytest.raises(ValueError, match="unknown backend"):
            small_tree.plot(backend="bogus")
