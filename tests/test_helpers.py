"""Test helper functions."""

from pathlib import Path
from typing import Never

import pytest

from powerfit_em import helpers


@pytest.fixture
def fake_make_blob_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for make_blob_url without a real Pyodide/js runtime."""
    monkeypatch.setattr(
        helpers,
        "make_blob_url",
        lambda data, mime="application/octet-stream": f"blob:{data.decode()}",
    )


def test_pyodide_available_is_false_outside_pyodide() -> None:
    """No `js` module in a normal Python env."""
    assert helpers.pyodide_available() is False


def test_make_blob_url_raises_outside_pyodide() -> None:
    """Guard raises instead of failing on a missing `js` import."""
    with pytest.raises(RuntimeError, match="requires running under Pyodide"):
        helpers.make_blob_url(b"data")


@pytest.mark.usefixtures("fake_make_blob_url")
def test_patch_download_urls_rewrites_download_node(tmp_path: Path) -> None:
    """A `download` node's url is replaced with a blob url."""
    (tmp_path / "density.map").write_bytes(b"some-map-bytes")

    node = {"kind": "download", "params": {"url": "density.map"}, "children": []}
    helpers.patch_download_urls(node, tmp_path)

    assert node["params"]["url"] == "blob:some-map-bytes"


@pytest.mark.usefixtures("fake_make_blob_url")
def test_patch_download_urls_nested_children(tmp_path: Path) -> None:
    """Nested `download` nodes, several levels deep should all get patched."""
    (tmp_path / "a.pdb").write_bytes(b"AAA")
    (tmp_path / "b.pdb").write_bytes(b"BBB")

    tree = {
        "kind": "root",
        "children": [
            {
                "kind": "download",
                "params": {"url": "a.pdb"},
                "children": [
                    {
                        "kind": "parse",
                        "params": {},
                        "children": [
                            {"kind": "download", "params": {"url": "b.pdb"}, "children": None},
                        ],
                    },
                ],
            },
        ],
    }
    helpers.patch_download_urls(tree, tmp_path)

    outer = tree["children"][0]
    inner = outer["children"][0]["children"][0]

    assert outer["params"]["url"] == "blob:AAA"
    assert inner["params"]["url"] == "blob:BBB"


def test_patch_download_urls_non_download_node_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-`download` nodes are never passed to make_blob_url."""

    def fail_if_called(*_args, **_kwargs) -> Never:
        msg = "make_blob_url should not be called for non-download nodes"
        raise AssertionError(msg)

    monkeypatch.setattr(helpers, "make_blob_url", fail_if_called)

    node = {"kind": "parse", "params": {"format": "pdb"}, "children": None}
    helpers.patch_download_urls(node, tmp_path)

    assert node == {"kind": "parse", "params": {"format": "pdb"}, "children": None}


@pytest.mark.usefixtures("fake_make_blob_url")
def test_blobify_local_refs_rewrites_local_file_ref(tmp_path: Path) -> None:
    """A relative href to a file that exists in base_dir is rewritten."""
    (tmp_path / "fit_1.pdb").write_bytes(b"PDB")

    html = '<a href="fit_1.pdb">download</a>'
    result = helpers.blobify_local_refs(html, tmp_path)

    assert result == '<a href="blob:PDB">download</a>'


@pytest.mark.usefixtures("fake_make_blob_url")
def test_blobify_local_refs_rewrites_href_and_src(tmp_path: Path) -> None:
    """Both `href` and `src` attributes are rewritten."""
    (tmp_path / "state.mvsj").write_bytes(b"{}")

    html = '<a href="state.mvsj">a</a><img src="state.mvsj">'
    result = helpers.blobify_local_refs(html, tmp_path)

    assert result == '<a href="blob:{}">a</a><img src="blob:{}">'


def test_blobify_local_refs_leaves_remote_urls_untouched(tmp_path: Path) -> None:
    """http(s) urls (e.g. the molstar CDN script) are left alone."""
    html = '<script src="https://cdn.jsdelivr.net/npm/molstar@4.18.0/x.js"></script>'

    assert helpers.blobify_local_refs(html, tmp_path) == html


def test_blobify_local_refs_leaves_anchor_links_untouched(tmp_path: Path) -> None:
    """In-page `#anchor` links are left alone."""
    html = '<a href="#solutions-table">jump</a>'

    assert helpers.blobify_local_refs(html, tmp_path) == html


def test_blobify_local_refs_leaves_missing_local_file_untouched(tmp_path: Path) -> None:
    """A ref to a filename that doesn't exist in base_dir is left alone."""
    html = '<a href="does-not-exist.pdb">missing</a>'

    assert helpers.blobify_local_refs(html, tmp_path) == html
