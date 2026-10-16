#!/usr/bin/env bash
# Build omph-tui's AppImage (#534): one file, no install, no `uv`, no checkout.
#
#     tools/appimage/build.sh [options]
#
#       --out FILE            the artifact (default: tools/appimage/dist/omph-tui-x86_64.AppImage)
#       --work-dir DIR        scratch space and the Python cache (default: $XDG_CACHE_HOME/omphalos/appimage)
#       --python VERSION      another CPython from that release (skips the sha256 check)
#       --refresh-requirements  re-resolve requirements.txt from requirements.in (needs the network)
#       --no-smoke            do not run the built artifact
#
# What goes in: a relocatable CPython (astral-sh/python-build-standalone's
# `install_only`, the release and sha256 pinned below), the TUI's runtime tree
# from requirements.txt (hashed, installed into that interpreter), and
# tools/omph_tui.py + tools/tui/. What stays out, deliberately: omph-server
# (ROCm, gfx1200, an 11 GiB .omph -- the AppRun looks for the host's), the
# weights, the vision encoder, and the rest of tools/pyproject.toml's
# dependencies (the TUI imports the standard library, Textual and Rich only).
#
# The build ends by running the artifact: `--screenshot` renders the real widgets
# headless and exits, so a bundle whose interpreter or Textual is broken fails
# here rather than on the user's first launch. It runs from the work directory,
# which has no checkout -- the AppImage has to stand on its own.
#
# Refreshing the Python pin: take the newest release at
# https://github.com/astral-sh/python-build-standalone/releases, update
# PB_RELEASE / PB_PYTHON, and `sha256sum` the install_only tarball into
# PB_SHA256. Refreshing the dependencies: --refresh-requirements.
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)

# The CPython the image carries: a build that relocates (it computes its prefix
# from the interpreter's own path), which is what an AppImage mount needs.
PB_RELEASE=20261013
PB_PYTHON=3.13.16
PB_SHA256=e588e50db882fa6a89f926d9a9ad64e445183af0eda5bda226a2c151b17db8a6

OUT=$HERE/dist/omph-tui-x86_64.AppImage
WORK=${XDG_CACHE_HOME:-$HOME/.cache}/omphalos/appimage
PYTHON=$PB_PYTHON
SMOKE=1
REFRESH=0

usage() {
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT=$2; shift 2 ;;
        --work-dir) WORK=$2; shift 2 ;;
        --python) PYTHON=$2; shift 2 ;;
        --refresh-requirements) REFRESH=1; shift ;;
        --no-smoke) SMOKE=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "build.sh: unknown option $1" >&2; usage >&2; exit 2 ;;
    esac
done

for tool in curl sha256sum tar uv rsvg-convert appimagetool; do
    command -v "$tool" >/dev/null 2>&1 || { echo "build.sh: $tool is missing" >&2; exit 1; }
done

BUNDLE_REL=usr/lib/omphalos
TARBALL="cpython-$PYTHON+$PB_RELEASE-x86_64-unknown-linux-gnu-install_only.tar.gz"
URL="https://github.com/astral-sh/python-build-standalone/releases/download/$PB_RELEASE/$TARBALL"
APPDIR=$WORK/AppDir
mkdir -p "$WORK"

# 1. the dependencies' pin (only on request: it needs the network and it moves)
if [ "$REFRESH" = 1 ]; then
    echo "build.sh: re-resolving requirements.txt from requirements.in"
    ( cd "$HERE" && uv pip compile --quiet --generate-hashes \
        --python-version "$(echo "$PYTHON" | cut -d. -f1,2)" -o requirements.txt requirements.in )
fi

# 2. the interpreter
if [ ! -f "$WORK/$TARBALL" ]; then
    echo "build.sh: fetching $TARBALL"
    curl -fL --retry 3 -o "$WORK/$TARBALL.part" "$URL"
    mv "$WORK/$TARBALL.part" "$WORK/$TARBALL"
fi
if [ "$PYTHON" = "$PB_PYTHON" ]; then
    echo "$PB_SHA256  $WORK/$TARBALL" | sha256sum -c --status - \
        || { echo "build.sh: $TARBALL is not the pinned one (sha256 mismatch)" >&2; exit 1; }
else
    echo "build.sh: --python $PYTHON, not the pin ($PB_PYTHON): no sha256 to check" >&2
fi

# 3. the AppDir
rm -rf "$APPDIR"
mkdir -p "$APPDIR/$BUNDLE_REL"
tar -xzf "$WORK/$TARBALL" -C "$APPDIR/$BUNDLE_REL" --strip-components=1
PY=$APPDIR/$BUNDLE_REL/bin/python3
"$PY" -c 'import sys; assert sys.version_info[:3] == tuple(int(p) for p in "'"$PYTHON"'".split("."))' \
    || { echo "build.sh: the tarball is not CPython $PYTHON" >&2; exit 1; }

# 4. the dependency tree, into that interpreter (`--system`: it is not a venv,
#    and it must stay relocatable, which a venv's absolute paths are not).
#    Bytecode is compiled into the image: the mount is read-only, so nothing
#    would be cached at the first launch otherwise.
echo "build.sh: installing the pinned requirements into the bundle"
uv pip install --quiet --python "$PY" --system --compile-bytecode --require-hashes \
    -r "$HERE/requirements.txt"
"$PY" -c 'from importlib.metadata import version; print("textual", version("textual"), "/ rich", version("rich"))'

# 5. the TUI itself, byte for byte the repository's
install -Dm644 "$ROOT/tools/omph_tui.py" "$APPDIR/$BUNDLE_REL/tools/omph_tui.py"
mkdir -p "$APPDIR/$BUNDLE_REL/tools/tui"
install -m644 "$ROOT/tools/tui/"*.py "$APPDIR/$BUNDLE_REL/tools/tui/"
install -Dm644 "$ROOT/LICENSE" "$APPDIR/$BUNDLE_REL/LICENSE"

# 6. the desktop entries: icon (from the SVG, shapes only), .DirIcon, .desktop
rsvg-convert -w 256 -h 256 "$HERE/omph-tui.svg" -o "$APPDIR/omph-tui.png"
ln -sf omph-tui.png "$APPDIR/.DirIcon"
install -Dm644 "$HERE/omph-tui.desktop" "$APPDIR/omph-tui.desktop"
install -Dm755 "$HERE/AppRun" "$APPDIR/AppRun"

# 7. pack it (VERSION lands in the Embedded section: `git describe` says which
#    tree this was built from)
mkdir -p "$(dirname "$OUT")"
VERSION=$(git -C "$ROOT" describe --tags --always --dirty 2>/dev/null || echo "0.0.0")
echo "build.sh: appimagetool, version $VERSION"
ARCH=x86_64 VERSION="$VERSION" appimagetool --no-appstream "$APPDIR" "$OUT"

# 8. run it, from a directory with no checkout in sight
if [ "$SMOKE" = 1 ]; then
    SVG=$WORK/smoke.svg
    rm -f "$SVG"
    echo "build.sh: smoke test in $WORK (no checkout, no server, no GPU)"
    ( cd "$WORK" && APPIMAGE_EXTRACT_AND_RUN=1 "$OUT" --screenshot "$SVG" )
    [ -s "$SVG" ] || { echo "build.sh: the artifact rendered nothing" >&2; exit 1; }
    echo "build.sh: rendered $(wc -c < "$SVG") bytes of widgets"
fi

ls -l "$OUT"
sha256sum "$OUT"
