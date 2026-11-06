#!/bin/sh
# zero on Linux, for this user: ./build.sh && ./install.sh     ./install.sh --uninstall removes it.
set -e
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
bindir="${XDG_BIN_HOME:-$HOME/.local/bin}"
apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
desktop="$apps/zero.desktop"
if [ "$1" = "--uninstall" ]; then
  rm -f "$bindir/zero" "$bindir/zero-gui" "$desktop"
  echo "zero removed"
  exit 0
fi
if [ ! -x "$here/bin/zero" ] || [ ! -x "$here/bin/zero-gui" ]; then
  echo "build first: ./build.sh" >&2
  exit 1
fi
mkdir -p "$bindir" "$apps"
cp "$here/bin/zero" "$here/bin/zero-gui" "$bindir/"
cat > "$desktop" <<EOF
[Desktop Entry]
Type=Application
Name=zero
Comment=Compress files and folders, and extract .zero archives
Exec=$bindir/zero-gui %F
Terminal=false
Categories=Utility;Archiving;Compression;
EOF
echo "zero installed in $bindir"
case ":$PATH:" in *":$bindir:"*) ;; *) echo "add $bindir to your PATH to use the zero command" ;; esac
