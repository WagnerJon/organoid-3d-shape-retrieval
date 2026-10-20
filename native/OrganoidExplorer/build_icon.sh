#!/bin/zsh
set -eu
PACKAGE_DIR="${0:A:h}"
ICONSET="$PACKAGE_DIR/../../.cache/OrganoidAtlas.iconset"
mkdir -p "$ICONSET"
for size in 16 32 128 256 512; do
    sips -z "$size" "$size" "$PACKAGE_DIR/Resources/OrganoidAtlas.png" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    retina_size=$((size * 2))
    sips -z "$retina_size" "$retina_size" "$PACKAGE_DIR/Resources/OrganoidAtlas.png" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$PACKAGE_DIR/Resources/OrganoidAtlas.icns"
