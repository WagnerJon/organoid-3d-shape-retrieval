# Sourced by the local build/check scripts. Never changes the installed toolchain.
TOOLCHAIN_CACHE="$PROJECT_ROOT/.cache/explorer-toolchain"
mkdir -p "$TOOLCHAIN_CACHE"
typeset -a COMPILER_FLAGS
COMPILER_FLAGS=(-swift-version 5 -target arm64-apple-macosx14.0 -module-cache-path "$PROJECT_ROOT/.cache/clang")
if [[ -n "${EXPLORER_SDK:-}" ]]; then
    COMPILER_FLAGS+=(-sdk "$EXPLORER_SDK")
elif [[ -d /Library/Developer/CommandLineTools/SDKs/MacOSX15.5.sdk ]]; then
    COMPILER_FLAGS+=(-sdk /Library/Developer/CommandLineTools/SDKs/MacOSX15.5.sdk)
fi
if [[ -f /Library/Developer/CommandLineTools/usr/include/swift/module.modulemap && -f /Library/Developer/CommandLineTools/usr/include/swift/bridging.modulemap ]]; then
    /usr/bin/python3 - "$TOOLCHAIN_CACHE" <<'PY'
from pathlib import Path
import json, sys
p = Path(sys.argv[1]).resolve()
(p/'empty.modulemap').write_text('// Hide obsolete duplicate SwiftBridging declaration for this build only.\n')
(p/'overlay.json').write_text(json.dumps({'version':0, 'roots':[
    {'type':'file', 'name':'/Library/Developer/CommandLineTools/usr/include/swift/module.modulemap',
     'external-contents':str(p/'empty.modulemap')}]}))
PY
    COMPILER_FLAGS+=(-vfsoverlay "$TOOLCHAIN_CACHE/overlay.json" -Xcc -ivfsoverlay -Xcc "$TOOLCHAIN_CACHE/overlay.json")
fi
