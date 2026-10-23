#!/bin/zsh
set -eu
PROJECT_ROOT="${0:A:h:h:h}"
PACKAGE_DIR="$PROJECT_ROOT/native/OrganoidExplorer"
APP_DIR="$PROJECT_ROOT/results/Organoid Atlas.app"
DEMO_ARCHIVE="$PACKAGE_DIR/Resources/DemoShapes.atlasdemo"
if [[ ! -f "$DEMO_ARCHIVE" ]]; then
    print -u2 "Missing bundled demo: $DEMO_ARCHIVE"
    exit 1
fi
source "$PACKAGE_DIR/toolchain.sh"
mkdir -p "$PROJECT_ROOT/.cache/explorer-direct"
swiftc -O -whole-module-optimization "${COMPILER_FLAGS[@]}" -module-name OrganoidExplorer "$PACKAGE_DIR"/Sources/OrganoidExplorer/*.swift -o "$PROJECT_ROOT/.cache/explorer-direct/OrganoidExplorer"
mkdir -p "$APP_DIR/Contents/MacOS" "$APP_DIR/Contents/Resources"
cp "$PROJECT_ROOT/.cache/explorer-direct/OrganoidExplorer" "$APP_DIR/Contents/MacOS/OrganoidExplorer"
cp "$PACKAGE_DIR/Resources/OrganoidAtlas.icns" "$PACKAGE_DIR/Resources/OrganoidAtlas.png" "$APP_DIR/Contents/Resources/"
cp "$DEMO_ARCHIVE" "$APP_DIR/Contents/Resources/"
/usr/bin/python3 - "$APP_DIR" "$PROJECT_ROOT/results/explorer_dataset" <<'PY'
import plistlib, sys
from pathlib import Path
app = Path(sys.argv[1])
info = dict(CFBundleExecutable='OrganoidExplorer', CFBundleIdentifier='org.local.OrganoidAtlas',
            CFBundleName='Organoid Atlas', CFBundleDisplayName='Organoid Atlas',
            CFBundleIconFile='OrganoidAtlas.icns',
            CFBundlePackageType='APPL', CFBundleShortVersionString='0.1.6', CFBundleVersion='1',
            NSLocalNetworkUsageDescription='Find paired Atlas workers and transfer microscopy stacks for processing on your local network.', NSBonjourServices=['_atlas-worker._tcp'], LSMinimumSystemVersion='14.0', NSHighResolutionCapable=True,
            ExplorerProjectPath=str(Path(sys.argv[2]).parent.parent), NSPrincipalClass='NSApplication')
info['UTImportedTypeDeclarations'] = [dict(UTTypeIdentifier='org.numpy.npy', UTTypeDescription='NumPy array', UTTypeConformsTo=['public.data'], UTTypeTagSpecification={'public.filename-extension':['npy'], 'public.mime-type':['application/x-npy']})]
(app/'Contents/Info.plist').write_bytes(plistlib.dumps(info))
PY
codesign --force --deep --sign - "$APP_DIR"
print "Built $APP_DIR"
