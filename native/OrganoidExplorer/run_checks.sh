#!/bin/zsh
set -eu
PROJECT_ROOT="${0:A:h:h:h}"
PACKAGE_DIR="$PROJECT_ROOT/native/OrganoidExplorer"
source "$PACKAGE_DIR/toolchain.sh"
mkdir -p "$PROJECT_ROOT/.cache/explorer-direct"
CHECK_ENTRY="$PACKAGE_DIR/Checks/main.swift"
if [[ "${1:-}" == "--worker" ]]; then
    CHECK_ENTRY="$PACKAGE_DIR/Checks/RefinementWorker.swift"
elif [[ "${1:-}" == "--slices" ]]; then
    CHECK_ENTRY="$PACKAGE_DIR/Checks/SliceCacheWorker.swift"
elif [[ "${1:-}" == "--shapes" ]]; then
    CHECK_ENTRY="$PACKAGE_DIR/Checks/InteractiveShapeWorker.swift"
fi
swiftc -O -whole-module-optimization "${COMPILER_FLAGS[@]}" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/SliceCache.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/Model.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/ComparisonMetrics.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/ShapeAnalysis.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/PointCloud.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/SceneViewer.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/PansegSettings.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/FocusSettings.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/CellposeSettings.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/Refinement.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/Segmentation.swift" \
    "$CHECK_ENTRY" -o "$PROJECT_ROOT/.cache/explorer-direct/checks"
"$PROJECT_ROOT/.cache/explorer-direct/checks" "$PROJECT_ROOT/results/explorer_dataset"
