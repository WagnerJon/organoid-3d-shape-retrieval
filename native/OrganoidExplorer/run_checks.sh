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
elif [[ "${1:-}" == "--workers" ]]; then
    CHECK_ENTRY="$PACKAGE_DIR/Checks/WorkerDiscovery.swift"
elif [[ "${1:-}" == "--ome-zarr" ]]; then
    CHECK_ENTRY="$PACKAGE_DIR/Checks/OMEZarrWorker.swift"
elif [[ "${1:-}" == "--prepare" ]]; then
    CHECK_ENTRY="$PACKAGE_DIR/Checks/PreparationWorker.swift"
elif [[ "${1:-}" == "--plot-colors" ]]; then
    CHECK_ENTRY="$PACKAGE_DIR/Checks/PlotColorWorker.swift"
elif [[ "${1:-}" == "--plot-rendering" ]]; then
    CHECK_ENTRY="$PACKAGE_DIR/Checks/PlotRenderWorker.swift"
elif [[ "${1:-}" == "--plot-cells" ]]; then
    CHECK_ENTRY="$PACKAGE_DIR/Checks/PlotCellSurfaceWorker.swift"
fi
swiftc -O -whole-module-optimization "${COMPILER_FLAGS[@]}" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/SliceCache.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/Model.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/PlotColoring.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/PlotRendering.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/PlotRender2D.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/ComparisonMetrics.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/ShapeAnalysis.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/PointCloud.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/SceneViewer.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/PlotCellSurfaces.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/PansegSettings.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/FocusSettings.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/CellposeSettings.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/AtlasWorkers.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/DatasetPreparation.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/OMEZarrImport.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/DatasetSliceTest.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/Refinement.swift" \
    "$PACKAGE_DIR/Sources/OrganoidExplorer/Segmentation.swift" \
    "$CHECK_ENTRY" -o "$PROJECT_ROOT/.cache/explorer-direct/checks"
"$PROJECT_ROOT/.cache/explorer-direct/checks" "$PROJECT_ROOT/results/explorer_dataset"
