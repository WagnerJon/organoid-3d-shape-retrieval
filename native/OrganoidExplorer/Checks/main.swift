import Foundation
import AppKit
import SceneKit
import Combine

func require(_ value: @autoclosure () -> Bool, _ message: String) {
    if !value() { fatalError(message) }
}
require(abs(representationDistance([1,0],[0,1],metric:"cosine")-1) < 1e-10, "Orthogonal cosine")
require(representationDistance([1,0],[5,0],metric:"cosine") < 1e-10, "Magnitude invariant cosine")
require(abs(representationDistance([0,0],[3,4],metric:"euclidean")-5) < 1e-10, "Euclidean distance")
require(representationDistance([0,0],[3,4],metric:"cosine").isInfinite, "Undefined cosine rejected")
require(representationDistance([1],[1,2],metric:"euclidean").isInfinite, "Dimension mismatch rejected")
require(adjustedRandIndex([0,0,1,1], [5,5,8,8]) == 1, "ARI ignores group numbering")
require(abs(adjustedRandIndex([0,0,1,1], [0,1,0,1])! + 0.5) < 1e-10, "Disagreeing partitions")
require(adjustedRandIndex([0,0,0], [1,1,1]) == 1, "Degenerate identical partitions")
require(adjustedRandIndex([0,1,2], [3,4,5]) == 1, "Singleton partitions")
require(adjustedRandIndex([0], [0]) == nil, "Insufficient ARI coverage")
require(adjustedRandIndex([0,1], [0]) == nil, "ARI length mismatch")
require(abs(representationDistance([2,2,3,0],[0,0,1,2],metric:"precomputed")-2) < 1e-12, "CAJAL lookup uses global IDs")
require(representationDistance([2,2,3,0],[0,0,1,9],metric:"precomputed").isInfinite, "CAJAL rejects asymmetric cache")
require(representationDistance([9,0],[0,0],metric:"precomputed").isInfinite, "CAJAL rejects invalid IDs")
print("PASS: CAJAL pairwise distances retain global identities and validate symmetric caches.")
// The native comparison must use the same orbit/fallback distance as export.
var shapeA = [Double](repeating: 0, count: 476)
var shapeB = shapeA
for frame in 0..<4 { shapeA[1+frame*95] = Double(frame*10); shapeB[1+frame*95] = Double((3-frame)*10) }
require(representationDistance(shapeA,shapeB,metric:"shape_hybrid95") == 0, "Shape orbit sign permutation")
shapeB[0] = 1; shapeB[381] = 3; shapeB[382] = 4
require(abs(representationDistance(shapeA,shapeB,metric:"shape_hybrid95")-5) < 1e-10, "Ambiguous shape uses signature")
require(abs(representationDistance(shapeB,shapeA,metric:"shape_hybrid95")-5) < 1e-10, "Fallback is symmetric")
require(representationDistance([0],[0],metric:"shape_hybrid95").isInfinite, "Reject invalid hybrid dimensions")
print("PASS: native spatial orbit and ambiguity fallback distances.")
let folder = URL(fileURLWithPath: CommandLine.arguments[1])
let dataset = try JSONDecoder().decode(Dataset.self, from: Data(contentsOf: folder.appendingPathComponent("dataset.json")))
try dataset.validate()
require(dataset.cells.count == 1088, "All cells present")
require(dataset.samples.count == 108, "All stacks present")
var meshes = 0
for sample in dataset.samples {
    let volume = try VolumeData(folder: folder.appendingPathComponent(sample.id), sample: sample)
    let cells = dataset.cells.filter { $0.sample == sample.id }
    let labels = Set(cells.map(\.label))
    require(Set(volume.labels.filter { $0 > 0 }.map(Int.init)).isSubset(of: labels), "Unknown preview labels")
    let z = sample.shape[0]-1, y = sample.shape[1]-1, x = sample.shape[2]-1
    let last = sample.shape.reduce(1,*)-1
    require(volume.index(x:x,y:y,slice:z,axis:0) == last, "XY index")
    require(volume.index(x:x,y:z,slice:y,axis:1) == last, "XZ index")
    require(volume.index(x:y,y:z,slice:x,axis:2) == last, "YZ index")
    for axis in 0..<3 {
        require(volume.image(axis:axis,fraction:0.5,selected:cells[0].label,overlay:true,contrast:1) != nil, "Raw image creation")
    }
    for cell in cells {
        let mesh = try MeshData(url: folder.appendingPathComponent(sample.id).appendingPathComponent("\(cell.label).mesh"))
        require(!mesh.vertices.isEmpty && mesh.faces.count % 3 == 0, "Mesh topology")
        require(mesh.vertices.allSatisfy { $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }, "Finite vertices")
        for method in dataset.methods {
            if let representation = cell.representations[method.id] {
                require(representation.vector.count == method.dimensions, "Representation association")
            } else { require(method.family == "validation" || method.scope == "cells", "Partial coverage requires an explicit scope") }
        }
        meshes += 1
    }
}
let temp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
try Data([0,1]).write(to:temp)
do { _ = try MeshData(url:temp); fatalError("Accepted truncated mesh") } catch { }
try FileManager.default.removeItem(at:temp)
print("PASS: vector metrics, all-axis slice indices and rendering, 108 stacks, \(meshes) meshes, \(dataset.methods.count) aligned representations, corrupt mesh rejection.")

let reference = "traditional"
let visible = dataset.cells.filter { $0.qc && $0.includedInAnalysis }
if dataset.methods.contains(where: { $0.coverage == "all_cells" }) {
    let full = dataset.methods.filter { $0.coverage == "all_cells" }
    require(full.count == 16, "All frozen shape readouts published")
    require(full.allSatisfy { method in dataset.cells.allSatisfy { $0.representations[method.id] != nil } }, "Full inference covers every cell, including review-only exclusions")
    let overview = dataset.methods.filter { $0.family != "validation" }
    require(visible.allSatisfy { item in overview.allSatisfy { item.representations[$0.id] != nil } }, "Every eligible cell enters the full-method overview")
    print("PASS: full frozen inference across all cells and all eligible cells shared by every overview method; validation reference remains separate.")
}
let agreements = compareMethods(methods: dataset.methods, reference: reference, items: visible,
                                query: visible.first, otherStacksOnly: true)
require(agreements.count == dataset.methods.count, "Agreement for each method")
let baseline = agreements.first { $0.id == reference }!
require(baseline.ari == 1 && baseline.neighborIntersection == 10, "Reference self-agreement")
require(agreements.allSatisfy { result in
    let shared = visible.filter { $0.representations[result.id] != nil }
    let queryAvailable = visible.first?.representations[result.id] != nil
    return result.sharedCount == shared.count && result.ari != nil && result.neighborCount == (queryAvailable ? 10 : 0)
}, "Comparison respects each method’s coverage")
if let validationMethod = dataset.methods.first(where: { $0.id == "cajal_gw_validation" }) {
    let cohort = visible.filter { $0.representations[validationMethod.id] != nil }
    require(cohort.count == 48, "Unquantized CAJAL is scoped to validation")
    let selfComparison = compareMethods(methods:[validationMethod],reference:validationMethod.id,items:cohort,query:cohort.first,otherStacksOnly:true)
    require(selfComparison.first?.neighborIntersection == 10 && selfComparison.first?.ari == 1, "Validation CAJAL self-agreement")
    print("PASS: partial validation coverage and unquantized CAJAL reference self-agreement.")
}
let oneStack = visible.filter { $0.sample == visible.first!.sample }
let emptyNeighbors = compareMethods(methods: dataset.methods, reference: reference, items: oneStack,
                                   query: visible.first, otherStacksOnly: true)
require(emptyNeighbors.allSatisfy { $0.neighborIntersection == nil && $0.neighborCount == 0 }, "No fabricated overlap with no candidates")
for result in agreements { print("ARI \(result.id): \(result.ari ?? 0) · neighbors \(result.neighborIntersection ?? 0)/\(result.neighborCount)") }
print("PASS: label-invariant ARI, degenerate partitions, insufficient coverage, reference self-agreement, shared comparison cohorts and empty candidate sets.")

// Segmentation previews share their grid across all planes and uint32 label IDs.
let segmentationSourceURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(".cache/segmentation-check/source/source.json")
try MainActor.assumeIsolated {
if FileManager.default.fileExists(atPath: segmentationSourceURL.path) {
    let segSource = try JSONDecoder().decode(SegSource.self, from: Data(contentsOf:segmentationSourceURL))
    let segmentation = SegmentationStore()
    segmentation.source = segSource
    segmentation.raw = Array(try Data(contentsOf:URL(fileURLWithPath:segSource.folder).appendingPathComponent("raw.u8")))
    segmentation.labelsA = [UInt32](repeating:900000,count:segmentation.raw.count)
    segmentation.labelsB = [UInt32](repeating:7000000,count:segmentation.raw.count)
    segmentation.agreement = SegAgreement(dice:1,meanIoU:1,matched:1,countA:1,countB:1,pairs:[SegPair(a:900000,b:7000000,iou:1)],note:"test")
    segmentation.select(label:900000,side:1)
    let selectedPartner = segmentation.selectedB
    require(selectedPartner == 7000000,"Segmentation selection uses matched IDs")
    segmentation.slice = 1
    for axis in 0..<3 {
        segmentation.axis = axis
        let (w,h,_) = segmentation.dimensions()
        let lastIndex = segmentation.index(w-1,h-1), voxelCount = segmentation.raw.count
        require(lastIndex == voxelCount-1,"Shared segmentation plane indexing")
        for side in 0...3 { let image = segmentation.image(side:side); require(image != nil,"Segmentation raw/overlays/disagreement image") }
    }
    segmentation.consumeOutput("@atlas-pro")
    segmentation.consumeOutput("gress {\"stage\":\"Cellpose XY\",\"fraction\":0.1,\"span\":0.2}\n\r 50%|█████     | 30/")
    segmentation.consumeOutput("60 [00:20]\r")
    let halfway = segmentation.progress
    require(abs(halfway-0.2) < 1e-9,"Fragmented stage JSON and tqdm progress")
    segmentation.consumeOutput("@atlas-progress {\"stage\":\"Aggregation\",\"fraction\":0.7,\"span\":0}\n\r 100%|█████| 100/100\r")
    let aggregation = segmentation.progress
    require(abs(aggregation-0.7) < 1e-9,"Unknown-duration stages ignore unrelated progress bars")
    segmentation.consumeOutput("@atlas-progress {\"stage\":\"Invalid\",\"fraction\":NaN,\"span\":0}\n")
    let finite = segmentation.progress.isFinite
    require(finite,"Malformed progress cannot break the native bar")
    print("PASS: fragmented progress events, completed-slice updates and stage transitions.")
    let workspace = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let trialFolder = workspace.appendingPathComponent("trial")
    try FileManager.default.createDirectory(at:trialFolder,withIntermediateDirectories:true)
    defer { try? FileManager.default.removeItem(at:workspace) }
    let fixtureID = UUID().uuidString
    let testSource = SegSource(path:segSource.path,fingerprint:fixtureID,folder:segSource.folder,dtype:segSource.dtype,shape:segSource.shape,nativeShape:segSource.nativeShape,spacing:segSource.spacing,low:segSource.low,high:segSource.high)
    let tuning = SegmentationStore(projectRoot:workspace)
    tuning.source = testSource; tuning.raw = segmentation.raw
    let labelData = segmentation.labelsA.withUnsafeBytes { Data($0) }
    try labelData.write(to:trialFolder.appendingPathComponent("labels.u32"))
    try Data(repeating:128,count:segmentation.raw.count).write(to:trialFolder.appendingPathComponent("boundaries.u8"))
    try Data([0]).write(to:trialFolder.appendingPathComponent("boundaries.tif"))
    var savedSettings = PansegSettings(); savedSettings.beta = 0.6; savedSettings.rescaleZ = 0.5
    let manifest: [String:Any] = ["id":"trial-fixture","method":"panseg","source":fixtureID,"folder":trialFolder.path,
        "provenance":"test","created":"test","labelHash":"test","shape":testSource.nativeShape,"spacing":testSource.spacing,
        "cells":[],"seconds":1,"purpose":"trial","settings":try savedSettings.json(),
        "prediction":["key":"test","hash":"test","shape":testSource.nativeShape,"spacing":testSource.spacing,"device":"mps"]]
    let manifestData = try JSONSerialization.data(withJSONObject:manifest)
    try manifestData.write(to:trialFolder.appendingPathComponent("result.json"))
    let completed = try JSONDecoder().decode(SegResult.self,from:manifestData)
    tuning.chooseTrial(completed); tuning.trialSettings = savedSettings
    let initialComparison = tuning.resultB == nil
    require(initialComparison,"Tuning a trial does not replace the comparison run")
    let reuseInitial = tuning.canReusePrediction
    require(reuseInitial,"Identical inference settings allow reuse")
    tuning.trialSettings.threshold = 0.7
    let reuseSegmentation = tuning.canReusePrediction
    require(reuseSegmentation,"Watershed changes retain boundary reuse")
    tuning.trialSettings.gaussianSigma = 1
    let reusePreprocessing = tuning.canReusePrediction
    require(!reusePreprocessing,"Preprocessing changes require new prediction")
    let transferred = tuning.transferTrial()
    let transferredBeta = tuning.comparisonSettings.beta, transferredSigma = tuning.comparisonSettings.gaussianSigma
    let transferredID = tuning.resultB?.id
    require(transferred && transferredID == completed.id,"Transfer selects the exact saved trial")
    require(transferredBeta == 0.6 && transferredSigma == 0,"Transfer uses saved settings, excluding unsaved edits")
    tuning.presetName = "test preset"; tuning.savePreset()
    let presetCount = tuning.presets.count, presetSettings = tuning.presets.first?.settings
    require(presetCount == 1 && presetSettings == tuning.trialSettings,"Preset round trip preserves all settings")
    let migrated = try PansegSettings.fromJSON(["multicut":false,"minSize":55])
    require(migrated.clustering == "watershed" && migrated.postMinSize == 55,"Old PanSeg settings remain usable")
    let trialImage = tuning.image(side:5), probabilityImage = tuning.image(side:6)
    require(trialImage != nil && probabilityImage != nil,"Workbench instance and probability rendering")
    var focusSettings = FocusSettings(); focusSettings.cell_radius = 23; focusSettings.checkpoint = "/test/model.pth"
    var focusManifest = manifest
    focusManifest["id"] = "focus-fixture"; focusManifest["method"] = "focus3d"; focusManifest["purpose"] = "comparison"
    focusManifest["settings"] = try focusSettings.json()
    let focusFolder = workspace.appendingPathComponent("focus")
    try FileManager.default.createDirectory(at:focusFolder,withIntermediateDirectories:true)
    focusManifest["folder"] = focusFolder.path
    let focusData = try JSONSerialization.data(withJSONObject:focusManifest)
    try focusData.write(to:focusFolder.appendingPathComponent("result.json"))
    try labelData.write(to:focusFolder.appendingPathComponent("labels.u32"))
    let focusResult = try JSONDecoder().decode(SegResult.self,from:focusData)
    tuning.results = [completed,focusResult]
    tuning.choose(focusResult)
    let focusChosen = tuning.comparisonMethod == "focus3d" && tuning.resultB?.id == focusResult.id
    let focusRadius = tuning.focusSettings.cell_radius
    require(focusChosen && focusRadius == 23,"FOCUS-3D run restores its own settings")
    tuning.setComparisonMethod("panseg")
    let restoredPanseg = tuning.resultB?.id == completed.id && tuning.comparisonSettings.beta == 0.6
    require(restoredPanseg,"Switching restores transferred PanSeg run")
    tuning.setComparisonMethod("focus3d")
    // No Cellpose run is selected in this fixture; verify focus preview independently.
    let restoredFocus = tuning.resultB?.id == focusResult.id && tuning.labelsB.count == segmentation.raw.count
    require(restoredFocus,"Switching restores FOCUS-3D preview")
    let invalidMethodBefore = tuning.comparisonMethod
    tuning.setComparisonMethod("unknown")
    let ignoredUnknown = tuning.comparisonMethod == invalidMethodBefore
    require(ignoredUnknown,"Unknown comparison methods rejected")
    tuning.consumeOutput("@atlas-progress {\"stage\":\"FOCUS-3D: stitching\",\"fraction\":0.8,\"span\":0}\n")
    let focusProgress = tuning.progress
    require(focusProgress == 0.8,"FOCUS-3D progress reaches native bar")
    UserDefaults.standard.removeObject(forKey:"pansegComparison.\(fixtureID)")
    UserDefaults.standard.removeObject(forKey:"focus3dComparison.\(fixtureID)")
    print("PASS: FOCUS-3D settings, run restoration, linked preview state and progress.")
    var cpResults: [SegResult] = []
    for (side,model) in [("A","cpsam_v2"),("B","cpdino-vitb")] {
        let cpFolder = workspace.appendingPathComponent("cellpose-"+side)
        try FileManager.default.createDirectory(at:cpFolder,withIntermediateDirectories:true)
        var cpManifest = focusManifest
        cpManifest["id"] = "cellpose-"+side; cpManifest["method"] = "cellpose"
        cpManifest["cellposeModel"] = model; cpManifest["comparisonSide"] = side; cpManifest["folder"] = cpFolder.path
        var controls = CellposeSettings(); controls.model = model; controls.scale = side == "A" ? 0.1 : 0.2
        cpManifest["settings"] = try controls.json()
        let data = try JSONSerialization.data(withJSONObject:cpManifest)
        try data.write(to:cpFolder.appendingPathComponent("result.json")); try labelData.write(to:cpFolder.appendingPathComponent("labels.u32"))
        cpResults.append(try JSONDecoder().decode(SegResult.self,from:data))
    }
    tuning.results += cpResults
    tuning.choose(cpResults[0],side:1); tuning.choose(cpResults[1],side:2)
    let separateCP = tuning.resultA?.id == cpResults[0].id && tuning.resultB?.id == cpResults[1].id
    let separateSettings = tuning.cellposeA.model == "cpsam_v2" && tuning.cellposeB.model == "cpdino-vitb" && tuning.cellposeA.scale == 0.1 && tuning.cellposeB.scale == 0.2
    require(separateCP && separateSettings,"Cellpose A/B runs and controls remain independent")
    tuning.setComparisonMethod("focus3d"); tuning.setComparisonMethod("cellpose")
    let restoredCP = tuning.resultA?.id == cpResults[0].id && tuning.resultB?.id == cpResults[1].id
    require(restoredCP,"Switching methods restores Cellpose B without replacing A")
    let originalCP = try CellposeSettings.fromJSON(["model":"cpdino","background_correction":false])
    require(originalCP.model == "cpdino" && !originalCP.background,"Imported reconstruction settings retain model/background")
    require(cpResults[1].title.contains("cpdino-vitb"),"Run history names the actual model")
    UserDefaults.standard.removeObject(forKey:"cellposeComparisonA.\(fixtureID)")
    UserDefaults.standard.removeObject(forKey:"cellposeComparisonB.\(fixtureID)")
    UserDefaults.standard.removeObject(forKey:"focus3dComparison.\(fixtureID)")
    print("PASS: independent Cellpose A/B models and settings, saved-run restoration and imported settings.")
    let refinedFolder = workspace.appendingPathComponent("refined")
    try FileManager.default.createDirectory(at:refinedFolder,withIntermediateDirectories:true)
    let parentData = try Data(contentsOf:URL(fileURLWithPath:cpResults[1].folder).appendingPathComponent("result.json"))
    try parentData.write(to:refinedFolder.appendingPathComponent("parent.json"))
    var childManifest = try JSONSerialization.jsonObject(with:parentData) as! [String:Any]
    childManifest["id"] = "refined-fixture"; childManifest["method"] = "refined"; childManifest["folder"] = refinedFolder.path
    childManifest["refinement"] = ["parentID":cpResults[1].id,"parentFolder":cpResults[1].folder,"parentMethod":"cellpose","changedVoxels":10,"initialLoss":0.2,"finalLoss":0.1,"rejectedCells":[]] as [String:Any]
    let childData = try JSONSerialization.data(withJSONObject:childManifest)
    try childData.write(to:refinedFolder.appendingPathComponent("result.json")); try labelData.write(to:refinedFolder.appendingPathComponent("labels.u32"))
    let child = try JSONDecoder().decode(SegResult.self,from:childData)
    tuning.results.append(child); tuning.showRefinement(child)
    let originalRefined = tuning.comparisonMethod == "refined" && tuning.resultA?.id == cpResults[1].id && tuning.resultB?.id == child.id
    let originalOnly = tuning.canRefine(side:1) && !tuning.canRefine(side:2)
    require(originalRefined && originalOnly,"Refinement restores exact parent and blocks recursive refinement")
    tuning.setComparisonMethod("cellpose")
    let returnedModelComparison = tuning.resultA?.id == cpResults[0].id && tuning.resultB?.id == cpResults[1].id
    require(returnedModelComparison,"Leaving refinement restores the earlier model comparison")
    tuning.setComparisonMethod("refined")
    let returnedRefinement = tuning.resultA?.id == cpResults[1].id && tuning.resultB?.id == child.id
    require(returnedRefinement,"Refinement history restores original/refined pair")
    tuning.setComparisonMethod("cellpose")
    tuning.resultA = nil; tuning.labelsA = []
    tuning.showRefinement(child); tuning.showRefinement(child)
    tuning.setComparisonMethod("cellpose")
    let noInventedOriginal = tuning.resultA == nil && tuning.comparisonMethod == "cellpose"
    require(noInventedOriginal,"Refinement remembers an empty prior A selection across history changes")
    let postProcessing = SegmentationStore(projectRoot:workspace)
    postProcessing.comparisonMethod = "refined"
    postProcessing.adoptPreparedSource(testSource,raw:segmentation.raw)
    let adopted = postProcessing.source?.fingerprint == testSource.fingerprint && postProcessing.spacing == testSource.spacing
    require(adopted,"Post processing adopts shared raw data and exact calibration")
    postProcessing.results = tuning.results
    postProcessing.choose(cpResults[1],side:1);postProcessing.showRefinement(child)
    let isolatedModels = tuning.resultA == nil && tuning.comparisonMethod == "cellpose"
    require(isolatedModels,"Post processing selection does not change model comparison")
    postProcessing.agreement = SegAgreement(dice:1,meanIoU:1,matched:1,countA:1,countB:1,pairs:[SegPair(a:900000,b:123,iou:1)],note:"wrong assignment fixture")
    postProcessing.select(label:900000,side:1)
    let identitySelected = postProcessing.selectedA == 900000 && postProcessing.selectedB == 900000
    require(identitySelected,"Before/after selection follows preserved identity, not overlap matching")
    postProcessing.labelsB = [UInt32](repeating:0,count:postProcessing.raw.count)
    postProcessing.slice = 0.5
    for plane in 0..<3 {
        postProcessing.axis = plane
        let changedImage = postProcessing.image(side:7)
        require(changedImage != nil,"Changed-voxel images render in every plane")
        let bytes = changedImage!.dataProvider!.data! as Data
        require(bytes[0] > bytes[1] && bytes[2] > bytes[1],"Removed foreground is magenta in change view")
    }
    UserDefaults.standard.removeObject(forKey:"refinedComparison.\(fixtureID)")
    print("PASS: independent post-processing state, source/calibration adoption, identity-linked selection and all-plane change overlays.")
    UserDefaults.standard.removeObject(forKey:"refinedComparison.\(fixtureID)")
    UserDefaults.standard.removeObject(forKey:"cellposeComparisonA.\(fixtureID)")
    UserDefaults.standard.removeObject(forKey:"cellposeComparisonB.\(fixtureID)")
    print("PASS: original/refined pairing, recursive refinement prevention and model comparison restoration.")
    print("PASS: trial isolation, inference reuse eligibility, exact-run transfer, preset round trips and workbench rendering.")
    print("PASS: native segmentation manifest, uint32 instance IDs, overlap-linked selection and all-plane raw/overlay/disagreement rendering.")
}
}

try MainActor.assumeIsolated {
    let store = ExplorerStore()
    store.dataset = dataset
    store.qcOnly = false
    let included = dataset.cells.filter(\.includedInAnalysis)
    let excluded = dataset.cells.filter { !$0.includedInAnalysis }
    require(included.count == 765 && excluded.count == 323, "Native membership classification retained")
    require(store.filtered.count == included.count, "Exclusion applies even with QC filtering off")
    store.selection = Set(excluded.map(\.id))
    require(store.measurementItems.isEmpty, "Marked detached cells cannot enter measurements")
    store.selectedID = excluded[0].id
    require(store.selected != nil && store.analysisSelected == nil, "Excluded cells inspectable but cannot be analysis queries")
    require(store.neighbors(method:"traditional").isEmpty, "No neighbors for an excluded query")
    let comparison = compareMethods(methods:dataset.methods,reference:"traditional",items:dataset.cells,query:excluded[0],otherStacksOnly:false)
    require(comparison.allSatisfy { result in result.sharedCount == included.filter { $0.representations[result.id] != nil }.count && result.neighborCount == 0 }, "Comparison rejects excluded observations and queries at the computation boundary")
    store.selection = []; store.level = "Organoids"
    for stack in store.filtered {
        let members = included.filter { $0.sample == stack.sample }
        require(stack.cellCount == members.count, "Organoid cell count includes main tissue only")
        require(abs(stack.volume - members.reduce(0) { $0 + $1.volume }) < 1e-7, "Organoid volume excludes detached cells")
    }
}
print("PASS: detached cells inspectable but excluded from tables, neighbors, comparison metrics and organoid aggregates.")

MainActor.assumeIsolated {
    let store = ExplorerStore()
    store.dataset = dataset
    for level in ["Cells", "Organoids"] {
        store.level = level; store.changeLevel()
        let rows = Array(store.filtered.prefix(3))
        require(rows.count == 3, "Linked marking fixture")
        let group = Set(rows.map(\.id))
        store.markPlotItems(group)
        require(store.selectedTableRows == group, "Every brushed point has a highlighted table row")
        require(Set(store.measurementItems.map(\.id)) == group, "Measurements retain selected-only behavior")
        store.selectTableRows([rows[1].id])
        require(store.selectedID == rows[1].id && store.selection == group, "Row inspection preserves the brushed cohort")
        require(store.selectedTableRows == [rows[1].id], "Table inspection updates its native highlight")
        store.markPlotItems([rows[2].id])
        require(store.selectedTableRows == [rows[2].id] && store.selectedID == rows[2].id, "Plot click replaces old marking and focuses the corresponding row")
        store.markPlotItems([])
        require(store.selectedTableRows.isEmpty && store.measurementItems.count == store.filtered.count, "Clicking plot background clears marking and restores all rows")
        store.selection = group
        require(store.selectedTableRows == group, "Other marking controls also synchronize table highlights")
        store.selection = []; store.markedOnly = false
    }
    store.level = "Cells"
    let excluded = dataset.cells.first { !$0.includedInAnalysis }!
    store.markPlotItems([excluded.id])
    require(store.selection.isEmpty && store.selectedTableRows.isEmpty, "Excluded cells cannot enter linked plot selection")
}
print("PASS: PCA clicks, brush groups, native multi-row highlights, row inspection and clearing in cell and organoid views.")

MainActor.assumeIsolated {
    let sample = dataset.samples.first!
    let cells = dataset.cells.filter { $0.sample == sample.id }
    let excluded = cells.first { !$0.includedInAnalysis }!
    let included = cells.first { $0.includedInAnalysis }!
    var config = ReconstructionView(folder:folder,sample:sample,cells:cells,selectedLabel:excluded.label,hideExcluded:true)
    let coordinator = ReconstructionView.Coordinator()
    coordinator.nodes = [excluded.label:SCNNode(),included.label:SCNNode()]
    coordinator.updateMaterials(config:config)
    require(coordinator.nodes[excluded.label]!.isHidden, "Excluded surface hidden even when selected")
    require(!coordinator.nodes[included.label]!.isHidden, "Included surface remains visible")
    config.hideExcluded = false
    coordinator.updateMaterials(config:config)
    require(!coordinator.nodes[excluded.label]!.isHidden, "Excluded surface restored when option is off")
}
let previewSample = dataset.samples.first!
let previewVolume = try VolumeData(folder:folder.appendingPathComponent(previewSample.id),sample:previewSample)
let hiddenLabels = Set(dataset.cells.filter { $0.sample == previewSample.id && !$0.includedInAnalysis }.map(\.label))
let voxel = previewVolume.labels.firstIndex { hiddenLabels.contains(Int($0)) }!
let width = previewSample.shape[2], height = previewSample.shape[1]
let z = voxel / (width*height), y = (voxel/width) % height, x = voxel % width
for plane in 0..<3 {
    let position = [z,y,x][plane]
    let depth = previewVolume.dimensions(axis:plane).2
    let fraction = min(1,(Double(position)+0.01)/Double(max(1,depth-1)))
    let label = Int(previewVolume.labels[voxel])
    let base = previewVolume.image(axis:plane,fraction:fraction,selected:label,overlay:false,contrast:1)!.dataProvider!.data! as Data
    let hidden = previewVolume.image(axis:plane,fraction:fraction,selected:label,overlay:true,contrast:1,excluded:hiddenLabels,hideExcluded:true)!.dataProvider!.data! as Data
    let shown = previewVolume.image(axis:plane,fraction:fraction,selected:label,overlay:true,contrast:1,excluded:hiddenLabels)!.dataProvider!.data! as Data
    let pixel = [y*width+x,z*width+x,z*height+y][plane]*4
    require(hidden[pixel..<(pixel+4)] == base[pixel..<(pixel+4)], "Hiding overlay preserves original grayscale microscopy in every plane")
    require(shown[pixel..<(pixel+4)] != hidden[pixel..<(pixel+4)], "Excluded label visible again when hiding is off")
}
print("PASS: hide excluded surfaces, selected exclusions, restore visibility and raw-preserving overlays in all three planes.")

// Replaying an identical sweep isolates slice-render work from disk access and UI layout.
let sweepVolume = try VolumeData(folder:folder.appendingPathComponent(dataset.samples[0].id),sample:dataset.samples[0])
let sweepDepth = sweepVolume.dimensions(axis:0).2
let preparationStarted = Date()
sweepVolume.prepareSlices(style:VolumeSliceStyle(axis:0,selected:1,overlay:true,contrast:1),near:0.5)
print(String(format:"SLICE CACHE PREPARATION: %.2f ms",Date().timeIntervalSince(preparationStarted)*1000))
let sweepStarted = Date()
for _ in 0..<3 {
    for frame in 0..<sweepDepth {
        _ = sweepVolume.image(axis:0,fraction:(Double(frame)+0.01)/Double(max(1,sweepDepth-1)),selected:1,overlay:true,contrast:1)
    }
}
let sweepMilliseconds = Date().timeIntervalSince(sweepStarted)*1000
print(String(format:"SLICE BENCHMARK: %d frames, %.2f ms total, %.4f ms/frame",sweepDepth*3,sweepMilliseconds,sweepMilliseconds/Double(sweepDepth*3)))

// No setting may accidentally reuse another setting's pixels.
let cacheSample = dataset.samples[0]
let cacheFolder = folder.appendingPathComponent(cacheSample.id)
let cachedPreview = try VolumeData(folder:cacheFolder,sample:cacheSample)
let cacheExcluded = Set(dataset.cells.filter { $0.sample == cacheSample.id && !$0.includedInAnalysis }.map(\.label))
for plane in 0..<3 {
    let styles = [
        VolumeSliceStyle(axis:plane,selected:1,overlay:true,contrast:1),
        VolumeSliceStyle(axis:plane,selected:2,overlay:true,contrast:1),
        VolumeSliceStyle(axis:plane,selected:2,overlay:true,contrast:2),
        VolumeSliceStyle(axis:plane,selected:2,overlay:false,contrast:2),
        VolumeSliceStyle(axis:plane,selected:2,overlay:true,contrast:1,excluded:cacheExcluded),
        VolumeSliceStyle(axis:plane,selected:2,overlay:true,contrast:1,excluded:cacheExcluded,hideExcluded:true)
    ]
    for style in styles {
        cachedPreview.prepareSlices(style:style,near:0.5)
        let cached = cachedPreview.image(style:style,fraction:0.5)!
        require(cached === cachedPreview.image(style:style,fraction:0.5), "Warm slice returns the existing image")
        let fresh = try VolumeData(folder:cacheFolder,sample:cacheSample)
        let expected = fresh.image(style:style,fraction:0.5)!
        require((cached.dataProvider!.data! as Data) == (expected.dataProvider!.data! as Data), "Cache preserves plane, highlight, contrast, overlay and membership pixels")
    }
}
MainActor.assumeIsolated {
    let store = ExplorerStore()
    var analysisSignals = 0, sliceSignals = 0
    let analysisObserver = store.objectWillChange.sink { analysisSignals += 1 }
    let sliceObserver = store.sliceNavigation.objectWillChange.sink { sliceSignals += 1 }
    for step in 0..<100 { store.slice = Double(step)/99 }
    require(analysisSignals == 0 && sliceSignals == 100, "Scrubbing notifies slice views without invalidating analyses")
    require(store.slice == 1, "Slice compatibility accessors use the isolated navigation state")
    analysisObserver.cancel(); sliceObserver.cancel()
    let cells = dataset.cells.filter { $0.sample == cacheSample.id }
    var config = ReconstructionView(folder:folder,sample:cacheSample,cells:cells,selectedLabel:1)
    let coordinator = ReconstructionView.Coordinator()
    coordinator.nodes = [1:SCNNode()]
    coordinator.updateMaterials(config:config)
    let appearance = coordinator.appliedAppearance, disabledClip = coordinator.appliedClip
    config.slice = 0.9
    require(config.appearance == appearance && config.clipState == disabledClip, "Unclipped Z scrolling requires no SceneKit material or clipping updates")
    config.clipping = true
    coordinator.updateMaterials(config:config)
    let enabledClip = coordinator.appliedClip
    config.slice = 0.2
    coordinator.updateMaterials(config:config)
    require(coordinator.appliedAppearance == appearance && coordinator.appliedClip != enabledClip, "3D clipping advances without rebuilding materials")
}
print("PASS: cached pixels match fresh renders for every plane and display setting; slice scrubbing leaves analyses and 3D materials unchanged.")

for cell in dataset.cells {
    for path in cell.pointReconstructions?.values ?? Dictionary<String,String>().values {
        let cloud = try PointCloudData(url:folder.appendingPathComponent(path))
        require(cloud.vertices.count == 2048,"Model reconstruction contains 2048 surface points")
        require(cloud.geometry().elements.first?.primitiveType == .point,"Model reconstruction is rendered as points")
    }
}
let pointFixture = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
var pointBytes = Data()
var pointCount = UInt32(2)
withUnsafeBytes(of:&pointCount) { pointBytes.append(contentsOf:$0) }
let coordinates: [Float] = [1,2,3,4,5,6]
coordinates.withUnsafeBufferPointer { pointBytes.append(Data(buffer:$0)) }
try pointBytes.write(to:pointFixture)
let decodedPoints = try PointCloudData(url:pointFixture)
require(decodedPoints.vertices.count == 2 && decodedPoints.vertices[1].z == 6,"Physical XYZ point decoding")
try Data([0,1]).write(to:pointFixture)
do { _ = try PointCloudData(url:pointFixture); fatalError("Accepted a truncated point cloud") } catch { }
try FileManager.default.removeItem(at:pointFixture)
print("PASS: physical point-cloud reconstruction loading, finite coordinates, point geometry and truncated artifact rejection.")
