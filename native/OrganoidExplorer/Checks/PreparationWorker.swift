import Foundation

@main struct PreparationChecks {
    @MainActor static func main() async throws {
        func require(_ value: @autoclosure () -> Bool, _ message: String) {
            if !value() { fatalError(message) }
        }
        let root = URL(fileURLWithPath: CommandLine.arguments[1]).deletingLastPathComponent().deletingLastPathComponent()
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("atlas-preparation-native-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let fixture = Process(); fixture.executableURL = root.appendingPathComponent(".venv/bin/python")
        fixture.currentDirectoryURL = root
        fixture.arguments = ["-c", """
import json,sys
from pathlib import Path
import numpy as np,tifffile
sys.path.insert(0,'scripts')
from prepare_atlas_dataset import inspect
p=Path(sys.argv[1]);z,y,x=np.indices((20,36,40));labels=np.zeros(z.shape,np.uint16)
labels[(z-10)**2+(y-16)**2+(x-15)**2<36]=1
labels[(z-10)**2+(y-16)**2+(x-25)**2<36]=2
raw=p/'raw.ome.tif';mask=p/'labels.tif'
tifffile.imwrite(raw,labels*1000,ome=True,metadata=dict(axes='ZYX',PhysicalSizeZ=2.,PhysicalSizeY=.5,PhysicalSizeX=.5),photometric='minisblack')
tifffile.imwrite(mask,labels,metadata={'axes':'ZYX'},photometric='minisblack')
npy=p/'raw.npy';np.save(npy,labels*1000)
a=inspect(npy);assert a['spacing']==[2.,1.,1.] and 'assume cubic stack' in a['calibrationSource'];a.update(spacing=[2.,.5,.5],labels=str(mask),labelStat=[mask.stat().st_size,mask.stat().st_mtime_ns])
r=dict(title='Native preparation check',inputs=[a],settings=dict(model='cpsam_v2',scale=.35,diameter=30.,gpu=False,background=True),excludeDetached=True,harmonics=False)
(p/'request.json').write_text(json.dumps(r))
""", temporary.path]
        try fixture.run(); fixture.waitUntilExit()
        require(fixture.terminationStatus == 0, "Fixture generation succeeds")
        let request = try JSONDecoder().decode(PreparationRequest.self, from: Data(contentsOf: temporary.appendingPathComponent("request.json")))
        require(PreparationCalibration.assumedCubicSpacing([64,512,512]) == [8,1,1], "Missing calibration makes a square stack cubic")
        require(PreparationCalibration.assumedCubicSpacing([64,256,512]) == [8,1,1], "Rectangular images retain equal XY pixel spacing")
        require(PreparationCalibration.assumedCubicSpacing([1024,256,512]) == [0.5,1,1], "Deep stacks also retain X = Y = 1")
        let estimatedGrid = OMEZarrLevel(index: 0, path: "0", dtype: "uint16", shape: [1,1,64,256,512], shapeZYX: [64,256,512], spacing: nil, origin: nil, chunkBytes: 1024)
        let calibratedGrid = OMEZarrLevel(index: 1, path: "1", dtype: "uint16", shape: [1,1,64,128,256], shapeZYX: [64,128,256], spacing: [3,0.5,0.75], origin: nil, chunkBytes: 1024)
        let image = OMEZarrImage(path: temporary.appendingPathComponent("image.ome.zarr").path, name: "Test image", fingerprint: "fixture", axes: ["t","c","z","y","x"], channels: ["membrane"], levels: [estimatedGrid,calibratedGrid], timepoints: 1, timestamps: [], labels: [], sourceStat: [0,0])
        let calibrationStore = DatasetPreparationStore(root: root)
        calibrationStore.addOMEZarr(image, channel: 0, level: 0, first: 0, last: 0, step: 1, label: nil)
        require(calibrationStore.inputs[0].spacing == [8,1,1] && calibrationStore.inputs[0].calibrationSource.contains("estimated"), "Native OME-Zarr selection estimates only Z when calibration is missing")
        calibrationStore.addOMEZarr(image, channel: 0, level: 1, first: 0, last: 0, step: 1, label: nil)
        require(calibrationStore.inputs[1].spacing == [3,0.5,0.75], "Known OME-Zarr calibration is never replaced by an estimate")
        require(request.settings.flowThreshold == 0.4, "Legacy settings retain Cellpose's default flow threshold")
        require(!request.settings.native3D, "Legacy requests retain u-Segment3D linking")
        var nativeSettings = request.settings
        nativeSettings.segmentationMode = "cellpose3d"; nativeSettings.flow3DSmooth = 1.5; nativeSettings.cellprobThreshold = -1; nativeSettings.minSize3D = 40
        let nativeRoundTrip = try JSONDecoder().decode(CellposeSettings.self, from: JSONEncoder().encode(nativeSettings))
        require(nativeRoundTrip == nativeSettings && nativeRoundTrip.native3D && nativeRoundTrip.valid3DSettings, "Native 3D controls survive saved requests")
        var thresholdSettings = request.settings; thresholdSettings.flowThreshold = 0.75
        let decodedThreshold = try JSONDecoder().decode(CellposeSettings.self, from: JSONEncoder().encode(thresholdSettings))
        require(decodedThreshold.flowThreshold == 0.75 && decodedThreshold != request.settings, "Flow threshold survives requests and marks previous slice results stale")
        let guide = try JSONDecoder().decode(PreparationSliceResult.self, from: Data("""
        {"path":"raw.tif","plane":"xy","note":"check","index":0,"nativeShape":[100,200],"spacing":[2,0.5,0.5]}
        """.utf8))
        let fitted = PreparationDiameterGuide.imageRect(image: CGSize(width: 400, height: 200), canvas: CGSize(width: 400, height: 300))
        require(fitted == CGRect(x: 0, y: 50, width: 400, height: 200), "Guide follows the fitted image, excluding letterboxing")
        require(PreparationDiameterGuide.displayDiameter(result: guide, imageRect: fitted, scale: 0.5, diameter: 30) == 120, "30 processed pixels at half scale cover 60 native pixels")
        require(PreparationDiameterGuide.displayDiameter(result: guide, imageRect: fitted, scale: 0.25, diameter: 30) == 240, "Changing processing scale updates the guide without reloading")
        let orthogonal = try JSONDecoder().decode(PreparationSliceResult.self, from: Data("""
        {"path":"raw.tif","plane":"yz","note":"check","index":0,"nativeShape":[25,100],"spacing":[2,0.5,0.5]}
        """.utf8))
        let square = PreparationDiameterGuide.imageRect(image: CGSize(width: 200, height: 200), canvas: CGSize(width: 400, height: 200))
        require(PreparationDiameterGuide.displayDiameter(result: orthogonal, imageRect: square, scale: 0.5, diameter: 30) == 120, "Anisotropic YZ previews use physical spacing and remain circular")
        require(PreparationDiameterGuide.displayDiameter(result: guide, imageRect: fitted, scale: 0, diameter: 30) == nil, "Invalid scale hides the guide")
        var big = request.inputs[0]
        big.shape = [578, 1920, 1920]; big.spacing = [0.75, 0.22872330084318845, 0.22872330084318845]; big.labels = nil
        let plan = PreparationMemoryPlan(input: big, scale: 0.85, totalMemory: 16 * 1024 * 1024 * 1024)!
        require(plan.shape == [1611, 1632, 1632] && !plan.fits && plan.suggestedScale == 0.17, "Memory plan matches Python's physical resampling and 16 GiB budget")
        require(request.linkingMode == nil, "Legacy preparation requests retain standard linking")
        var streamedRequest = request; streamedRequest.linkingMode = "streamed"
        let roundTrip = try JSONDecoder().decode(PreparationRequest.self, from: JSONEncoder().encode(streamedRequest))
        require(roundTrip.linkingMode == "streamed", "Linking choice survives saved requests")
        let streamed = PreparationMemoryPlan(input: big, scale: 0.85, totalMemory: 16 * 1024 * 1024 * 1024, linkingMode: "streamed")!
        require(!streamed.fits && streamed.suggestedScale! > plan.suggestedScale!, "Streaming permits a larger grid while retaining the memory guard")
        let planner = DatasetPreparationStore(root: root)
        planner.inputs = [big]; planner.settings.scale = 0.85; planner.confirmed = true
        let failed = temporary.appendingPathComponent("failed-preparation")
        try FileManager.default.createDirectory(at: failed, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: failed.appendingPathComponent("preparation.json"))
        planner.output = failed
        require(planner.memoryBlocked && !planner.valid, "An oversized grid cannot start")
        planner.useSuggestedScale()
        require(planner.settings.scale < 0.85 && !planner.memoryBlocked && !planner.confirmed, "Suggested scale requires calibration review")
        require(planner.output != failed && FileManager.default.fileExists(atPath: failed.appendingPathComponent("preparation.json").path), "Changed scale keeps failed checkpoints and chooses a fresh folder")
        let store = DatasetPreparationStore(root: root)
        store.title = request.title; store.inputs = request.inputs; store.settings = request.settings
        store.linkingMode = "streamed"; store.harmonics = false; store.output = temporary.appendingPathComponent("output")
        require(!store.valid, "Unconfirmed calibration cannot start")
        store.confirmed = true; require(store.valid, "Confirmed settings can start")
        store.start(); require(store.busy && store.preparing && !store.valid, "Native worker starts and locks edits")
        try await Task.sleep(nanoseconds: 400_000_000)
        store.cancel()
        let cancelledAt = Date()
        while store.busy && Date().timeIntervalSince(cancelledAt) < 20 { try await Task.sleep(nanoseconds: 100_000_000) }
        require(!store.busy && store.result == nil && store.error == nil, "Cancellation terminates the process without publishing a dataset")
        require(store.valid, "Cancelled preparation can resume")
        store.start()
        let startedAt = Date()
        while store.busy && Date().timeIntervalSince(startedAt) < 60 { try await Task.sleep(nanoseconds: 100_000_000) }
        require(!store.busy && store.result != nil && store.error == nil, "Native worker resumes and returns the completed dataset: \(store.error ?? store.log.suffix(1200).description)")
        require(store.progress == 1, "Completion updates progress")
        let saved = try JSONDecoder().decode(PreparationManifestCheck.self, from: Data(contentsOf: store.output!.appendingPathComponent("preparation.json")))
        require(saved.request.linkingMode == "streamed", "The native workflow sends and saves the selected linking mode")
        let dataset = try JSONDecoder().decode(Dataset.self, from: Data(contentsOf: store.result!.appendingPathComponent("dataset.json")))
        try dataset.validate()
        require(dataset.cells.count == 2 && dataset.samples.count == 1, "The NumPy result is valid for the real Atlas loader")
        store.reset()
        store.inputs = request.inputs
        store.previewSlice(input: request.inputs[0], plane: "xz", index: 12, segment: false)
        require(store.busy && store.previewing, "Native slice preview launches independently of dataset preparation")
        let previewStarted = Date()
        while store.busy && Date().timeIntervalSince(previewStarted) < 20 { try await Task.sleep(nanoseconds: 100_000_000) }
        require(store.sliceRaw != nil && store.sliceLabels == nil && store.result == nil && store.error == nil, "Raw slice preview is separate from dataset results")
        require(store.sliceResult?.plane == "xz" && store.sliceResult?.index == 12, "Native slice request uses the selected plane and index")
        store.clearSlice()
        require(store.sliceRaw == nil && store.sliceResult == nil, "Changing slice clears old preview assets")
        store.reset()
        require(store.inputs.isEmpty && store.result == nil && !store.valid, "The workflow can prepare another dataset")
        if let path = ProcessInfo.processInfo.environment["ATLAS_NATIVE3D_DATASET"] {
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            let nativeDataset = try JSONDecoder().decode(Dataset.self, from: data)
            try nativeDataset.validate()
            require(!nativeDataset.cells.isEmpty && nativeDataset.samples.count == 1, "The real native Cellpose 3D export loads in Atlas")
            print("PASS: real native Cellpose 3D Atlas dataset; \(nativeDataset.cells.count) cells.")
        }
        print("PASS: calibration gate; native process launch; cancellation; resume; progress; Atlas dataset validation; raw slice preview; reset.")
    }
}

private struct PreparationManifestCheck: Decodable { let request: PreparationRequest }
