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
a=inspect(raw);a.update(labels=str(mask),labelStat=[mask.stat().st_size,mask.stat().st_mtime_ns])
r=dict(title='Native preparation check',inputs=[a],settings=dict(model='cpsam_v2',scale=.35,diameter=30.,gpu=False,background=True),excludeDetached=True,harmonics=False)
(p/'request.json').write_text(json.dumps(r))
""", temporary.path]
        try fixture.run(); fixture.waitUntilExit()
        require(fixture.terminationStatus == 0, "Fixture generation succeeds")
        let request = try JSONDecoder().decode(PreparationRequest.self, from: Data(contentsOf: temporary.appendingPathComponent("request.json")))
        var big = request.inputs[0]
        big.shape = [578, 1920, 1920]; big.spacing = [0.75, 0.22872330084318845, 0.22872330084318845]; big.labels = nil
        let plan = PreparationMemoryPlan(input: big, scale: 0.85, totalMemory: 16 * 1024 * 1024 * 1024)!
        require(plan.shape == [1611, 1632, 1632] && !plan.fits && plan.suggestedScale == 0.17, "Memory plan matches Python's physical resampling and 16 GiB budget")
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
        store.harmonics = false; store.output = temporary.appendingPathComponent("output")
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
        let dataset = try JSONDecoder().decode(Dataset.self, from: Data(contentsOf: store.result!.appendingPathComponent("dataset.json")))
        try dataset.validate()
        require(dataset.cells.count == 2 && dataset.samples.count == 1, "The result is valid for the real Atlas loader")
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
        print("PASS: calibration gate; native process launch; cancellation; resume; progress; Atlas dataset validation; raw slice preview; reset.")
    }
}
