import SwiftUI
import AppKit
import UniformTypeIdentifiers

private let atlasStackTypes: [UTType] = [.tiff, UTType(filenameExtension: "npy") ?? UTType(importedAs: "org.numpy.npy", conformingTo: .data)]

enum PreparationCalibration {
    static let estimatedSource = "Missing calibration; assume cubic stack (estimated): X = Y = 1 nominal µm, Z = longest XY dimension / slice count. Absolute physical scale is unknown (editable)."
    static func assumedCubicSpacing(_ shape: [Int]) -> [Double] {
        guard shape.count == 3, shape.allSatisfy({ $0 > 0 }) else { return [1,1,1] }
        return [Double(max(shape[1],shape[2])) / Double(shape[0]), 1, 1]
    }
}

struct PreparationInput: Codable, Identifiable {
    var path: String
    var name: String
    var shape: [Int]
    var axes: String
    var dtype: String
    var spacing: [Double]
    var calibrationSource: String
    var sourceStat: [Int64]
    var labels: String? = nil
    var labelStat: [Int64]? = nil
    var omeZarr: OMEZarrSelection? = nil
    var tiffSelection: TIFFSelection? = nil
    var hasProvidedLabels: Bool { labels != nil || omeZarr?.label != nil }
    var id: String {
        if let t = tiffSelection { return path + "|series=\(t.series)|t=\(t.time)|c=\(t.channel)" }
        guard let z = omeZarr else { return path }
        return path + "|t=\(z.time)|c=\(z.channel)|level=\(z.level)"
    }
}
private struct InspectedInput: Decodable {
    let path, name, axes, dtype, calibrationSource: String
    let shape: [Int]
    let spacing: [Double]?
    let sourceStat: [Int64]
    var input: PreparationInput { PreparationInput(path: path, name: name, shape: shape, axes: axes, dtype: dtype,
        spacing: spacing ?? PreparationCalibration.assumedCubicSpacing(shape),
        calibrationSource: spacing == nil ? PreparationCalibration.estimatedSource : calibrationSource, sourceStat: sourceStat) }
}
private struct InspectionResult: Decodable {
    let inputs: [InspectedInput]
    let tiffImages: [TIFFImage]?
    let ignored: [String]?
}
struct PreparationRequest: Codable {
    var title: String
    var inputs: [PreparationInput]
    var settings: CellposeSettings
    var excludeDetached: Bool
    var harmonics: Bool
    var execution: AtlasExecution? = nil
    var linkingMode: String? = nil
    var segmentationMethod: String? = nil
    var focusSettings: FocusSettings? = nil
}
private struct PreparationManifest: Decodable { let request: PreparationRequest }
private struct PreparationProgress: Decodable { let stage: String; let fraction: Double }

struct PreparationMemoryPlan {
    let shape: [Int]
    let imageGiB, estimatedPeakGiB, budgetGiB, estimatedDiskGiB, scratchGiB: Double
    let fits: Bool
    let suggestedScale: Double?
    init?(input: PreparationInput, scale: Double, totalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory, linkingMode: String = "standard") {
        guard input.shape.count == 3, input.spacing.count == 3, input.shape.allSatisfy({ $0 > 0 }),
              input.spacing.allSatisfy({ $0.isFinite && $0 > 0 }), scale.isFinite, (0.05...1).contains(scale) else { return nil }
        let gib = 1_073_741_824.0, minimum = input.spacing.min()!
        func dimensions(_ factor: Double) -> [Int]? {
            let values = zip(input.shape, input.spacing).map { floor(Double($0.0) * (factor * ($0.1 / minimum))) }
            guard values.allSatisfy({ $0.isFinite && $0 >= 0 && $0 < Double(Int.max) }) else { return nil }
            return values.map(Int.init)
        }
        guard let dimensions = dimensions(scale) else { return nil }
        shape = dimensions
        let voxels = shape.reduce(1.0) { $0 * Double($1) }
        imageGiB = voxels * 4 / gib
        let allowance = linkingMode == "streamed" ? 48.0 : 256.0
        estimatedPeakGiB = 1.25 + voxels * allowance / gib
        estimatedDiskGiB = (144 * voxels + 4 * input.shape.reduce(1.0) { $0 * Double($1) }) / gib + 1
        scratchGiB = 24 * voxels / gib
        budgetGiB = min(11.5, 0.65 * Double(totalMemory) / gib)
        fits = (shape.min() ?? 0) >= 2 && estimatedPeakGiB <= budgetGiB
        var suggested: Double?
        // Match the conservative backend planning allowance, not an accuracy guarantee.
        for step in stride(from: min(100, Int(floor(scale * 100))), through: 5, by: -1) {
            let candidate = Double(step) / 100
            let dims = zip(input.shape, input.spacing).map { floor(Double($0.0) * (candidate * ($0.1 / minimum))) }
            if (dims.min() ?? 0) >= 2 && 1.25 + dims.reduce(1, *) * allowance / gib <= budgetGiB { suggested = candidate; break }
        }
        suggestedScale = suggested
    }
}

struct FocusPreparationMemoryPlan {
    let shape: [Int]
    let estimatedPeakGiB, budgetGiB: Double
    var fits: Bool { estimatedPeakGiB <= budgetGiB }
    init?(input: PreparationInput, settings: FocusSettings, totalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory) {
        guard settings.valid, input.shape.count == 3, input.spacing.count == 3,
              input.spacing.allSatisfy({ $0.isFinite && $0 > 0 }) else { return nil }
        let factor = settings.cell_radius / 15, zFactor = input.spacing[0] / input.spacing[1] > 1 ? 1 : factor
        shape = zip(input.shape, [zFactor,factor,factor]).map { max(1,Int((Double($0.0)/$0.1).rounded(.toNearestOrEven))) }
        let padded = zip(zip(shape,[32,96,96]),[settings.stride_z,settings.stride_y,settings.stride_x]).map { max($0.0.1,$0.0.0+2*$0.1) }
        estimatedPeakGiB = 6 + (32*padded.reduce(1.0) { $0*Double($1) } + 16*input.shape.reduce(1.0) { $0*Double($1) }) / 1_073_741_824
        budgetGiB = min(11.5,0.65*Double(totalMemory)/1_073_741_824)
    }
}

@MainActor final class DatasetPreparationStore: ObservableObject {
    let workers = AtlasWorkers()
    @Published var title = "New organoid dataset"
    @Published var inputs: [PreparationInput] = []
    @Published var settings = CellposeSettings(model: "cpsam_v2", scale: 0.35)
    @Published var segmentationMethod = "cellpose"
    @Published var focusSettings = FocusSettings()
    @Published var linkingMode = "standard"
    @Published var excludeDetached = true
    @Published var harmonics = true
    @Published var confirmed = false
    @Published var zarrCatalog: OMEZarrCatalog?
    @Published var tiffCatalog: TIFFCatalog?
    @Published var output: URL?
    @Published private(set) var busy = false
    @Published private(set) var preparing = false
    @Published private(set) var previewing = false
    @Published private(set) var sliceRaw: NSImage?
    @Published private(set) var sliceLabels: NSImage?
    @Published private(set) var sliceResult: PreparationSliceResult?
    private var sliceFolder: URL?
    private var operation = ""
    @Published private(set) var cancelling = false
    @Published private(set) var progress = 0.0
    @Published private(set) var stage = "Choose raw microscopy stacks to begin."
    @Published private(set) var log = ""
    @Published private(set) var result: URL?
    @Published var error: String?
    @Published private(set) var workerProgress: [Int: String] = [:]
    private var process: Process?
    private var pending = ""
    private var token = UUID()
    private let root: URL
    init(root: URL? = nil) {
        self.root = root ?? URL(fileURLWithPath: Bundle.main.object(forInfoDictionaryKey: "ExplorerProjectPath") as? String ?? FileManager.default.currentDirectoryPath)
        focusSettings.checkpoint = UserDefaults.standard.string(forKey: "focus3dCheckpoint") ?? ""
    }
    var usesFocus: Bool { segmentationMethod == "focus3d" }
    var focusInstalled: Bool {
        FileManager.default.isExecutableFile(atPath: root.appendingPathComponent(".cache/focus3d-env/bin/python").path) &&
        FileManager.default.fileExists(atPath: root.appendingPathComponent("vendor/focus3d/src/focus3d/segmentation/FOCUS3D/inference_win.py").path)
    }
    var focusReady: Bool {
        focusInstalled && focusSettings.valid && FileManager.default.fileExists(atPath: focusSettings.checkpoint) &&
        (focusSettings.config.isEmpty || FileManager.default.fileExists(atPath: focusSettings.config)) &&
        inputs.filter { !$0.hasProvidedLabels }.allSatisfy { $0.spacing.count == 3 && abs($0.spacing[1]-$0.spacing[2]) <= 0.0001*max($0.spacing[1],$0.spacing[2]) }
    }
    var largestFocusMemoryPlan: FocusPreparationMemoryPlan? {
        inputs.filter { !$0.hasProvidedLabels }.compactMap { FocusPreparationMemoryPlan(input: $0, settings: focusSettings) }.max { $0.estimatedPeakGiB < $1.estimatedPeakGiB }
    }
    var largestMemoryPlan: PreparationMemoryPlan? {
        inputs.filter { !$0.hasProvidedLabels }.compactMap { PreparationMemoryPlan(input: $0, scale: settings.scale, linkingMode: linkingMode) }.max { $0.estimatedPeakGiB < $1.estimatedPeakGiB }
    }
    var memoryBlocked: Bool {
        inputs.enumerated().contains { index, input in
            if usesFocus {
                guard !input.hasProvidedLabels, let plan = FocusPreparationMemoryPlan(input: input, settings: focusSettings), !plan.fits else { return false }
                let completed = output?.appendingPathComponent(String(format: "stacks/Stack_%03d/aggregate.completed.json", index + 1))
                return completed.map { !FileManager.default.fileExists(atPath: $0.path) } ?? true
            }
            guard !input.hasProvidedLabels, let plan = PreparationMemoryPlan(input: input, scale: settings.scale, linkingMode: linkingMode), !plan.fits else { return false }
            let completed = output?.appendingPathComponent(String(format: "stacks/Stack_%03d/aggregate.completed.json", index + 1))
            return completed.map { !FileManager.default.fileExists(atPath: $0.path) } ?? true
        }
    }
    var diskEstimateGiB: Double {
        inputs.filter { !$0.hasProvidedLabels }.compactMap { PreparationMemoryPlan(input: $0, scale: settings.scale, linkingMode: linkingMode)?.estimatedDiskGiB }.reduce(0, +)
    }
    var availableDiskGiB: Double? {
        guard var url = output else { return nil }
        while !FileManager.default.fileExists(atPath: url.path), url.path != "/" { url.deleteLastPathComponent() }
        guard let attributes = try? FileManager.default.attributesOfFileSystem(forPath: url.path), let bytes = attributes[.systemFreeSize] as? NSNumber else { return nil }
        return bytes.doubleValue / 1_073_741_824
    }
    func useSuggestedScale() {
        guard !busy, let value = largestMemoryPlan?.suggestedScale, value < settings.scale else { return }
        clearSlice(); settings.scale = value; confirmed = false; error = nil
        // Changed settings must not adopt a checkpoint folder from a different scale.
        if let original = output, FileManager.default.fileExists(atPath: original.appendingPathComponent("preparation.json").path) {
            let suffix = String(format: "-scale-%.2f", value)
            var candidate = original.deletingLastPathComponent().appendingPathComponent(original.lastPathComponent + suffix)
            var counter = 2
            while FileManager.default.fileExists(atPath: candidate.path) {
                candidate = original.deletingLastPathComponent().appendingPathComponent(original.lastPathComponent + suffix + "-\(counter)"); counter += 1
            }
            output = candidate
        }
        stage = "Lower scale selected. Retest the cell diameter at this scale and confirm calibration. A new folder is used if earlier checkpoints exist."
    }
    var valid: Bool {
        !busy && !memoryBlocked && result == nil && confirmed && !title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !inputs.isEmpty && output != nil &&
        inputs.allSatisfy { $0.spacing.count == 3 && $0.spacing.allSatisfy { $0.isFinite && $0 > 0 } } &&
        (usesFocus ? (inputs.allSatisfy(\.hasProvidedLabels) || focusReady) :
         settings.scale.isFinite && (0.05...1).contains(settings.scale) && settings.diameter.isFinite && (2...500).contains(settings.diameter) && settings.valid3DSettings && settings.validModel)
    }
    func reset() {
        guard !busy else { return }
        clearSlice(); zarrCatalog = nil; tiffCatalog = nil; title = "New organoid dataset"; inputs = []; settings = CellposeSettings(model: "cpsam_v2", scale: 0.35)
        segmentationMethod = "cellpose"; focusSettings = FocusSettings(); focusSettings.checkpoint = UserDefaults.standard.string(forKey: "focus3dCheckpoint") ?? ""
        linkingMode = "standard"; excludeDetached = true; harmonics = true; confirmed = false; output = nil; result = nil
        progress = 0; workerProgress = [:]; error = nil; log = ""; stage = "Choose raw microscopy stacks to begin."
    }
    func clearSlice() {
        guard !busy else { return }
        sliceRaw = nil; sliceLabels = nil; sliceResult = nil
        if let folder = sliceFolder { try? FileManager.default.removeItem(at: folder) }
        sliceFolder = nil
    }
    func previewSlice(input: PreparationInput, plane: String, index: Int, segment: Bool) {
        guard !busy, result == nil else { return }
        guard !segment || !usesFocus else { error = "FOCUS-3D needs a complete 3D stack; the single-slice segmentation test is for Cellpose."; return }
        clearSlice()
        do {
            let destination = FileManager.default.temporaryDirectory.appendingPathComponent("atlas-slice-\(UUID().uuidString)")
            sliceFolder = destination
            let entry = try JSONSerialization.jsonObject(with: JSONEncoder().encode(input))
            let settings = try self.settings.json()
            progress = 0; stage = segment ? "Starting slice test…" : "Loading raw slice…"
            execute(segment ? "test-slice" : "slice", request: ["input": entry, "plane": plane, "index": index, "settings": settings], destination: destination) { [weak self] in
                guard let self else { return }
                do {
                    let value = try JSONDecoder().decode(PreparationSliceResult.self, from: Data(contentsOf: destination.appendingPathComponent("slice.json")))
                    guard let raw = NSImage(contentsOf: destination.appendingPathComponent("raw.png")) else { throw DataError.invalid("Raw slice preview is missing") }
                    self.sliceRaw = raw
                    self.sliceLabels = NSImage(contentsOf: destination.appendingPathComponent("labels.png"))
                    if segment && self.sliceLabels == nil { throw DataError.invalid("Cell label preview is missing") }
                    self.sliceResult = value; self.progress = 1
                    self.stage = segment ? "Slice test ready. Adjust the settings and test again, or prepare the full dataset." : self.usesFocus ? "Raw slice ready. Prepare the full stack to run FOCUS-3D." : "Raw slice ready. Test the current Cellpose settings when ready."
                } catch { self.error = error.localizedDescription }
            }
        } catch { self.error = error.localizedDescription }
    }
    func inspectOMEZarr() {
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false
        panel.treatsFilePackagesAsDirectories = true
        panel.message = "Choose an extracted .ome.zarr folder, or a parent containing raw and deconvolved images."
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let destination = FileManager.default.temporaryDirectory.appendingPathComponent("atlas-zarr-inspect-\(UUID().uuidString).json")
        stage = "Inspecting OME-Zarr images, channels and resolution levels…"
        execute("inspect-zarr", request: ["paths": [url.path]], destination: destination) { [weak self] in
            guard let self else { return }
            do {
                self.zarrCatalog = try JSONDecoder().decode(OMEZarrCatalog.self, from: Data(contentsOf: destination))
                try? FileManager.default.removeItem(at: destination)
            } catch { self.error = error.localizedDescription }
        }
    }
    func addOMEZarr(_ image: OMEZarrImage, channel: Int, level: Int, first: Int, last: Int, step: Int, label: String?) {
        guard !busy, image.channels.indices.contains(channel), let grid = image.levels.first(where: { $0.index == level }),
              first >= 0, first <= last, last < image.timepoints, step > 0 else { return }
        for time in stride(from: first, through: last, by: step) {
            let selection = OMEZarrSelection(time: time, channel: channel, level: level, label: label, fingerprint: image.fingerprint,
                                            origin: grid.origin, timestamp: image.timestamps.indices.contains(time) ? image.timestamps[time] : nil)
            let input = PreparationInput(path:image.path, name:"\(image.name) · T\(time) · \(image.channels[channel]) · level \(level)",
                shape:grid.shapeZYX, axes:"ZYX", dtype:grid.dtype, spacing:grid.spacing ?? PreparationCalibration.assumedCubicSpacing(grid.shapeZYX),
                calibrationSource:grid.spacing == nil ? PreparationCalibration.estimatedSource : "OME-NGFF level-specific spatial transforms (µm)", sourceStat:image.sourceStat, omeZarr:selection)
            if !inputs.contains(where: { $0.id == input.id }) { inputs.append(input) }
        }
        zarrCatalog = nil; confirmed = false
        stage = "OME-Zarr selections added. Confirm voxel spacing, then test a slice or prepare the dataset."
    }
    func addTIFF(_ image: TIFFImage, channel: Int, first: Int, last: Int, step: Int) {
        guard !busy, image.channels.indices.contains(channel), first >= 0, first <= last, last < image.timepoints, step > 0 else { return }
        for time in stride(from: first, through: last, by: step) {
            let selection = TIFFSelection(series: image.series, time: time, channel: channel, axes: image.axes, sourceAxes: image.sourceAxes, sourceShape: image.sourceShape)
            let input = PreparationInput(path: image.path, name: "\(image.name) · series \(image.series) · T\(time) · \(image.channels[channel])",
                shape: image.shapeZYX, axes: "ZYX", dtype: image.dtype, spacing: image.spacing,
                calibrationSource: image.calibrationSource, sourceStat: image.sourceStat, tiffSelection: selection)
            if !inputs.contains(where: { $0.id == input.id }) { inputs.append(input) }
        }
        confirmed = false; stage = "TIFF selections added. Confirm voxel spacing, then test a slice or prepare the dataset."
    }
    func addStacks() {
        let panel = NSOpenPanel(); panel.allowedContentTypes = atlasStackTypes; panel.allowsMultipleSelection = true
        panel.message = "Choose TIFF stacks or hyperstacks, or NumPy (.npy) Z/Y/X arrays. Multi-axis TIFFs open a series, time and channel selector."
        guard panel.runModal() == .OK else { return }
        let paths = panel.urls.map(\.path)
        guard !paths.isEmpty else { return }
        let destination = FileManager.default.temporaryDirectory.appendingPathComponent("atlas-inspect-\(UUID().uuidString).json")
        execute("inspect", request: ["paths": paths], destination: destination) { [weak self] in
            guard let self else { return }
            do {
                let inspected = try JSONDecoder().decode(InspectionResult.self, from: Data(contentsOf: destination))
                for input in inspected.inputs.map(\.input) where !self.inputs.contains(where: { $0.id == input.id }) { self.inputs.append(input) }
                if let images = inspected.tiffImages, !images.isEmpty {
                    self.tiffCatalog = TIFFCatalog(images: images, ignored: inspected.ignored ?? [])
                }
                self.confirmed = false
                self.stage = "Confirm voxel spacing for every stack."
                try? FileManager.default.removeItem(at: destination)
            } catch { self.error = error.localizedDescription }
        }
    }
    func chooseOutput() {
        let panel = NSSavePanel(); panel.nameFieldStringValue = title.replacingOccurrences(of: "/", with: "-") + " analysis"
        panel.canCreateDirectories = true; panel.message = "Choose a new results folder. Raw images will remain in their original locations."
        if let first = inputs.first { panel.directoryURL = URL(fileURLWithPath: first.path).deletingLastPathComponent() }
        if panel.runModal() == .OK { output = panel.url; result = nil }
    }
    func importLabels(_ index: Int) {
        let panel = NSOpenPanel(); panel.allowedContentTypes = atlasStackTypes
        panel.message = "Optional: existing integer TIFF or NumPy cell labels on the same Z/Y/X grid. Background must be zero."
        if panel.runModal() == .OK, let url = panel.url {
            do {
                let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
                inputs[index].labels = url.path
                inputs[index].labelStat = [(attributes[.size] as? NSNumber)?.int64Value ?? 0,
                    Int64(((attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)*1_000_000_000)]
                confirmed = false
            } catch { self.error = error.localizedDescription }
        }
    }
    func resume() {
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false
        panel.message = "Choose the results folder from an interrupted preparation."
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let manifest = try JSONDecoder().decode(PreparationManifest.self, from: Data(contentsOf: url.appendingPathComponent("preparation.json")))
            let request = manifest.request
            title = request.title; inputs = request.inputs; settings = request.settings; linkingMode = request.linkingMode ?? "standard"
            segmentationMethod = request.segmentationMethod ?? "cellpose"; focusSettings = request.focusSettings ?? FocusSettings()
            excludeDetached = request.excludeDetached; harmonics = request.harmonics; workers.restore(request.execution)
            output = url; confirmed = false; result = nil
            stage = "Review the saved settings, then prepare to resume completed steps."
        } catch { self.error = "This is not an Atlas preparation folder.\n\(error.localizedDescription)" }
    }
    func start() {
        if settings.native3D || usesFocus { workers.splitStacks = false; linkingMode = "standard" }
        guard usesFocus || workers.useLocal || !workers.enabled.isEmpty else { error = "Select this Mac or a paired worker in Nearby Macs."; return }
        guard valid, let output else { return }
        do {
            let request = PreparationRequest(title: title, inputs: inputs, settings: settings, excludeDetached: excludeDetached, harmonics: harmonics, execution: usesFocus ? nil : workers.execution, linkingMode: linkingMode == "standard" ? nil : linkingMode,
                segmentationMethod: usesFocus ? "focus3d" : nil, focusSettings: usesFocus ? focusSettings : nil)
            let json = try JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as! [String: Any]
            preparing = true; workerProgress = [:]; progress = 0; stage = "Checking inputs and output folder…"
            execute("build", request: json, destination: output) { [weak self] in
                guard let self else { return }
                self.result = output.appendingPathComponent("atlas_dataset"); self.progress = 1
                self.stage = "Dataset ready. Review cell exclusions and QC in Atlas."
            }
        } catch { self.error = error.localizedDescription; preparing = false }
    }
    func cancel() {
        guard let process, process.isRunning else { return }
        cancelling = true; stage = "Stopping after cancellation…"
        let pid = process.processIdentifier
        if kill(-pid, SIGTERM) != 0 { process.terminate() }
        DispatchQueue.global().asyncAfter(deadline: .now()+3) {
            if process.isRunning { kill(-pid, SIGKILL); kill(pid, SIGKILL) }
        }
    }
    private func consume(_ text: String) {
        log = String((log+text).suffix(24000)); pending += text
        while let range = pending.rangeOfCharacter(from: .newlines) {
            let line = String(pending[..<range.lowerBound]); pending.removeSubrange(..<range.upperBound)
            if line.hasPrefix("@worker-progress "), let data = line.dropFirst(17).data(using: .utf8),
               let event = try? JSONSerialization.jsonObject(with: data) as? [String: Any], let index = event["index"] as? Int {
                let worker = event["worker"] as? String ?? "Worker"
                let unit = event["unit"] as? String ?? "stack"
                // Show the latest task per Mac instead of growing hundreds of batch rows.
                let previous = workerProgress.keys.filter { $0 != index && workerProgress[$0]?.hasPrefix(worker + " · ") == true }
                for key in previous { workerProgress.removeValue(forKey: key) }
                workerProgress[index] = "\(worker) · \(unit) \(index+1) · \(event["stage"] as? String ?? "")"
            }
            if line.hasPrefix("@dataset-progress "), let data = line.dropFirst(18).data(using: .utf8),
               let event = try? JSONDecoder().decode(PreparationProgress.self, from: data), event.fraction.isFinite {
                progress = max(progress, min(1, max(0, event.fraction))); stage = event.stage
            }
        }
        pending = String(pending.suffix(24000))
    }
    private func execute(_ operation: String, request: [String: Any], destination: URL, completion: @escaping () -> Void) {
        guard !busy else { return }
        do {
            let python = root.appendingPathComponent(".venv/bin/python")
            guard FileManager.default.isExecutableFile(atPath: python.path) else { throw DataError.invalid("The processing environment is missing. Restore the project's Python environment before preparing datasets.") }
            let req = FileManager.default.temporaryDirectory.appendingPathComponent("atlas-preparation-\(UUID().uuidString).json")
            try JSONSerialization.data(withJSONObject: request, options: [.prettyPrinted, .sortedKeys]).write(to: req)
            let process = Process(); process.executableURL = python; process.currentDirectoryURL = root
            process.arguments = ["-u", root.appendingPathComponent("scripts/prepare_atlas_dataset.py").path, operation, "--request", req.path, "--output", destination.path]
            var env = ProcessInfo.processInfo.environment; env["PYTHONUNBUFFERED"] = "1"; process.environment = env
            let pipe = Pipe(); process.standardOutput = pipe; process.standardError = pipe
            let generation = UUID(); token = generation; self.operation = operation; previewing = ["slice", "test-slice"].contains(operation); busy = true; cancelling = false; error = nil; log = ""; pending = ""
            pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
                let bytes = handle.availableData
                if !bytes.isEmpty { Task { @MainActor in guard let self, self.token == generation else { return }; self.consume(String(decoding: bytes, as: UTF8.self)) } }
            }
            process.terminationHandler = { [weak self] finished in
                pipe.fileHandleForReading.readabilityHandler = nil
                Task { @MainActor in
                    guard let self, self.token == generation else { return }
                    self.busy = false; self.preparing = false; self.previewing = false; self.process = nil
                    try? FileManager.default.removeItem(at: req)
                    if self.cancelling {
                        self.cancelling = false; self.stage = self.operation == "build" ? "Cancelled. Completed steps are saved; resume with the same settings." : "Slice preview or inspection cancelled."
                    } else if finished.terminationStatus == 0 { completion() }
                    else { self.stage = self.operation == "build" ? "Preparation stopped. Completed steps are saved." : "Preview or inspection stopped · see processing log"; self.error = "\(operation.hasPrefix("inspect") ? "Stack inspection" : operation == "build" ? "Dataset preparation" : "Slice preview") failed.\n\(self.log.suffix(3500))" }
                }
            }
            self.process = process; try process.run()
        } catch { busy = false; preparing = false; previewing = false; process = nil; self.error = error.localizedDescription }
    }
}

struct DatasetPreparationView: View {
    @ObservedObject var preparation: DatasetPreparationStore
    let open: (URL) -> Void
    let close: () -> Void
    @State private var showLog = false
    @State private var showWorkers = false
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Prepare a new dataset").font(.title2.bold())
                    Text("Raw stacks → cell segmentation → surfaces and measurements → shape comparisons")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Button("Close", action: close).disabled(preparation.busy)
            }
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    GroupBox("1 · Raw images and calibration") {
                        VStack(alignment: .leading, spacing: 10) {
                            HStack {
                                Button("Add TIFF / NumPy stacks…") { preparation.addStacks() }
                                Button("Import OME-Zarr…") { preparation.inspectOMEZarr() }
                                Button("Resume saved preparation…") { preparation.resume() }
                                Spacer()
                                Text("\(preparation.inputs.count) stacks").foregroundStyle(.secondary)
                            }
                            ForEach(preparation.inputs.indices, id: \.self) { i in
                                VStack(alignment: .leading, spacing: 6) {
                                    HStack {
                                        Text(preparation.inputs[i].name).font(.headline).lineLimit(1).help(preparation.inputs[i].path)
                                        Spacer()
                                        Button("Remove") { preparation.inputs.remove(at: i); preparation.confirmed = false }
                                    }
                                    Text("Z/Y/X: \(preparation.inputs[i].shape.map(String.init).joined(separator: " × ")) · \(preparation.inputs[i].dtype) · interpreted as one 3D stack")
                                        .font(.caption).foregroundStyle(.secondary)
                                    HStack {
                                        ForEach(0..<3) { axis in
                                            Text(["Z", "Y", "X"][axis]+" µm")
                                            TextField("Spacing", value: $preparation.inputs[i].spacing[axis], format: .number)
                                                .frame(width: 85).textFieldStyle(.roundedBorder)
                                        }
                                    }
                                    Text(preparation.inputs[i].calibrationSource).font(.caption).foregroundStyle(.secondary)
                                    if let zarr = preparation.inputs[i].omeZarr {
                                        Text("Time \(zarr.time) · channel \(zarr.channel) · level \(zarr.level) · \(zarr.label.map { "existing " + $0 + " instances" } ?? "automatic segmentation")")
                                            .font(.caption).foregroundStyle(.secondary)
                                    } else { HStack {
                                        Button("Use existing labels…") { preparation.importLabels(i) }
                                        if let labels = preparation.inputs[i].labels {
                                            Text(URL(fileURLWithPath: labels).lastPathComponent).font(.caption).lineLimit(1)
                                            Button("Segment instead") { preparation.inputs[i].labels = nil; preparation.inputs[i].labelStat = nil }
                                        } else { Text("Optional; otherwise the selected method segments this stack.").font(.caption).foregroundStyle(.secondary) }
                                    }
                                    }
                                    Divider()
                                }
                            }
                            Text("NumPy arrays use Z/Y/X order. Missing calibration assumes a cubic stack: X = Y = 1 and Z = longest XY dimension ÷ slice count. Only Z is estimated; rectangular images keep their XY aspect ratio. These are nominal µm, not measured physical sizes; edit the values if known. Known microscopy calibration is preserved. Each stack becomes one organoid entry.")
                                .font(.caption).foregroundStyle(.secondary)
                        }.padding(6)
                    }
                    GroupBox("2 · Segmentation and comparisons") {
                        VStack(alignment: .leading, spacing: 10) {
                            Picker("Initial segmentation", selection: $preparation.segmentationMethod) {
                                Text("Cellpose").tag("cellpose"); Text("FOCUS-3D").tag("focus3d")
                            }.pickerStyle(.segmented).frame(maxWidth: 440)
                            if preparation.usesFocus {
                                FocusPreparationControls(settings: $preparation.focusSettings)
                                if !preparation.focusInstalled { Text("The FOCUS-3D environment is missing. Install it with scripts/setup_focus3d.sh.").font(.caption).foregroundStyle(.orange) }
                                if !preparation.focusReady, !preparation.inputs.allSatisfy(\.hasProvidedLabels) {
                                    Text("Select model weights and valid settings. Check that X and Y calibration match for each stack.").font(.caption).foregroundStyle(.orange)
                                }
                                if let plan = preparation.largestFocusMemoryPlan {
                                    Text("Largest FOCUS inference grid: \(plan.shape.map(String.init).joined(separator: " × ")) · conservative memory estimate \(plan.estimatedPeakGiB, specifier: "%.1f") GiB / budget \(plan.budgetGiB, specifier: "%.1f") GiB.").font(.caption).foregroundStyle(.secondary)
                                    if preparation.memoryBlocked { Text("Use a smaller crop or a Mac with more memory. Small model patches still require whole-volume buffers; do not change the estimated cell radius just to bypass the budget.").font(.caption).foregroundStyle(.orange) }
                                }
                            } else {
                            CellposeModelControls(settings: $preparation.settings)
                            HStack {
                                Text("Scale")
                                TextField("Scale", value: $preparation.settings.scale, format: .number).frame(width: 65)
                                Text("Diameter")
                                TextField("Diameter", value: $preparation.settings.diameter, format: .number).frame(width: 65)
                                Text("processed px").font(.caption).foregroundStyle(.secondary)
                            }
                            Text("Scale 0.05–1. Smaller values reduce memory use and detail. Choose u-Segment3D linking or Cellpose's native 3D mask formation. First use may download model weights.")
                                .font(.caption).foregroundStyle(.secondary)
                            CellposeModeControls(settings: $preparation.settings)
                            Picker("3D linking", selection: $preparation.linkingMode) {
                                Text("Standard").tag("standard")
                                Text("Memory-saving fine grid").tag("streamed")
                            }.frame(maxWidth: 480).disabled(preparation.settings.native3D)
                            if preparation.linkingMode == "streamed" {
                                Text("Keeps the selected resolution. Saves Cellpose predictions and links cells using temporary disk files. Progress is saved every 8 image planes and flow chunks; resume with the same settings. Temporary linking arrays are removed when labels are verified.")
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                            if let plan = preparation.largestMemoryPlan {
                                VStack(alignment: .leading, spacing: 5) {
                                    Text("Largest processing grid: \(plan.shape.map(String.init).joined(separator: " × ")) · \(plan.imageGiB, specifier: "%.2f") GiB per float image")
                                    Text("Conservative memory estimate: \(plan.estimatedPeakGiB, specifier: "%.1f") GiB · planning budget: \(plan.budgetGiB, specifier: "%.1f") GiB. Actual memory varies with foreground coverage and other apps.")
                                    if preparation.memoryBlocked {
                                        Text("This scale exceeds the memory budget on this Mac. Image preparation and exports still need RAM. Sharing inference batches does not pool RAM.").foregroundStyle(.orange)
                                        if let scale = plan.suggestedScale, scale < preparation.settings.scale {
                                            Button(String(format: "Use lower scale %.2f", scale)) { preparation.useSuggestedScale() }
                                            Text("Reduces spatial detail; retest the diameter, which is measured in processed pixels. Existing checkpoints are kept in their original folder.")
                                        } else { Text("Use a smaller input crop or a Mac with more memory.") }
                                    }
                                }.font(.caption).foregroundStyle(.secondary)
                            }
                            HStack {
                                Toggle("Correct background", isOn: $preparation.settings.background)
                                Toggle("Use GPU when available", isOn: $preparation.settings.gpu)
                            }
                            }
                            DatasetSliceTestView(preparation: preparation)
                            Toggle("Exclude cells outside the main organoid", isOn: $preparation.excludeDetached)
                            Text("Uses the largest connected component on the segmentation grid. Excluded cells remain available for review and manual inclusion.").font(.caption).foregroundStyle(.secondary)
                            Toggle("Compute multi-shell harmonic shape descriptors", isOn: $preparation.harmonics)
                            Text("Always includes calibrated cell surfaces, QC flags, traditional features, size-only and shape-only comparisons. Harmonics add the shape descriptor used in our five-organoid benchmark; no model training is required.")
                                .font(.caption).foregroundStyle(.secondary)
                        }.padding(6)
                    }
                    if preparation.usesFocus {
                        Text("FOCUS-3D processes complete stacks on this Mac using its own patch stitching. Cellpose/u-Segment3D plane batches and Nearby Macs are not used.").font(.caption).foregroundStyle(.secondary)
                    } else {
                        AtlasWorkerSelection(workers: preparation.workers, showWorkers: $showWorkers, allowPlaneBatches: !preparation.settings.native3D)
                    }
                    GroupBox("3 · Save and prepare") {
                        VStack(alignment: .leading, spacing: 10) {
                            TextField("Dataset name", text: $preparation.title).textFieldStyle(.roundedBorder)
                            HStack {
                                Button("Choose results folder…") { preparation.chooseOutput() }
                                Text(preparation.output?.path ?? "Choose a new, empty folder").font(.caption).lineLimit(2).textSelection(.enabled)
                            }
                            Text("Stacks and segmentation stages run sequentially, with checkpoints and a memory safeguard. All results go in this folder. Raw images stay in their original locations. NumPy volumes are cached as calibrated TIFFs in the results folder.")
                                .font(.caption).foregroundStyle(.secondary)
                            if preparation.linkingMode == "streamed", !preparation.inputs.allSatisfy({ $0.hasProvidedLabels }) {
                                Text("Disk estimate for a fresh run: \(preparation.diskEstimateGiB, specifier: "%.1f") GiB across all stacks, including cached predictions, labels and temporary files.")
                                    .font(.caption).foregroundStyle(.secondary)
                                if let free = preparation.availableDiskGiB {
                                    Text("Available on the results drive: \(free, specifier: "%.1f") GiB. Disk space is checked before processing and linking; cached files count toward resumed runs.")
                                        .font(.caption).foregroundStyle(free < preparation.diskEstimateGiB ? .orange : .secondary)
                                }
                            }
                            Toggle("I have checked the Z/Y/X interpretation and voxel spacing for every stack", isOn: $preparation.confirmed)
                        }.padding(6)
                    }
                }.disabled(preparation.busy || preparation.result != nil)
            }
            Divider()
            ForEach(preparation.workerProgress.keys.sorted(), id: \.self) { index in
                Text(preparation.workerProgress[index] ?? "").font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
            Text(preparation.stage).font(.callout).textSelection(.enabled)
            if preparation.busy {
                if preparation.preparing || preparation.previewing { ProgressView(value: preparation.progress) }
                else { ProgressView("Inspecting TIFF metadata…").controlSize(.small) }
            }
            if let error = preparation.error { Text(error).font(.caption).foregroundStyle(.red).lineLimit(5).textSelection(.enabled) }
            DisclosureGroup("Processing log", isExpanded: $showLog) {
                ScrollView { Text(preparation.log).font(.system(size: 10, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }.frame(height: 100)
            }
            HStack {
                if preparation.busy { Button(preparation.cancelling ? "Stopping…" : "Cancel processing") { preparation.cancel() }.disabled(preparation.cancelling) }
                Spacer()
                if let result = preparation.result {
                    Button("Prepare another dataset") { preparation.reset() }
                    Button("Open dataset in Atlas") { open(result) }.buttonStyle(.borderedProminent)
                } else {
                    Button("Prepare dataset") { preparation.start() }.buttonStyle(.borderedProminent).disabled(!preparation.valid)
                }
            }
        }.padding(22).frame(width: 900, height: 800).interactiveDismissDisabled(preparation.busy)
            .sheet(item: $preparation.zarrCatalog) { catalog in
                OMEZarrImportView(catalog: catalog, add: { image, channel, level, first, last, step, label in
                    preparation.addOMEZarr(image, channel: channel, level: level, first: first, last: last, step: step, label: label)
                }, close: { preparation.zarrCatalog = nil })
            }
            .sheet(item: $preparation.tiffCatalog) { catalog in
                TIFFImportView(catalog: catalog, add: { image, channel, first, last, step in
                    preparation.addTIFF(image, channel: channel, first: first, last: last, step: step)
                }, close: { preparation.tiffCatalog = nil })
            }
            .sheet(isPresented: $showWorkers) { AtlasWorkersView(workers: preparation.workers, close: { showWorkers = false }) }
            .onChange(of: preparation.settings.segmentationMode) { _ in
                if preparation.settings.native3D { preparation.workers.splitStacks = false; preparation.linkingMode = "standard" }
            }
            .onChange(of: preparation.segmentationMethod) { _ in
                preparation.clearSlice()
                if preparation.usesFocus { preparation.linkingMode = "standard"; preparation.workers.splitStacks = false }
            }
    }
}
