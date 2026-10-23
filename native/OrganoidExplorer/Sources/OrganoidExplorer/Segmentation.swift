import SwiftUI
import AppKit
import Darwin

struct SegProgressEvent: Decodable {
    let stage: String
    let fraction, span: Double
}

struct SegSource: Codable {
    let path, fingerprint, folder, dtype: String
    let shape, nativeShape: [Int]
    let spacing: [Double]
    let low, high: Double
}
struct SegCell: Codable, Identifiable {
    let label: Int
    let volume: Double
    let centroid: [Double]
    let border: Bool
    var inMainOrganoid: Bool? = nil
    var includedInAnalysis: Bool { inMainOrganoid ?? true }
    var id: Int { label }
}
struct SegResult: Codable, Identifiable {
    let id, method, source, folder, provenance, created, labelHash: String
    let shape: [Int]
    let spacing: [Double]
    let cells: [SegCell]
    let seconds: Double
    let purpose: String?
    let prediction: SegPrediction?
    let cellposeModel: String?
    let cellposeMode: String?
    let comparisonSide: String?
    let refinement: RefinementInfo?
    var analysisCells: [SegCell] { cells.filter(\.includedInAnalysis) }
    // Backend settings are heterogeneous; retain exact JSON separately when comparing.
    var cellposeModeName: String { cellposeMode == "cellpose3d" ? "Native 3D" : "u-Segment3D" }
    var title: String { "\(method == "refined" ? ((refinement?.algorithmVersion ?? 1) >= 2 ? "Shape-preserving · " : "Legacy refinement · ") : method == "cellpose" ? (cellposeModel ?? "cpsam_v2") + " · " + cellposeModeName + " · " : "")\(created) · \(analysisCells.count) main-organoid cells" }
}
struct SegPair: Codable { let a, b: Int; let iou: Double }
struct SegAgreement: Codable {
    let dice, meanIoU: Double
    let matched, countA, countB: Int
    let pairs: [SegPair]
    let note: String
    var excludedA: [Int]? = nil
    var excludedB: [Int]? = nil
}

@MainActor final class SegmentationStore: ObservableObject {
    @Published var source: SegSource?
    @Published var raw: [UInt8] = []
    @Published var results: [SegResult] = []
    @Published var resultA: SegResult?
    @Published var resultB: SegResult?
    @Published var labelsA: [UInt32] = []
    @Published var labelsB: [UInt32] = []
    @Published var agreement: SegAgreement?
    @Published var progress = 0.0
    @Published var progressStage = "Starting…"
    private var progressBase = 0.0
    private var progressSpan = 0.0
    private var pendingOutput = ""
    @Published var busy = false
    @Published var status = "Select an experiment stack or open a 3D TIFF."
    @Published var log = ""
    @Published var error: String?
    @Published var axis = 0
    @Published var slice = 0.5
    @Published var brightness = 1.0
    @Published var overlay = true
    @Published var boundaries = false
    @Published var selectedA = 0
    @Published var selectedB = 0
    @Published var cellposeA = CellposeSettings()
    @Published var cellposeB = CellposeSettings(model:"cpdino-vitb")
    @Published var trialSettings = PansegSettings()
    @Published var comparisonSettings = PansegSettings()
    @Published var comparisonMethod = "focus3d"
    @Published var focusSettings = FocusSettings()
    @Published var refinementSettings = RefinementSettings()
    private var beforeRefinementA: SegResult?
    private var refinementContextCaptured = false
    @Published var trialResult: SegResult?
    @Published var trialLabels: [UInt32] = []
    @Published var trialBoundaries: [UInt8] = []
    @Published var selectedTrial = 0
    @Published var presets: [PansegPreset] = []
    @Published var presetName = ""
    private var runningTrial = false
    private var reuseTrial = false
    @Published var spacingZ = 1.0
    @Published var spacingY = 0.1733
    @Published var spacingX = 0.1733
    private var process: Process?
    private var generation = UUID()
    private var baseline: String?
    private var runQueue: [String] = []
    let root: URL
    let jobs: URL
    init(projectRoot: URL? = nil) {
        let path = Bundle.main.object(forInfoDictionaryKey: "ExplorerProjectPath") as? String ?? FileManager.default.currentDirectoryPath
        root = projectRoot ?? URL(fileURLWithPath:path)
        jobs = root.appendingPathComponent("results/segmentation_explorer")
        focusSettings.checkpoint = UserDefaults.standard.string(forKey:"focus3dCheckpoint") ?? ""
    }
    var spacing: [Double] { [spacingZ, spacingY, spacingX] }
    var calibrationChanged: Bool { source.map { $0.spacing != spacing } ?? false }
    var canRun: Bool { source != nil && !busy && !calibrationChanged }
    var pansegInstalled: Bool { FileManager.default.isExecutableFile(atPath: root.appendingPathComponent(".cache/panseg-env/bin/python").path) && FileManager.default.fileExists(atPath: root.appendingPathComponent("vendor/panseg/panseg").path) }
    var focusInstalled: Bool { FileManager.default.isExecutableFile(atPath:root.appendingPathComponent(".cache/focus3d-env/bin/python").path) && FileManager.default.fileExists(atPath:root.appendingPathComponent("vendor/focus3d/src/focus3d/segmentation/FOCUS3D/inference_win.py").path) }
    var focusCheckpointAvailable: Bool { !focusSettings.checkpoint.isEmpty && FileManager.default.fileExists(atPath:focusSettings.checkpoint) }
    var focusReady: Bool { focusInstalled && focusCheckpointAvailable }
    var comparisonReady: Bool { comparisonMethod != "refined" && (comparisonMethod == "cellpose" || (comparisonMethod == "focus3d" ? focusReady : pansegInstalled)) }
    var comparisonName: String { comparisonMethod == "refined" ? "Membrane-refined" : comparisonMethod == "cellpose" ? "Cellpose B" : comparisonMethod == "focus3d" ? "FOCUS-3D" : "PanSeg" }
    var leftName: String { comparisonMethod == "refined" ? "Original · \(resultA.map { methodName($0.method) } ?? "segmentation")" : "Cellpose A · \(resultA.map { $0.cellposeModel ?? "cpsam_v2" } ?? cellposeA.model) · \(resultA?.cellposeModeName ?? cellposeA.modeName)" }
    var rightName: String { comparisonMethod == "cellpose" ? "Cellpose B · \(resultB.map { $0.cellposeModel ?? "cpsam_v2" } ?? cellposeB.model) · \(resultB?.cellposeModeName ?? cellposeB.modeName)" : comparisonName }
    func methodName(_ method:String) -> String { method == "imported" ? "Imported labels" : method == "refined" ? "Membrane-guided refinement" : method == "cellpose" ? "Cellpose + u-Segment3D" : method == "focus3d" ? "FOCUS-3D" : "PanSeg" }
    func openFocusFile(config: Bool = false) {
        let panel = NSOpenPanel(); panel.message = config ? "Choose a FOCUS-3D YAML configuration" : "Choose a downloaded FOCUS-3D checkpoint (.pth)"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        if config { focusSettings.config = url.path }
        else { focusSettings.checkpoint = url.path; UserDefaults.standard.set(url.path,forKey:"focus3dCheckpoint") }
    }
    func setComparisonMethod(_ method:String) {
        guard !busy, ["cellpose","panseg","focus3d","refined"].contains(method) else { return }
        let wasRefined = comparisonMethod == "refined"
        if method == "refined" && !wasRefined { beforeRefinementA = resultA; refinementContextCaptured = true }
        comparisonMethod = method; resultB = nil; labelsB = []; selectedA = 0; selectedB = 0; agreement = nil
        if wasRefined && method != "refined" { restoreBeforeRefinement() }
        restoreComparisonRun(); compare()
    }
    private func restoreBeforeRefinement() {
        let previous = beforeRefinementA; beforeRefinementA = nil; refinementContextCaptured = false
        resultA = nil; labelsA = []; selectedA = 0
        if let previous { choose(previous,side:1) }
    }
    private func restoreComparisonRun() {
        guard let source else { return }
        let key = comparisonMethod == "cellpose" ? "cellposeComparisonB" : "\(comparisonMethod)Comparison"
        let saved = UserDefaults.standard.string(forKey:"\(key).\(source.fingerprint)")
        let fallback = results.first(where:{$0.method == comparisonMethod && $0.purpose != "trial" && (comparisonMethod != "cellpose" || ($0.id != resultA?.id && ($0.cellposeModel != resultA?.cellposeModel || $0.cellposeModeName != resultA?.cellposeModeName)))})
        if let result = results.first(where:{$0.id == saved && $0.method == comparisonMethod}) ?? fallback {
            if result.method == "refined" { showRefinement(result) }
            else { choose(result,side:2) }
        }
    }

    func open() {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.tiff]; panel.canChooseDirectories = false
        if panel.runModal() == .OK, let url = panel.url { prepare(url.path) }
    }
    func prepare(_ path: String, baseline: String? = nil, spacing: [Double]? = nil) {
        guard !busy else { return }
        if let s = spacing, s.count == 3 { spacingZ = s[0]; spacingY = s[1]; spacingX = s[2] }
        self.baseline = baseline
        beforeRefinementA = nil; refinementContextCaptured = false
        source = nil; raw = []; results = []; resultA = nil; resultB = nil; labelsA = []; labelsB = []; agreement = nil
        selectedA = 0; selectedB = 0; selectedTrial = 0; trialResult = nil; trialLabels = []; trialBoundaries = []; slice = 0.5
        let folder = jobs.appendingPathComponent("sources/\(UUID().uuidString)")
        execute("prepare", request: ["path": path, "spacing": self.spacing], output: folder.appendingPathComponent("source.json")) { [weak self] url in
            guard let self else { return }
            do {
                let source = try JSONDecoder().decode(SegSource.self, from: Data(contentsOf: url))
                let bytes = try Data(contentsOf: URL(fileURLWithPath: source.folder).appendingPathComponent("raw.u8"))
                guard bytes.count == source.shape.reduce(1, *) else { throw DataError.invalid("Raw preview size mismatch") }
                self.source = source; self.raw = Array(bytes); self.refresh()
                self.status = "Ready · \(source.nativeShape.map(String.init).joined(separator: " × ")) voxels · \(source.dtype)"
            } catch { self.error = error.localizedDescription }
        }
    }
    func reloadCalibration() { if let source { prepare(source.path, baseline: baseline) } }
    func loadRunHistory() {
        guard let source else { return }
        let folder = jobs.appendingPathComponent("runs")
        let files = (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
        results = files.compactMap { try? JSONDecoder().decode(SegResult.self, from: Data(contentsOf: $0.appendingPathComponent("result.json"))) }
            .filter { $0.source == source.fingerprint && $0.shape == source.nativeShape && $0.spacing == source.spacing }
            .sorted { $0.created > $1.created }
    }
    func refresh() {
        guard let source else { return }
        loadRunHistory()
        let savedA = UserDefaults.standard.string(forKey:"cellposeComparisonA.\(source.fingerprint)")
        if let a = results.first(where:{$0.method == "cellpose" && $0.id == savedA}) ?? results.first(where:{$0.method == "cellpose" && $0.comparisonSide != "B"}) { choose(a,side:1) }
        restoreComparisonRun()
        if let trial = results.first(where: { $0.method == "panseg" && $0.prediction != nil }) { chooseTrial(trial) }
        if resultA != nil && resultB != nil { compare() }
    }
    func adoptPreparedSource(_ source:SegSource, raw:[UInt8]) {
        guard !busy, raw.count == source.shape.reduce(1,*) else { return }
        self.source = source; self.raw = raw
        spacingZ = source.spacing[0]; spacingY = source.spacing[1]; spacingX = source.spacing[2]
        resultA = nil; resultB = nil; labelsA = []; labelsB = []; agreement = nil; selectedA = 0; selectedB = 0
        beforeRefinementA = nil; refinementContextCaptured = false; slice = 0.5; error = nil
        loadRunHistory(); status = "Ready for post processing"
    }
    func choose(_ result: SegResult, side: Int? = nil) {
        guard let source, ["cellpose","panseg","focus3d","refined","imported"].contains(result.method) else { return }
        guard result.source == source.fingerprint, result.shape == source.nativeShape, result.spacing == source.spacing else { error = "Run does not belong to this stack and calibration."; return }
        do {
            let bytes = try Data(contentsOf: URL(fileURLWithPath: result.folder).appendingPathComponent("labels.u32"))
            guard bytes.count == source.shape.reduce(1, *) * 4 else { throw DataError.invalid("Label preview size mismatch") }
            let labels = bytes.withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
            let targetSide = side ?? (result.method == "cellpose" && result.comparisonSide != "B" ? 1 : 2)
            if targetSide == 2 && result.method != "refined" && comparisonMethod == "refined" {
                comparisonMethod = result.method; restoreBeforeRefinement()
            }
            if targetSide == 1 && (result.method == "cellpose" || comparisonMethod == "refined") {
                resultA = result; labelsA = labels; selectedA = 0; selectedB = 0
                if comparisonMethod != "refined" {
                    if let saved = cellposeSettings(for:result) { cellposeA = saved }
                    UserDefaults.standard.set(result.id,forKey:"cellposeComparisonA.\(source.fingerprint)")
                }
            }
            else {
                comparisonMethod = result.method; resultB = result; labelsB = labels; selectedA = 0; selectedB = 0
                if result.method == "panseg" { comparisonSettings = (try? settings(for:result)) ?? PansegSettings() }
                else if result.method == "cellpose" { if let saved = cellposeSettings(for:result) { cellposeB = saved } }
                else if let data = try? Data(contentsOf:URL(fileURLWithPath:result.folder).appendingPathComponent("result.json")),
                        let json = try? JSONSerialization.jsonObject(with:data) as? [String:Any],
                        let settings = json["settings"] as? [String:Any], settings["checkpoint"] != nil,
                        let saved = try? FocusSettings.fromJSON(settings) { focusSettings = saved }
                let key = result.method == "cellpose" ? "cellposeComparisonB" : "\(result.method)Comparison"
                UserDefaults.standard.set(result.id,forKey:"\(key).\(source.fingerprint)")
            }
            agreement = nil
        } catch { self.error = error.localizedDescription }
    }
    func cellposeSettings(for result:SegResult) -> CellposeSettings? {
        guard let data = try? Data(contentsOf:URL(fileURLWithPath:result.folder).appendingPathComponent("result.json")),
              let json = try? JSONSerialization.jsonObject(with:data) as? [String:Any], let settings = json["settings"] as? [String:Any] else { return nil }
        if let original = settings["originalSettings"] as? [String:Any] { return try? CellposeSettings.fromJSON(original["config"] as? [String:Any] ?? original) }
        if settings["importedFrom"] != nil { return nil }
        return try? CellposeSettings.fromJSON(settings)
    }
    func selectRun(_ id: String, side: Int? = nil) {
        guard !busy, let result = results.first(where: { $0.id == id }) else { return }
        if result.method == "refined" { showRefinement(result); compare(); return }
        choose(result,side:side); compare()
    }
    func canRefine(side:Int) -> Bool { canRun && (side == 1 ? resultA : resultB).map { $0.method != "refined" } == true }
    func showRefinement(_ result:SegResult) {
        guard let info = result.refinement else { return }
        do {
            let parent = try JSONDecoder().decode(SegResult.self,from:Data(contentsOf:URL(fileURLWithPath:result.folder).appendingPathComponent("parent.json")))
            guard parent.id == info.parentID else { throw DataError.invalid("Refinement parent mismatch") }
            if comparisonMethod != "refined" || !refinementContextCaptured { beforeRefinementA = resultA; refinementContextCaptured = true }
            comparisonMethod = "refined"
            choose(parent,side:1)
            guard resultA?.id == parent.id else { return }
            choose(result,side:2)
        } catch { self.error = error.localizedDescription }
    }
    func refineDisplayed(side:Int) {
        guard canRefine(side:side), let source, let parent = side == 1 ? resultA : resultB else { return }
        do {
            let request: [String:Any] = ["source":try JSONSerialization.jsonObject(with:JSONEncoder().encode(source)),
                "parent":try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:parent.folder).appendingPathComponent("result.json"))),
                "settings":try refinementSettings.json()]
            let folder = jobs.appendingPathComponent("runs/\(UUID().uuidString)")
            execute("refine",request:request,output:folder.appendingPathComponent("result.json")) { [weak self] url in
                guard let self else { return }
                do {
                    let result = try JSONDecoder().decode(SegResult.self,from:Data(contentsOf:url))
                    self.results.insert(result,at:0); self.showRefinement(result); self.compare()
                } catch { self.error = error.localizedDescription }
            }
        } catch { self.error = error.localizedDescription }
    }
    func settings(for result: SegResult) throws -> PansegSettings {
        let data = try Data(contentsOf:URL(fileURLWithPath:result.folder).appendingPathComponent("result.json"))
        guard let json = try JSONSerialization.jsonObject(with:data) as? [String:Any], let settings = json["settings"] as? [String:Any] else { throw DataError.invalid("Missing saved PanSeg settings") }
        return try PansegSettings.fromJSON(settings)
    }
    func chooseTrial(_ result: SegResult) {
        guard let source, result.method == "panseg", result.source == source.fingerprint, result.shape == source.nativeShape, result.spacing == source.spacing else { error = "Trial does not belong to this stack and calibration."; return }
        do {
            let folder = URL(fileURLWithPath:result.folder)
            let bytes = try Data(contentsOf:folder.appendingPathComponent("labels.u32"))
            let boundary = (try? Data(contentsOf:folder.appendingPathComponent("boundaries.u8"))) ?? Data()
            let count = source.shape.reduce(1,*)
            guard bytes.count == count*4, boundary.isEmpty || boundary.count == count else { throw DataError.invalid("Trial preview size mismatch") }
            trialLabels = bytes.withUnsafeBytes { Array($0.bindMemory(to:UInt32.self)) }
            trialBoundaries = Array(boundary); trialResult = result; selectedTrial = 0
        } catch { self.error = error.localizedDescription }
    }
    func loadTrialSettings() {
        guard !busy, let result = trialResult else { return }
        do { trialSettings = try settings(for:result) } catch { self.error = error.localizedDescription }
    }
    var canReusePrediction: Bool {
        guard let result = trialResult, result.prediction != nil, let saved = try? settings(for:result) else { return false }
        return saved.inferenceIdentity == trialSettings.inferenceIdentity && FileManager.default.fileExists(atPath:URL(fileURLWithPath:result.folder).appendingPathComponent("boundaries.tif").path)
    }
    var trialSettingsChanged: Bool { trialResult.flatMap { try? settings(for:$0) }.map { $0 != trialSettings } ?? false }
    func runTrial(reuse: Bool = false) {
        guard canRun, !reuse || canReusePrediction else { return }
        runningTrial = true; reuseTrial = reuse; runQueue = ["panseg"]; runNext()
    }
    func transferTrial() -> Bool {
        guard !busy, let source, let result = trialResult, result.prediction != nil, result.source == source.fingerprint, result.shape == source.nativeShape, result.spacing == source.spacing else { return false }
        do {
            let saved = try settings(for:result)
            choose(result)
            guard resultB?.id == result.id else { return false }
            comparisonSettings = saved
            UserDefaults.standard.set(result.id,forKey:"pansegComparison.\(source.fingerprint)")
            status = "Transferred displayed PanSeg run and saved settings"
            compare(); return true
        } catch { self.error = error.localizedDescription; return false }
    }
    func refreshPresets() {
        let folder = jobs.appendingPathComponent("presets")
        presets = ((try? FileManager.default.contentsOfDirectory(at:folder,includingPropertiesForKeys:nil)) ?? []).compactMap { try? JSONDecoder().decode(PansegPreset.self,from:Data(contentsOf:$0)) }.sorted { $0.name < $1.name }
    }
    func savePreset() {
        guard !busy else { return }
        do {
            let preset = PansegPreset(id:UUID().uuidString,name:presetName.isEmpty ? trialSettings.summary : presetName,settings:trialSettings)
            let folder = jobs.appendingPathComponent("presets"); try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
            try JSONEncoder().encode(preset).write(to:folder.appendingPathComponent(preset.id+".json"),options:.atomic)
            refreshPresets(); status = "Saved settings preset: \(preset.name)"
        } catch { self.error = error.localizedDescription }
    }
    func exportPreset() {
        let panel = NSSavePanel(); panel.allowedContentTypes = [.json]; panel.nameFieldStringValue = "panseg-settings.json"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted,.sortedKeys]; try encoder.encode(trialSettings).write(to:url,options:.atomic) } catch { self.error = error.localizedDescription }
    }
    func importPreset() {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            guard let json = try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any] else { throw DataError.invalid("Expected a PanSeg settings JSON object") }
            trialSettings = try PansegSettings.fromJSON((json["settings"] as? [String:Any]) ?? json)
        } catch { self.error = error.localizedDescription }
    }
    func openModelFile(weights: Bool) {
        let panel = NSOpenPanel(); panel.message = weights ? "Choose U-Net weights" : "Choose a PanSeg U-Net YAML configuration"
        if panel.runModal() == .OK, let url = panel.url { if weights { trialSettings.weightsPath = url.path } else { trialSettings.configPath = url.path } }
    }
    func run(_ methods: [String]) { guard canRun else { return }; runningTrial = false; reuseTrial = false; runQueue = methods; runNext() }
    private func runNext() {
        guard let source, !runQueue.isEmpty else { return }
        let job = runQueue.removeFirst()
        let side = job == "cellposeB" ? 2 : 1
        let method = job == "cellposeB" ? "cellpose" : job
        if method == "panseg" && !pansegInstalled { error = "PanSeg is not installed. Run scripts/setup_panseg.sh from this project, then retry."; runQueue = []; return }
        if method == "focus3d" && !focusReady { error = "Install FOCUS-3D with zsh scripts/setup_focus3d.sh and select a downloaded .pth checkpoint."; runQueue = []; return }
        let settings: [String: Any]
        do {
            if method == "cellpose" { settings = try (side == 1 ? cellposeA : cellposeB).json() }
            else if method == "focus3d" { settings = try focusSettings.json() }
            else { settings = try (runningTrial ? trialSettings : comparisonSettings).json() }
        }
        catch { self.error = error.localizedDescription; runQueue = []; return }
        let sourceJSON = try! JSONSerialization.jsonObject(with: JSONEncoder().encode(source))
        let folder = jobs.appendingPathComponent("runs/\(UUID().uuidString)")
        var request: [String:Any] = ["source":sourceJSON,"method":method,"settings":settings,"purpose":runningTrial ? "trial" : "comparison"]
        if method == "cellpose" { request["comparisonSide"] = side == 1 ? "A" : "B" }
        if runningTrial && reuseTrial, let result = trialResult { request["reuse"] = URL(fileURLWithPath:result.folder).appendingPathComponent("result.json").path }
        execute("run", request: request, output: folder.appendingPathComponent("result.json"), panseg: method == "panseg", focus:method == "focus3d") { [weak self] url in
            guard let self else { return }
            do {
                let result = try JSONDecoder().decode(SegResult.self, from: Data(contentsOf: url))
                self.results.insert(result, at: 0)
                if self.runningTrial { self.chooseTrial(result); self.status = "Trial finished · \(result.cells.count) cells · \(Int(result.seconds)) s"; return }
                self.choose(result,side:method == "cellpose" ? side : 2)
                if self.runQueue.isEmpty { self.status = "Finished \(method) · \(result.cells.count) cells · \(Int(result.seconds)) s"; self.compare() }
                else { self.runNext() }
            } catch { self.error = error.localizedDescription; self.runQueue = [] }
        }
    }
    func importLabels(method: String, existing: Bool = false, side: Int? = nil) {
        guard canRun, let source else { return }
        var path: String?
        if existing, let baseline {
            let native = URL(fileURLWithPath: baseline).deletingLastPathComponent().appendingPathComponent("labels_native.tif")
            if FileManager.default.fileExists(atPath: native.path) { path = native.path }
        }
        if path == nil {
            let panel = NSOpenPanel(); panel.allowedContentTypes = [.tiff]; panel.message = "Choose \(method) instance labels on the same native ZYX image grid; background must be 0."
            if panel.runModal() == .OK { path = panel.url?.path }
        }
        guard let path else { return }
        let folder = jobs.appendingPathComponent("runs/\(UUID().uuidString)")
        var request: [String:Any] = ["source": try! JSONSerialization.jsonObject(with: JSONEncoder().encode(source)), "labels": path, "method": method]
        if method == "cellpose" { request["comparisonSide"] = side == 2 ? "B" : "A" }
        execute("import", request:request, output: folder.appendingPathComponent("result.json")) { [weak self] url in
            guard let self else { return }
            do {
                let result = try JSONDecoder().decode(SegResult.self, from: Data(contentsOf: url))
                self.results.insert(result, at: 0); self.choose(result,side:side); self.compare()
            } catch { self.error = error.localizedDescription }
        }
    }
    func compare() {
        guard !busy, let a = resultA, let b = resultB, let source else { return }
        do {
            // Preserve exact heterogeneous settings in backend manifests.
            let request: [String: Any] = ["a": try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath:a.folder).appendingPathComponent("result.json"))), "b": try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath:b.folder).appendingPathComponent("result.json"))), "source": try JSONSerialization.jsonObject(with: JSONEncoder().encode(source))]
            execute("compare", request: request, output: jobs.appendingPathComponent("comparisons/\(UUID().uuidString).json")) { [weak self] url in
                guard let self else { return }
                do { self.agreement = try JSONDecoder().decode(SegAgreement.self, from: Data(contentsOf:url)); self.status = "Comparison ready · main-organoid cells on native grid" }
                catch { self.error = error.localizedDescription }
            }
        } catch { self.error = error.localizedDescription }
    }
    func settingsText(_ result: SegResult) -> String {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath:result.folder).appendingPathComponent("result.json")),
              let json = try? JSONSerialization.jsonObject(with:data) as? [String:Any], let settings = json["settings"],
              let text = try? JSONSerialization.data(withJSONObject:settings, options:[.prettyPrinted,.sortedKeys]) else { return "Settings unavailable" }
        return String(data:text,encoding:.utf8) ?? "Settings unavailable"
    }
    func cancel() {
        runQueue = []; generation = UUID()
        if let p = process, p.isRunning { if kill(-p.processIdentifier, SIGTERM) != 0 { p.terminate() }; let pid = p.processIdentifier; DispatchQueue.global().asyncAfter(deadline: .now()+2) { if p.isRunning { kill(-pid, SIGKILL); kill(pid, SIGKILL) } } }
        process = nil; busy = false; status = "Cancelled. Incomplete jobs are excluded from run history."
    }
    func consumeOutput(_ text: String) {
        log = String((log + text).suffix(30000))
        pendingOutput += text
        while let delimiter = pendingOutput.firstIndex(where: { $0 == "\n" || $0 == "\r" }) {
            let line = String(pendingOutput[..<delimiter])
            pendingOutput.removeSubrange(...delimiter)
            if line.hasPrefix("@atlas-progress "),
               let event = try? JSONDecoder().decode(SegProgressEvent.self, from: Data(line.dropFirst(16).utf8)),
               event.fraction.isFinite, event.span.isFinite {
                progressBase = min(1,max(0,event.fraction))
                progressSpan = min(1-progressBase,max(0,event.span))
                progress = max(progress,progressBase)
                progressStage = event.stage
            } else if progressSpan > 0 {
                // Cellpose emits tqdm records separated by carriage returns.
                let pattern = #"\|\s*(\d+)\s*/\s*(\d+)"#
                if let expression = try? NSRegularExpression(pattern:pattern),
                   let match = expression.firstMatch(in:line,range:NSRange(line.startIndex...,in:line)),
                   let currentRange = Range(match.range(at:1),in:line), let totalRange = Range(match.range(at:2),in:line),
                   let current = Double(line[currentRange]), let total = Double(line[totalRange]), total > 0 {
                    progress = max(progress,min(1,progressBase + progressSpan * min(1,current/total)))
                }
            }
        }
        pendingOutput = String(pendingOutput.suffix(30000))
    }
    private func execute(_ operation: String, request: [String: Any], output: URL, panseg: Bool = false, focus: Bool = false, completion: @escaping (URL) -> Void) {
        guard !busy else { return }
        do {
            let requestDir = jobs.appendingPathComponent("requests")
            try FileManager.default.createDirectory(at: requestDir, withIntermediateDirectories:true)
            // Prediction creates its own fresh directory; other commands may share existing directories.
            if operation != "run" && operation != "refine" { try FileManager.default.createDirectory(at:output.deletingLastPathComponent(), withIntermediateDirectories:true) }
            else { try FileManager.default.createDirectory(at: output.deletingLastPathComponent().deletingLastPathComponent(), withIntermediateDirectories:true) }
            let req = requestDir.appendingPathComponent("\(UUID().uuidString).json")
            try JSONSerialization.data(withJSONObject:request,options:[.prettyPrinted,.sortedKeys]).write(to:req)
            let python = root.appendingPathComponent(focus ? ".cache/focus3d-env/bin/python" : panseg ? ".cache/panseg-env/bin/python" : ".venv/bin/python")
            guard FileManager.default.isExecutableFile(atPath:python.path) else { throw DataError.invalid("Missing Python environment at \(python.path)") }
            let p = Process(); p.executableURL = python; p.currentDirectoryURL = root
            p.arguments = ["-u",root.appendingPathComponent("scripts/segmentation_explorer.py").path,operation,"--request",req.path,"--output",output.path]
            var env = ProcessInfo.processInfo.environment; env["PYTHONUNBUFFERED"] = "1"; env["OMP_NUM_THREADS"] = "4"; env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"; p.environment = env
            let pipe = Pipe(); p.standardOutput = pipe; p.standardError = pipe
            let token = UUID(); generation = token; busy = true; log = ""; pendingOutput = ""; progress = 0; progressBase = 0; progressSpan = 0; progressStage = "Starting worker…"; status = operation == "run" ? "Running \(request["method"] as? String ?? "segmentation")…" : "\(operation.capitalized)…"
            pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
                let data = handle.availableData
                if !data.isEmpty { let text = String(decoding:data,as:UTF8.self); Task { @MainActor in guard let self, self.generation == token else { return }; self.consumeOutput(text) } }
            }
            p.terminationHandler = { [weak self] process in
                pipe.fileHandleForReading.readabilityHandler = nil
                Task { @MainActor in
                    guard let self, self.generation == token else { return }
                    self.busy = false; self.process = nil
                    if process.terminationStatus == 0 { self.progress = 1; completion(output) }
                    else { self.runQueue = []; self.status = "Job failed · see run log"; self.error = "\(operation.capitalized) failed (exit \(process.terminationStatus)).\n\(self.log.suffix(3000))" }
                }
            }
            process = p; try p.run()
        } catch { busy = false; process = nil; runQueue = []; self.error = error.localizedDescription }
    }
    func dimensions() -> (Int,Int,Int) {
        guard let s = source?.shape else { return (1,1,1) }
        return axis == 0 ? (s[2],s[1],s[0]) : axis == 1 ? (s[2],s[0],s[1]) : (s[1],s[0],s[2])
    }
    func index(_ x: Int, _ y: Int) -> Int {
        guard let s = source?.shape else { return 0 }
        let z = min(dimensions().2-1,max(0,Int(slice * Double(dimensions().2-1))))
        return axis == 0 ? (z*s[1]+y)*s[2]+x : axis == 1 ? (y*s[1]+z)*s[2]+x : (y*s[1]+x)*s[2]+z
    }
    func select(label: Int, side: Int, center: Bool = false) {
        if comparisonMethod == "refined" {
            selectedA = label > 0 && labelsA.contains(UInt32(label)) ? label : 0
            selectedB = label > 0 && labelsB.contains(UInt32(label)) ? label : 0
        } else if side == 1 { selectedA = label; selectedB = agreement?.pairs.first(where: { $0.a == label })?.b ?? 0 }
        else { selectedB = label; selectedA = agreement?.pairs.first(where: { $0.b == label })?.a ?? 0 }
        if center, let s = source, let c = (side == 1 ? resultA : resultB)?.cells.first(where: { $0.label == label }) {
            let a = axis == 0 ? 0 : axis == 1 ? 1 : 2
            slice = min(1,max(0,c.centroid[a] / Double(max(1,s.nativeShape[a]-1))))
        }
    }
    func excludedLabels(side: Int) -> Set<Int> {
        let result = side == 1 ? resultA : side == 5 ? trialResult : resultB
        let automatic = result?.cells.filter { !$0.includedInAnalysis }.map(\.label) ?? []
        return Set(automatic + (side == 1 ? agreement?.excludedA ?? [] : side == 5 ? [] : agreement?.excludedB ?? []))
    }
    func analysisCells(side: Int) -> [SegCell] {
        let result = side == 1 ? resultA : side == 5 ? trialResult : resultB
        let excluded = excludedLabels(side: side)
        return result?.cells.filter { !excluded.contains($0.label) } ?? []
    }
    func image(side: Int) -> CGImage? {
        guard source != nil, !raw.isEmpty else { return nil }
        let (w,h,_) = dimensions(); var pixels = [UInt8](repeating:0,count:w*h*4)
        let labels = side == 1 ? labelsA : side == 5 ? trialLabels : labelsB
        let excluded = excludedLabels(side: side), excludedA = excludedLabels(side: 1), excludedB = excludedLabels(side: 2)
        let matched = Dictionary(uniqueKeysWithValues:(agreement?.pairs.filter { $0.iou >= 0.5 } ?? []).map { ($0.b,$0.a) })
        for y in 0..<h { for x in 0..<w {
            let i = index(x,y), j=(y*w+x)*4; let gray = min(255,Double(side == 6 && !trialBoundaries.isEmpty ? trialBoundaries[i] : raw[i])*brightness)
            var color = [gray,gray,gray]; var alpha = 0.0
            let label = labels.isEmpty || side == 6 ? 0 : Int(labels[i]); let selected = side == 1 ? selectedA : side == 5 ? selectedTrial : selectedB
            if side == 7, !labelsA.isEmpty, !labelsB.isEmpty {
                let a=excludedA.contains(Int(labelsA[i])) ? 0 : Int(labelsA[i]), b=excludedB.contains(Int(labelsB[i])) ? 0 : Int(labelsB[i])
                if a == 0 && b > 0 { color=[0,230,180]; alpha=0.8 }
                else if a > 0 && b == 0 { color=[245,70,200]; alpha=0.8 }
                else if a != b { color=[255,180,40]; alpha=0.8 }
            } else if side == 7 {
                alpha = 0
            } else if side == 3, !labelsA.isEmpty, !labelsB.isEmpty {
                let a=excludedA.contains(Int(labelsA[i])) ? 0 : Int(labelsA[i]), b=excludedB.contains(Int(labelsB[i])) ? 0 : Int(labelsB[i])
                if a > 0 && b == 0 { color=[0,230,220];alpha=0.75 }
                else if a == 0 && b > 0 { color=[245,70,200];alpha=0.75 }
                else if a > 0 && b > 0 && matched[b] != a { color=[255,180,40];alpha=0.65 }
            } else if side > 0 && overlay && label > 0 {
                let key = side == 1 || side == 5 || comparisonMethod == "refined" ? label : matched[label] ?? label+100003
                let c = NSColor(calibratedHue:Double((key*137)%360)/360,saturation:0.7,brightness:0.95,alpha:1).usingColorSpace(.deviceRGB)!
                color=[c.redComponent*255,c.greenComponent*255,c.blueComponent*255];alpha=0.4
                if label == selected { color=[255,200,40];alpha=0.75 }
                if excluded.contains(label) { color=[255,65,90]; alpha=0.55 }
                if boundaries && label != selected {
                    let edge = (x>0 && labels[index(x-1,y)] != labels[i]) || (y>0 && labels[index(x,y-1)] != labels[i]) || (x<w-1 && labels[index(x+1,y)] != labels[i]) || (y<h-1 && labels[index(x,y+1)] != labels[i])
                    alpha = edge ? 0.85 : 0
                }
            }
            for k in 0..<3 { pixels[j+k]=UInt8(max(0,min(255,gray*(1-alpha)+color[k]*alpha))) }; pixels[j+3]=255
        } }
        guard let p = CGDataProvider(data:Data(pixels) as CFData) else { return nil }
        return CGImage(width:w,height:h,bitsPerComponent:8,bitsPerPixel:32,bytesPerRow:w*4,space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGBitmapInfo(rawValue:CGImageAlphaInfo.last.rawValue),provider:p,decode:nil,shouldInterpolate:false,intent:.defaultIntent)
    }
}
struct SegmentationView: View {
    @EnvironmentObject var atlas: ExplorerStore
    @EnvironmentObject var store: SegmentationStore
    @State private var stackID = ""
    @State private var showLog = false
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Cell segmentation lab").font(.system(size:26,weight:.semibold))
                        Text("One raw stack · two image-only pipelines · linked slices and cell matching").foregroundStyle(.secondary)
                    }
                    Spacer()
                    if store.busy { ProgressView().controlSize(.small); Button("Cancel") { store.cancel() } }
                }
                HStack {
                    Picker("Experiment stack", selection: $stackID) {
                        Text("Choose stack…").tag("")
                        ForEach(atlas.dataset?.samples ?? []) { Text($0.id).tag($0.id) }
                    }.frame(maxWidth:260).disabled(store.busy)
                    Button("Open raw TIFF…") { store.open(); stackID = "" }.disabled(store.busy)
                    Spacer()
                    Text(store.status).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                }
                if let source = store.source {
                    Text(source.path).font(.caption.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)
                }
                if store.busy { runningProgress }
                Picker("Compare Cellpose with",selection:Binding(get:{store.comparisonMethod},set:{store.setComparisonMethod($0)})) {
                    Text("FOCUS-3D").tag("focus3d")
                    Text("PanSeg").tag("panseg")
                    Text("Cellpose model / 3D mode").tag("cellpose")
                }.pickerStyle(.segmented).frame(maxWidth:420).disabled(store.busy)
                configuration
                Button("Open Post Processing…") { atlas.activeTab = "postprocessing" }
                if let source = store.source {
                    HStack {
                        Picker("Plane",selection:$store.axis) { Text("XY").tag(0);Text("XZ").tag(1);Text("YZ").tag(2) }.pickerStyle(.segmented).frame(width:160)
                        Slider(value:$store.slice,in:0...1)
                        let a = store.axis == 0 ? 0 : store.axis == 1 ? 1 : 2
                        Text("\(Int(store.slice * Double(source.nativeShape[a]-1))+1)/\(source.nativeShape[a])").font(.caption.monospacedDigit()).frame(width:80)
                        Text("Brightness").font(.caption); Slider(value:$store.brightness,in:0.2...4).frame(width:100)
                        Toggle("Overlay",isOn:$store.overlay); Toggle("Contours",isOn:$store.boundaries)
                    }
                    HStack(alignment:.top,spacing:12) {
                        slicePanel("Raw microscopy",side:0)
                        slicePanel(store.leftName,side:1)
                        slicePanel(store.rightName,side:2)
                    }
                    HStack(alignment:.top,spacing:16) {
                        VStack(alignment:.leading,spacing:8) {
                            slicePanel("Disagreement",side:3)
                            Text("Cyan: \(store.leftName) foreground only · Magenta: \(store.rightName) only\nAmber: overlapping instances without a match at IoU ≥ 0.5").font(.caption2).foregroundStyle(.secondary)
                        }.frame(maxWidth:.infinity)
                        VStack(alignment:.leading,spacing:12) {
                            Text("Method agreement").font(.headline)
                            if let a = store.agreement {
                                HStack(spacing:25) {
                                    metric("Foreground Dice",value:String(format:"%.3f",a.dice))
                                    metric("Matched cells",value:"\(a.matched)")
                                    metric("Mean matched IoU",value:String(format:"%.3f",a.meanIoU))
                                }
                                Text(a.note).font(.caption).foregroundStyle(.secondary)
                            } else { Text("Run or load both methods on this stack to compare.").foregroundStyle(.secondary) }
                            Text("Click a cell in either overlay or table to highlight its overlap partner. Table selections center the shared slice.").font(.caption).foregroundStyle(.secondary)
                            selectedPair
                            HStack(alignment:.top) { cellTable(side:1);cellTable(side:2) }
                        }.frame(maxWidth:.infinity)
                    }
                    Text("Display preview: \(source.shape.map(String.init).joined(separator:" × ")). Predictions, volumes and matching use the full original stack. Matched cells share colors; unmatched IDs are independent.").font(.caption2).foregroundStyle(.secondary)
                } else {
                    ContentUnavailableView("Load a raw microscopy stack",systemImage:"square.stack.3d.up",description:Text("Choose an experiment stack above, or open a new single-channel 3D TIFF."))
                }
                DisclosureGroup("Run log & provenance",isExpanded:$showLog) {
                    VStack(alignment:.leading,spacing:8) {
                        if let a = store.resultA { provenance(a) }
                        if let b = store.resultB { provenance(b) }
                        Text(store.log.isEmpty ? "No active run log." : store.log).font(.system(size:10,design:.monospaced)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading)
                    }
                }
            }.padding(20)
        }
        .onAppear {
            if store.source != nil { stackID = atlas.dataset?.samples.first(where: {$0.sourceImage == store.source?.path})?.id ?? "" }
            else if !store.busy && stackID.isEmpty {
                stackID = atlas.selected?.sample ?? atlas.dataset?.samples.first?.id ?? ""
            }
        }
        .onChange(of:stackID) {
            guard !stackID.isEmpty, let sample = atlas.dataset?.samples.first(where:{$0.id == stackID}) else { return }
            // Dataset spacing describes its preview; prefer original run calibration.
            let settings = URL(fileURLWithPath:sample.sourceLabels).deletingLastPathComponent().appendingPathComponent("settings.json")
            var spacing: [Double]? = nil
            if let data = try? Data(contentsOf:settings), let json = try? JSONSerialization.jsonObject(with:data) as? [String:Any] { spacing = (json["config"] as? [String:Any] ?? json)["spacing_zyx_um"] as? [Double] }
            if store.source?.path != sample.sourceImage { store.prepare(sample.sourceImage,baseline:sample.sourceLabels,spacing:spacing) }
        }
        .alert("Segmentation action failed",isPresented:Binding(get:{store.error != nil && atlas.activeTab == "segmentation"},set:{if !$0 {store.error=nil}})) { Button("OK") {store.error=nil} } message: { Text(store.error ?? "") }
    }
    var runningProgress: some View {
        VStack(alignment:.leading,spacing:6) {
            HStack {
                Text(store.progressStage).font(.callout.weight(.medium))
                Spacer()
                Text("\(Int(store.progress * 100))%").font(.callout.monospacedDigit())
            }
            ProgressView(value:store.progress,total:1).progressViewStyle(.linear)
            Text("Stage-based progress; inference updates by completed slices or patches. Stage durations vary.")
                .font(.caption2).foregroundStyle(.secondary)
        }.padding(12).background(Color.teal.opacity(0.08),in:RoundedRectangle(cornerRadius:8))
    }
    @ViewBuilder var selectedPair: some View {
        if let pair = store.agreement?.pairs.first(where: {$0.a == store.selectedA && $0.b == store.selectedB}) {
            HStack {
                Text("Selected pair: A \(pair.a) ↔ \(store.comparisonName) \(pair.b)")
                Text("IoU \(pair.iou,specifier:"%.3f")")
                if pair.iou < 0.5 { Text("Below match threshold") }
            }.font(.caption).foregroundStyle(.orange)
        }
    }
    var configuration: some View {
                HStack(alignment:.top,spacing:12) {
                    GroupBox("Raw stack & calibration") {
                        VStack(alignment:.leading,spacing:8) {
                            Text("Single-channel TIFF · ZYX order").font(.caption).foregroundStyle(.secondary)
                            HStack {
                                spacingField("Z",value:$store.spacingZ)
                                spacingField("Y",value:$store.spacingY)
                                spacingField("X",value:$store.spacingX)
                            }
                            Text("Voxel size in µm. Verify before running.").font(.caption2).foregroundStyle(.secondary)
                            if store.calibrationChanged { Button("Apply calibration") { store.reloadCalibration() }.disabled(store.busy) }
                            Button("Run A + \(store.comparisonName)") { store.run(["cellpose",store.comparisonMethod == "cellpose" ? "cellposeB" : store.comparisonMethod]) }.buttonStyle(.borderedProminent).disabled(!store.canRun || !store.comparisonReady)
                        }.frame(maxWidth:.infinity,alignment:.leading).padding(6)
                    }
                    if store.comparisonMethod == "refined" {
                        GroupBox("Original / membrane-refined") {
                            Text("The original saved labels are displayed on the left, with a separate refined run on the right. Select a refined run below to restore its parent. Change the comparison mode to return to model comparison.").font(.caption).padding(8)
                        }
                    } else {
                    GroupBox("Cellpose A + u-Segment3D") { CellposeControls(side:1).environmentObject(store) }
                    if store.comparisonMethod == "focus3d" {
                        GroupBox("FOCUS-3D") { FocusControls().environmentObject(store) }
                    } else if store.comparisonMethod == "cellpose" {
                        GroupBox("Cellpose B + u-Segment3D") { CellposeControls(side:2).environmentObject(store) }
                    } else {
                    GroupBox("PanSeg") {
                        VStack(alignment:.leading,spacing:8) {
                            Text(store.comparisonSettings.summary).font(.caption).fixedSize(horizontal:false,vertical:true)
                            Text("Tune a pipeline in PanSeg Try Out, then transfer the successful run and its saved settings here.").font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal:false,vertical:true)
                            Button("Open PanSeg Try Out") { atlas.activeTab = "panseg" }
                            HStack { Button("Run transferred settings") { store.run(["panseg"]) }.disabled(!store.canRun || !store.pansegInstalled); Button("Import labels…") { store.importLabels(method:"panseg") }.disabled(!store.canRun) }
                            if !store.pansegInstalled { Text("Install with scripts/setup_panseg.sh").font(.caption).foregroundStyle(.orange) }
                        }.frame(maxWidth:.infinity,alignment:.leading).padding(6)
                    }
                    }
                    }
                }.textFieldStyle(.roundedBorder).disabled(store.busy)
    }
    func spacingField(_ label:String,value:Binding<Double>) -> some View { HStack(spacing:3) { Text(label).font(.caption);TextField(label,value:value,format:.number).frame(minWidth:45) } }
    func metric(_ name:String,value:String) -> some View { VStack(alignment:.leading) { Text(value).font(.title2.monospacedDigit().bold());Text(name).font(.caption2).foregroundStyle(.secondary) } }
    func provenance(_ result:SegResult) -> some View {
        VStack(alignment:.leading,spacing:4) {
            Text(store.methodName(result.method)).bold()
            Text(result.provenance).font(.caption)
            Text(store.settingsText(result)).font(.system(size:10,design:.monospaced)).textSelection(.enabled)
            HStack { Text("\(result.created) · \(Int(result.seconds)) s").font(.caption);Button("Reveal labels & settings") {NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath:result.folder)])} }
        }
    }
    func slicePanel(_ title:String,side:Int) -> some View {
        VStack(alignment:.leading,spacing:8) {
            Text(title).font(.headline)
            if side == 1 || side == 2 {
                let runs = store.results.filter { side == 1 && store.comparisonMethod == "refined" ? $0.id == store.resultB?.refinement?.parentID : $0.method == (side == 1 ? "cellpose" : store.comparisonMethod) }
                Picker("Run",selection:Binding(get:{ (side == 1 ? store.resultA : store.resultB)?.id ?? "" },set:{store.selectRun($0,side:side)})) {
                    Text("No result").tag("")
                    ForEach(runs) { Text($0.title).tag($0.id) }
                }.labelsHidden().disabled(store.busy)
            } else { Text(side == 0 ? "Shared plane, slice and brightness" : "Full-grid instance matches").font(.caption).foregroundStyle(.secondary).frame(height:24) }
            GeometryReader { geo in
                if let image = store.image(side:side), let source = store.source {
                    let a = store.axis == 0 ? (2,1) : store.axis == 1 ? (2,0) : (1,0)
                    let aspect = Double(source.nativeShape[a.0])*source.spacing[a.0] / (Double(source.nativeShape[a.1])*source.spacing[a.1])
                    let width=min(geo.size.width,geo.size.height*aspect),height=width/aspect
                    Image(decorative:image,scale:1).resizable().interpolation(.none).frame(width:width,height:height)
                        .contentShape(Rectangle())
                        .gesture(SpatialTapGesture().onEnded { tap in
                            guard side == 1 || side == 2 else { return }
                            let (w,h,_) = store.dimensions(); let x=min(w-1,max(0,Int(tap.location.x / width * Double(w)))),y=min(h-1,max(0,Int(tap.location.y / height * Double(h))))
                            let labels=side == 1 ? store.labelsA : store.labelsB
                            if !labels.isEmpty { store.select(label:Int(labels[store.index(x,y)]),side:side) }
                        })
                        .position(x:geo.size.width/2,y:geo.size.height/2)
                    if (side == 1 && store.resultA == nil) || (side == 2 && store.resultB == nil) || (side == 3 && store.agreement == nil) {
                        Text(side == 3 ? "Awaiting comparison" : "Awaiting segmentation").padding(8).background(.black.opacity(0.7),in:RoundedRectangle(cornerRadius:6)).foregroundStyle(.white).frame(maxWidth:.infinity,maxHeight:.infinity)
                    }
                }
            }.frame(height:side == 3 ? 180 : 245).background(.black,in:RoundedRectangle(cornerRadius:8)).clipped()
            if side == 1 || side == 2 {
                let label = side == 1 ? store.selectedA : store.selectedB
                let result = side == 1 ? store.resultA : store.resultB
                Text(label > 0 ? "Cell \(label) · \(result?.cells.first(where:{$0.label == label})?.volume ?? 0,specifier:"%.1f") µm³" : "Click an instance to inspect").font(.caption).foregroundStyle(label > 0 ? .orange : .secondary)
            }
        }.frame(maxWidth:.infinity)
    }
    func cellTable(side:Int) -> some View {
        let result = side == 1 ? store.resultA : store.resultB
        return VStack(alignment:.leading,spacing:5) {
            Text(side == 1 ? (store.comparisonMethod == "refined" ? "Original cells" : "Cellpose A cells") : "\(store.comparisonName) cells").font(.caption.bold())
            ScrollView {
                LazyVStack(spacing:2) {
                    ForEach(store.analysisCells(side: side)) { cell in
                        Button {store.select(label:cell.label,side:side,center:true)} label: {
                            HStack { Text("\(cell.label)").monospacedDigit();Spacer();Text("\(cell.volume,specifier:"%.0f") µm³").monospacedDigit();if cell.border {Image(systemName:"square.dashed")} }
                                .font(.caption).padding(5).background((side == 1 ? store.selectedA : store.selectedB) == cell.label ? Color.orange.opacity(0.25) : Color.clear,in:RoundedRectangle(cornerRadius:4))
                        }.buttonStyle(.plain)
                    }
                }
            }.frame(height:135)
            Text("Red overlay: excluded detached cells · dashed square: stack border").font(.caption2).foregroundStyle(.secondary)
        }.frame(maxWidth:.infinity)
    }
}
