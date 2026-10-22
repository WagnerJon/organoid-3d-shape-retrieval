import SwiftUI
import AppKit

struct Representation: Codable {
    let vector: [Double]
    let point: [Double]
    let cluster: Int
}
struct Method: Codable, Identifiable {
    let id, name, metric: String
    let variance, stackVariance: [Double]
    let dimensions: Int
    var family: String? = nil
    var description: String? = nil
    var projection: String? = nil
    var scope: String? = nil
    var coverage: String? = nil
}
struct Specimen: Codable, Identifiable {
    let id, sample, split, flags: String
    let label: Int
    let qc: Bool
    let metrics: [String: Double]
    let representations: [String: Representation]
    let cellCount: Int?
    var inMainOrganoid: Bool? = nil
    var automaticMainOrganoid: Bool? = nil
    var mainOrganoidFraction: Double? = nil
    var membershipOverride: Bool? = nil
    var pointReconstructions: [String:String]? = nil
    var includedInAnalysis: Bool { inMainOrganoid ?? true }
    var volume: Double { metrics["volume_um3"] ?? 0 }
    var sphericity: Double { metrics["sphericity"] ?? 0 }
    var elongation: Double { metrics["elongation"] ?? 0 }
}
struct Sample: Codable, Identifiable {
    let id: String
    let shape: [Int] // ZYX
    let spacing, extent: [Double]
    let rawAvailable: Bool
    let sourceImage, sourceLabels: String
    let meshCount: Int
    var name: String? = nil
    var displayName: String { name ?? id }
}
struct Dataset: Codable {
    let version: Int
    let title, provenance: String
    let methods: [Method]
    let samples: [Sample]
    let cells, organoids: [Specimen]
    func validate() throws {
        guard version == 1, !cells.isEmpty, !methods.isEmpty else { throw DataError.invalid("Unsupported or empty dataset") }
        let sampleIDs = Set(samples.map(\.id))
        guard sampleIDs.count == samples.count, Set(cells.map(\.id)).count == cells.count else { throw DataError.invalid("Duplicate identifiers") }
        for s in samples {
            guard s.shape.count == 3, s.shape.allSatisfy({ $0 > 0 && $0 <= 1024 }), s.extent.count == 3,
                  s.extent.allSatisfy({ $0.isFinite && $0 > 0 }), !s.id.contains("/"), s.id != ".." else { throw DataError.invalid("Invalid sample dimensions or path") }
        }
        for item in cells + organoids {
            guard sampleIDs.contains(item.sample) else { throw DataError.invalid("Missing parent stack for \(item.id)") }
            for path in item.pointReconstructions?.values ?? Dictionary<String,String>().values {
                guard !path.hasPrefix("/"), !path.split(separator:"/").contains("..") else { throw DataError.invalid("Invalid learned reconstruction path") }
            }
            for method in methods {
                if let rep = item.representations[method.id] {
                    guard rep.vector.count == method.dimensions, rep.point.count == 2,
                          (rep.vector + rep.point).allSatisfy(\.isFinite) else { throw DataError.invalid("Invalid representation for \(item.id)") }
                }
            }
        }
    }
}
enum DataError: LocalizedError {
    case invalid(String)
    var errorDescription: String? { if case .invalid(let message) = self { return message }; return nil }
}
func representationDistance(_ a: [Double], _ b: [Double], metric: String) -> Double {
    guard a.count == b.count, !a.isEmpty else { return .infinity }
    if metric == "precomputed" {
        guard a.count > 1, a[0].isFinite, b[0].isFinite,
              a[0] >= 0, b[0] >= 0, a[0] < Double(a.count-1), b[0] < Double(b.count-1),
              a[0].rounded() == a[0], b[0].rounded() == b[0] else { return .infinity }
        let forward = a[Int(b[0])+1], reverse = b[Int(a[0])+1]
        guard forward.isFinite, reverse.isFinite, forward >= 0, reverse >= 0,
              abs(forward-reverse) < 1e-8 else { return .infinity }
        return (forward+reverse)/2
    }
    if metric == "shape_hybrid95" {
        guard a.count == 476 else { return .infinity }
        if a[0] > 0.5 || b[0] > 0.5 {
            return sqrt(zip(a[381...], b[381...]).reduce(0) { $0 + pow($1.0 - $1.1, 2) })
        }
        var best = Double.infinity
        for i in 0..<4 { for j in 0..<4 {
            var squared = 0.0
            for k in 0..<95 { squared += pow(a[1+i*95+k] - b[1+j*95+k], 2) }
            best = min(best, squared)
        } }
        return sqrt(best)
    }
    if metric == "cosine" {
        let dot = zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
        let norms = sqrt(a.reduce(0) { $0 + $1*$1 } * b.reduce(0) { $0 + $1*$1 })
        return norms > 1e-12 ? max(0, min(2, 1 - dot / norms)) : .infinity
    }
    return sqrt(zip(a, b).reduce(0) { $0 + pow($1.0 - $1.1, 2) })
}
struct Neighbor: Identifiable {
    let item: Specimen
    let distance: Double
    var id: String { item.id }
}
struct VolumeData {
    let cacheID = UUID()
    private let sliceCache = VolumeSliceCache()
    let sample: Sample
    let raw: [UInt8]
    let labels: [UInt16]
    init(folder: URL, sample: Sample) throws {
        self.sample = sample
        let count = sample.shape.reduce(1, *)
        let data = try Data(contentsOf: folder.appendingPathComponent("labels.u16"))
        guard data.count == count * 2 else { throw DataError.invalid("Label preview size mismatch") }
        labels = data.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
        if sample.rawAvailable {
            let bytes = try Data(contentsOf: folder.appendingPathComponent("raw.u8"))
            guard bytes.count == count else { throw DataError.invalid("Raw preview size mismatch") }
            raw = Array(bytes)
        } else { raw = [UInt8](repeating: 15, count: count) }
    }
    func dimensions(axis: Int) -> (Int, Int, Int) {
        switch axis {
        case 0: return (sample.shape[2], sample.shape[1], sample.shape[0])
        case 1: return (sample.shape[2], sample.shape[0], sample.shape[1])
        default: return (sample.shape[1], sample.shape[0], sample.shape[2])
        }
    }
    func index(x: Int, y: Int, slice: Int, axis: Int) -> Int {
        let w = sample.shape[2], h = sample.shape[1]
        switch axis {
        case 0: return (slice * h + y) * w + x
        case 1: return (y * h + slice) * w + x
        default: return (y * h + x) * w + slice
        }
    }
    func image(axis: Int, fraction: Double, selected: Int, overlay: Bool, contrast: Double, excluded: Set<Int> = [], hideExcluded: Bool = false) -> CGImage? {
        image(style: VolumeSliceStyle(axis:axis, selected:selected, overlay:overlay, contrast:contrast,
                                      excluded:excluded, hideExcluded:hideExcluded), fraction:fraction)
    }
    func image(style: VolumeSliceStyle, fraction: Double) -> CGImage? {
        let depth = dimensions(axis:style.axis).2
        return image(style:style, slice:min(depth-1,max(0,Int(fraction * Double(depth-1)))))
    }
    /// Warm the entire displayed plane stack, prioritizing the current slice.
    /// Very large custom previews use a bounded neighbourhood rather than evicting every frame.
    func prepareSlices(style: VolumeSliceStyle, near fraction: Double) {
        let (width,height,depth) = dimensions(axis:style.axis)
        let current = min(depth-1,max(0,Int(fraction * Double(depth-1))))
        let capacity = max(1,VolumeSliceCache.byteLimit / (width*height*4))
        let order = (0..<depth).sorted {
            let a = abs($0-current), b = abs($1-current)
            return a == b ? $0 < $1 : a < b
        }.prefix(capacity)
        for index in order {
            if Task.isCancelled { return }
            _ = image(style:style, slice:index)
        }
    }
    private func image(style: VolumeSliceStyle, slice: Int) -> CGImage? {
        if let cached = sliceCache.image(style:style,index:slice) { return cached }
        let axis = style.axis, selected = style.selected, overlay = style.overlay
        let contrast = style.contrast, excluded = style.excluded, hideExcluded = style.hideExcluded
        let (w,h,_) = dimensions(axis:axis)
        var pixels = [UInt8](repeating: 0, count: w*h*4)
        for y in 0..<h { for x in 0..<w {
            let i = index(x: x, y: y, slice: slice, axis: axis), j = (y*w+x)*4
            let gray = UInt8(min(255, Double(raw[i]) * contrast))
            pixels[j] = gray; pixels[j+1] = gray; pixels[j+2] = gray; pixels[j+3] = 255
            if overlay && labels[i] > 0 && !(hideExcluded && excluded.contains(Int(labels[i]))) {
                let color: (Double, Double, Double) = excluded.contains(Int(labels[i])) ? (255,70,100) : Int(labels[i]) == selected ? (255,190,50) : (45,210,190)
                let alpha = Int(labels[i]) == selected ? 0.65 : 0.25
                pixels[j] = UInt8(Double(gray)*(1-alpha)+color.0*alpha)
                pixels[j+1] = UInt8(Double(gray)*(1-alpha)+color.1*alpha)
                pixels[j+2] = UInt8(Double(gray)*(1-alpha)+color.2*alpha)
            }
        } }
        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
        guard let image = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w*4,
                       space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { return nil }
        sliceCache.insert(image, style: style, index: slice)
        return image
    }
}

@MainActor final class SliceNavigation: ObservableObject {
    @Published var fraction = 0.5
}

@MainActor final class ExplorerStore: ObservableObject {
    @Published var dataset: Dataset?
    @Published var folder: URL?
    @Published var error: String?
    @Published var loading = false
    @Published var membershipBusy = false
    @Published var analysisRevision = UUID()
    @Published var level = "Cells"
    @Published var activeTab = "explore"
    @Published var referenceMethod = "traditional"
    @Published var sharedMethodColors = true
    @Published var plotColorSettings: [String: PlotColorSettings] = [:]
    @Published var unifiedPlotColorLevels: Set<String> = []
    @Published var unifiedPlotColorSettings: [String: PlotColorSettings] = [:]
    @Published var selectedID: String?
    @Published var method = "simclr3d_raw_isolated"
    @Published var search = ""
    @Published var qcOnly = true
    @Published var sampleFilter = "All stacks"
    @Published var clusterFilter = -1
    @Published var splitFilter = "All splits"
    @Published var excludeSameStack = true
    @Published var selection: Set<String> = [] {
        didSet { tableSelection = selection }
    }
    @Published var tableSelection: Set<String> = []
    @Published var markedOnly = false
    @Published var axis = 0
    let sliceNavigation = SliceNavigation()
    var slice: Double {
        get { sliceNavigation.fraction }
        set { sliceNavigation.fraction = newValue }
    }
    @Published var clip = false
    @Published var isolate = false
    @Published var hideExcluded = false
    @Published var overlay = true
    @Published var contrast = 1.0
    @Published var volume: VolumeData?
    @Published var volumeError: String?
    @Published var compared: Specimen?
    @Published var showPreparation = false
    @Published var showCompare = false
    @Published var shapeComparisonMethodID: String?
    @Published var cameraReset = 0
    @Published var syncCameras = true
    private var volumeGeneration = UUID()
    private var displayedVolumeKey: String?
    private var loadingVolumeKey: String?
    private let volumeCache: NSCache<NSString,CachedVolume> = {
        let cache = NSCache<NSString,CachedVolume>()
        cache.countLimit = 3
        cache.totalCostLimit = 128 * 1024 * 1024
        return cache
    }()
    var excludedCells: [Specimen] { dataset?.cells.filter { !$0.includedInAnalysis } ?? [] }
    var analysisSelected: Specimen? { selected.flatMap { $0.includedInAnalysis ? $0 : nil } }
    var allItems: [Specimen] { level == "Cells" ? dataset?.cells ?? [] : dataset?.organoids ?? [] }
    var selected: Specimen? { allItems.first { $0.id == selectedID } }
    var sample: Sample? { dataset?.samples.first { $0.id == selected?.sample } }
    var learnedMethods: [Method] { dataset?.methods.filter { $0.id != "traditional" && $0.family != "traditional" && $0.family != "validation" && (level == "Cells" || ($0.scope != "cells" && !["shape_hybrid95", "precomputed"].contains($0.metric))) } ?? [] }
    var groupMethod: String { activeTab == "methods" ? referenceMethod : method }
    var filtered: [Specimen] {
        allItems.filter { item in
            item.includedInAnalysis && (!qcOnly || item.qc) && (sampleFilter == "All stacks" || item.sample == sampleFilter) &&
            (splitFilter == "All splits" || item.split == splitFilter) &&
            (search.isEmpty || item.id.localizedCaseInsensitiveContains(search)) &&
            (clusterFilter < 0 || item.representations[groupMethod]?.cluster == clusterFilter) &&
            (!markedOnly || selection.contains(item.id))
        }
    }
    var measurementItems: [Specimen] {
        guard !selection.isEmpty else { return filtered }
        return filtered.filter { selection.contains($0.id) }
    }
    var selectedTableRows: Set<String> {
        tableSelection.intersection(Set(measurementItems.map(\.id)))
    }
    /// Plot marking drives the table's native multi-row highlight and the 3D focus.
    func markPlotItems(_ ids: Set<String>) {
        selection = ids.intersection(Set(filtered.map(\.id)))
        if selection.isEmpty { markedOnly = false }
        else if selectedID.map({ !selection.contains($0) }) ?? true {
            select(selection.sorted().first!)
        }
    }
    /// Inspecting one row keeps a brushed group available in the measurements list.
    func selectTableRows(_ ids: Set<String>) {
        let rows = ids.intersection(Set(measurementItems.map(\.id)))
        let added = rows.subtracting(tableSelection)
        tableSelection = rows
        if let focus = added.sorted().first ?? (rows.contains(selectedID ?? "") ? selectedID : rows.sorted().first) {
            select(focus)
        }
    }
    func select(_ id: String) {
        let changed = selectedID != id
        selectedID = id
        if selection.isEmpty { tableSelection = [id] }
        if changed, let selected, let sample, selected.label > 0,
           let coordinate = selected.metrics[["centroid_z_um", "centroid_y_um", "centroid_x_um"][axis]] {
            slice = min(1, max(0, coordinate / sample.extent[axis]))
        }
        loadVolume()
    }
    func changeLevel() {
        if !learnedMethods.contains(where: { $0.id == method }), let first = learnedMethods.first { method = first.id }
        selection = []; markedOnly = false; splitFilter = "All splits"; clusterFilter = -1
        select(filtered.first?.id ?? "")
    }
    func neighbors(method key: String, count: Int = 8) -> [Neighbor] {
        guard let selected = analysisSelected, let rep = selected.representations[key],
              let m = dataset?.methods.first(where: { $0.id == key }) else { return [] }
        return filtered.compactMap { item -> Neighbor? in
            guard item.id != selected.id, (!excludeSameStack || item.sample != selected.sample),
                  let r = item.representations[key] else { return nil }
            let d = representationDistance(rep.vector, r.vector, metric: m.metric)
            return d.isFinite ? Neighbor(item: item, distance: d) : nil
        }.sorted { $0.distance == $1.distance ? $0.id < $1.id : $0.distance < $1.distance }.prefix(count).map { $0 }
    }
    func openPanel() {
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false
        panel.message = "Choose an exported folder containing dataset.json"
        if panel.runModal() == .OK, let url = panel.url { load(url) }
    }
    func load(_ url: URL) {
        loading = true; error = nil
        volumeGeneration = UUID(); displayedVolumeKey = nil; loadingVolumeKey = nil
        volume = nil; volumeCache.removeAllObjects()
        Task {
            do {
                let data = try await Task.detached { () -> Dataset in
                    let d = try JSONDecoder().decode(Dataset.self, from: Data(contentsOf: url.appendingPathComponent("dataset.json")))
                    try d.validate(); return d
                }.value
                folder = url
                if data.cells.contains(where: { $0.inMainOrganoid == nil }) {
                    dataset = nil; loading = false
                    updateMembership()
                    return
                }
                dataset = data; analysisRevision = UUID(); loading = false
                UserDefaults.standard.set(url.path, forKey: "datasetPath")
                search = ""; sampleFilter = "All stacks"; selection = []; markedOnly = false
                clusterFilter = -1; splitFilter = "All splits"
                if !data.methods.contains(where: { $0.id == method }) { method = data.methods.last!.id }
                select(filtered.first?.id ?? "")
            } catch { self.error = error.localizedDescription; loading = false }
        }
    }
    func loadVolume() {
        guard let sample, let folder else {
            volumeGeneration = UUID(); loadingVolumeKey = nil; displayedVolumeKey = nil; volume = nil
            return
        }
        let key = folder.standardizedFileURL.path + "/" + sample.id
        if displayedVolumeKey == key || loadingVolumeKey == key { return }
        let generation = UUID(); volumeGeneration = generation
        volumeError = nil; loadingVolumeKey = nil
        if let cached = volumeCache.object(forKey:key as NSString) {
            displayedVolumeKey = key; volume = cached.volume
            return
        }
        displayedVolumeKey = nil; volume = nil; loadingVolumeKey = key
        Task {
            do {
                let v = try await Task.detached { try VolumeData(folder:folder.appendingPathComponent(sample.id),sample:sample) }.value
                if generation == volumeGeneration {
                    let entry = CachedVolume(v)
                    volumeCache.setObject(entry,forKey:key as NSString,cost:entry.cost)
                    loadingVolumeKey = nil; displayedVolumeKey = key; volume = v
                }
            } catch {
                if generation == volumeGeneration { loadingVolumeKey = nil; volumeError = error.localizedDescription }
            }
        }
    }
    /// Refit every descriptive analysis after a membership change, then publish atomically.
    func updateMembership(cellID: String? = nil, included: Bool? = nil) {
        guard !membershipBusy, let folder else { return }
        membershipBusy = true
        let root = URL(fileURLWithPath: Bundle.main.object(forInfoDictionaryKey: "ExplorerProjectPath") as? String ?? FileManager.default.currentDirectoryPath)
        let request = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".json")
        do {
            let overrides: [String: Any] = cellID.map { [$0: included.map { $0 as Any } ?? NSNull()] } ?? [:]
            try JSONSerialization.data(withJSONObject: overrides).write(to: request)
        } catch { self.error = error.localizedDescription; membershipBusy = false; return }
        Task {
            do {
                let updated = try await Task.detached { () -> Dataset in
                    defer { try? FileManager.default.removeItem(at: request) }
                    let log = folder.appendingPathComponent("membership-update.log")
                    FileManager.default.createFile(atPath: log.path, contents: nil)
                    let handle = try FileHandle(forWritingTo: log)
                    defer { try? handle.close() }
                    let process = Process()
                    process.executableURL = root.appendingPathComponent(".venv/bin/python")
                    process.arguments = [root.appendingPathComponent("scripts/organoid_membership.py").path, "--dataset", folder.path, "--overrides", request.path]
                    process.currentDirectoryURL = root
                    process.standardOutput = handle; process.standardError = handle
                    try process.run(); process.waitUntilExit()
                    guard process.terminationStatus == 0 else {
                        throw DataError.invalid("Could not update main-organoid membership. See " + log.path)
                    }
                    let data = try JSONDecoder().decode(Dataset.self, from: Data(contentsOf: folder.appendingPathComponent("dataset.json")))
                    try data.validate(); return data
                }.value
                guard self.folder == folder else { membershipBusy = false; return }
                dataset = updated; analysisRevision = UUID()
                let eligible = Set(updated.cells.filter(\.includedInAnalysis).map(\.id) + updated.organoids.map(\.id))
                selection.formIntersection(eligible); compared = nil; showCompare = false
                if selected == nil { select(filtered.first?.id ?? "") }
            } catch { self.error = error.localizedDescription }
            membershipBusy = false
        }
    }
    func exportSelection() {
        let panel = NSSavePanel(); panel.nameFieldStringValue = "organoid-selection.csv"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let rows = selection.isEmpty ? filtered : filtered.filter { selection.contains($0.id) }
        func quote(_ value: String) -> String { "\"" + value.replacingOccurrences(of: "\"", with: "\"\"") + "\"" }
        let csv = "id,stack,volume_um3,sphericity,elongation,qc,in_main_organoid\n" + rows.map {
            "\(quote($0.id)),\(quote($0.sample)),\($0.volume),\($0.sphericity),\($0.elongation),\($0.qc),\($0.includedInAnalysis)"
        }.joined(separator: "\n")
        do { try csv.write(to: url, atomically: true, encoding: .utf8) } catch { self.error = error.localizedDescription }
    }
}
let clusterColors: [Color] = [.teal, .purple, .orange, .blue, .pink, .green]
