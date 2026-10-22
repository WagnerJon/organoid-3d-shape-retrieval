import SwiftUI

struct OMEZarrSelection: Codable, Equatable {
    let time, channel, level: Int
    var label: String?
    let fingerprint: String
    let origin: [Double]?
    let timestamp: Double?
}
struct OMEZarrLevel: Decodable, Identifiable {
    let index: Int
    let path, dtype: String
    let shape, shapeZYX: [Int]
    let spacing, origin: [Double]?
    let chunkBytes: Int64
    var id: Int { index }
    var summary: String { "Level \(index) · \(shapeZYX.map(String.init).joined(separator: " × "))" }
}
struct OMEZarrLabels: Decodable, Identifiable {
    let name, path: String
    let levels: [OMEZarrLevel]
    let axes: [String]
    let timepoints: Int
    let instanceEligible: Bool
    var id: String { name }
}
struct OMEZarrImage: Decodable, Identifiable {
    let path, name, fingerprint: String
    let axes, channels: [String]
    let levels: [OMEZarrLevel]
    let timepoints: Int
    let timestamps: [Double]
    let labels: [OMEZarrLabels]
    let sourceStat: [Int64]
    var id: String { path }
}
struct OMEZarrCatalog: Decodable, Identifiable {
    let images: [OMEZarrImage]
    let ignored: [String]
    var id: String { images.first?.path ?? "OME-Zarr" }
}

struct OMEZarrImportView: View {
    let catalog: OMEZarrCatalog
    let add: (OMEZarrImage, Int, Int, Int, Int, Int, String?) -> Void
    let close: () -> Void
    @State private var imageID = ""
    @State private var channel = 0
    @State private var level = 0
    @State private var first = 0
    @State private var last = 0
    @State private var step = 1
    @State private var label = ""
    var image: OMEZarrImage? { catalog.images.first { $0.path == imageID } ?? catalog.images.first }
    var grid: OMEZarrLevel? { image?.levels.first { $0.index == level } }
    var count: Int { last >= first ? (last-first)/max(1,step)+1 : 0 }
    var valid: Bool { image != nil && count > 0 && grid?.shapeZYX.allSatisfy { $0 >= 2 } == true && (grid?.chunkBytes ?? Int64.max) <= 256*1024*1024 }
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Import OME-Zarr").font(.title2.bold())
                Spacer(); Button("Cancel", action: close).keyboardShortcut(.cancelAction)
            }
            if let image, let grid {
                Picker("Image", selection: $imageID) {
                    ForEach(catalog.images) { Text($0.name).tag($0.path) }
                }
                Text("Axes: \(image.axes.joined(separator: ", ")) · \(image.timepoints) time points · \(image.channels.count) channels")
                    .font(.caption).foregroundStyle(.secondary)
                Text(image.path).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                Picker("Channel", selection: $channel) {
                    ForEach(image.channels.indices, id: \.self) { Text("\($0) · \(image.channels[$0])").tag($0) }
                }
                Picker("Resolution", selection: $level) {
                    ForEach(image.levels) { Text($0.summary).tag($0.index) }
                }
                if let spacing = grid.spacing {
                    Text("Voxel spacing Z/Y/X: \(spacing.map { String(format: "%.4g", $0) }.joined(separator: " / ")) µm. Lower resolution levels have larger voxels and less detail.")
                        .font(.caption).foregroundStyle(.secondary)
                } else { Text("Physical calibration is missing or unsupported. Voxel spacing will default to 1 × 1 × 1 µm; edit it after adding the volumes if known.").font(.caption).foregroundStyle(.orange) }
                GroupBox("Time points · zero-based indices") {
                    VStack(alignment: .leading, spacing: 8) {
                        Stepper("First: \(first)", value: $first, in: 0...max(0,image.timepoints-1))
                            .onChange(of: first) { if last < first { last = first } }
                        Stepper("Last: \(last)", value: $last, in: first...max(first,image.timepoints-1))
                        Stepper("Every \(step) frame(s)", value: $step, in: 1...max(1,image.timepoints))
                        if image.timestamps.indices.contains(first) {
                            Text("First selected frame: \(Date(timeIntervalSince1970: image.timestamps[first]).formatted(date:.abbreviated,time:.standard))")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }.padding(6)
                }
                Picker("Segmentation", selection: $label) {
                    Text("Run Cellpose on the selected channel").tag("")
                    ForEach(image.labels.filter(\.instanceEligible)) { Text("Use existing \($0.name) instances").tag($0.name) }
                }
                if !label.isEmpty {
                    Text("Existing labels must match this level’s spatial grid. Labels may be sparse and IDs are frame-local; they do not establish cell identity across time.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if image.labels.contains(where: { !$0.instanceEligible }) {
                    Text("Lumen/epithelium semantic masks are not offered as individual-cell instances.").font(.caption).foregroundStyle(.secondary)
                }
                if !valid { Text("Choose a 3D level with decoded chunks no larger than 256 MiB and a valid frame range.").font(.caption).foregroundStyle(.orange) }
                Text("\(count) volume(s) will be added. Each frame is analyzed independently. Only the selected channel, frames and level are copied into calibrated 3D stacks inside your results folder when preparation starts. The full 5D movie is not loaded into memory. GEFF tracking and lineage browsing are not imported.")
                    .font(.caption).foregroundStyle(.secondary)
                if !catalog.ignored.isEmpty {
                    Text("Skipped: " + catalog.ignored.joined(separator: "; ")).font(.caption2).foregroundStyle(.secondary)
                }
                HStack {
                    Spacer()
                    Button("Add \(count) volume(s)") { add(image, channel, level, first, last, step, label.isEmpty ? nil : label) }
                        .buttonStyle(.borderedProminent).disabled(!valid).keyboardShortcut(.defaultAction)
                }
            }
        }.padding(24).frame(width: 650)
        .onAppear { imageID = catalog.images.first?.path ?? "" }
        .onChange(of: imageID) { channel = 0; level = 0; first = 0; last = 0; step = 1; label = "" }
    }
}
