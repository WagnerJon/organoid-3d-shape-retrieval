import SwiftUI
import Darwin

struct MissingMethodGroup: Codable, Identifiable {
    let id: String
    let methods, names: [String]
    let available: Bool
    let reason: String
}
struct MissingMethodsReport: Codable {
    let dataset: String
    let cells, organoids, runnable, missing: Int
    let groups: [MissingMethodGroup]
    var completed: [String]?
    var failed: [MissingMethodFailure]?
    var status: String?
}
struct MissingMethodFailure: Codable {
    let group, reason, details: String
}

@MainActor final class MissingMethodsStore: ObservableObject {
    @Published var busy = false
    @Published var cancelling = false
    @Published var showing = false
    @Published var progress = 0.0
    @Published var stage = ""
    @Published var log = ""
    @Published var error: String?
    @Published var report: MissingMethodsReport?
    @Published var datasetFolder: URL?
    private var process: Process?
    private var pending = ""
    private var token = UUID()
    let root: URL

    init(root: URL? = nil) {
        self.root = root ?? URL(fileURLWithPath: Bundle.main.object(forInfoDictionaryKey: "ExplorerProjectPath") as? String ?? FileManager.default.currentDirectoryPath)
    }
    func start(_ folder: URL, completion: @escaping (URL) -> Void) {
        showing = true
        guard !busy else { return }
        do {
            let python = root.appendingPathComponent(".venv/bin/python")
            guard FileManager.default.isExecutableFile(atPath: python.path) else {
                throw DataError.invalid("The analysis environment is missing. Restore the project's Python environment to calculate representations.")
            }
            let destination = FileManager.default.temporaryDirectory.appendingPathComponent("atlas-missing-methods-\(UUID().uuidString).json")
            let p = Process(); p.executableURL = python; p.currentDirectoryURL = root
            p.arguments = ["-u", root.appendingPathComponent("scripts/atlas_missing_methods.py").path,
                           "run", "--dataset", folder.path, "--output", destination.path]
            var environment = ProcessInfo.processInfo.environment
            environment["PYTHONUNBUFFERED"] = "1"; p.environment = environment
            let pipe = Pipe(); p.standardOutput = pipe; p.standardError = pipe
            let generation = UUID(); token = generation
            busy = true; cancelling = false; progress = 0; stage = "Checking missing methods…"
            error = nil; log = ""; pending = ""; report = nil; datasetFolder = folder
            pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
                let bytes = handle.availableData
                if !bytes.isEmpty {
                    Task { @MainActor in
                        guard let self, self.token == generation else { return }
                        self.consume(String(decoding: bytes, as: UTF8.self))
                    }
                }
            }
            p.terminationHandler = { [weak self] finished in
                pipe.fileHandleForReading.readabilityHandler = nil
                Task { @MainActor in
                    guard let self, self.token == generation else { return }
                    self.busy = false; self.process = nil
                    if let bytes = try? Data(contentsOf: destination) {
                        self.report = try? JSONDecoder().decode(MissingMethodsReport.self, from: bytes)
                    }
                    try? FileManager.default.removeItem(at: destination)
                    if self.cancelling {
                        self.stage = "Stopped. Completed methods are saved; run again to continue."
                        self.cancelling = false
                    } else if finished.terminationStatus != 0 {
                        self.stage = "Calculation stopped"
                        self.error = String(self.log.suffix(2500))
                    } else if let report = self.report {
                        self.progress = 1
                        let count = report.completed?.count ?? 0
                        self.stage = count > 0 ? "Added \(count) methods for \(report.cells) cells" : report.missing == 0 ? "All transferable methods are already calculated" : "No additional methods could be calculated"
                    }
                    // Also reload after cancellation/failure: earlier families
                    // may already have been published successfully.
                    completion(folder)
                }
            }
            process = p
            try p.run()
        } catch {
            busy = false; process = nil; self.error = error.localizedDescription
        }
    }
    private func consume(_ text: String) {
        log = String((log + text).suffix(24000)); pending += text
        while let range = pending.rangeOfCharacter(from: .newlines) {
            let line = String(pending[..<range.lowerBound]); pending.removeSubrange(..<range.upperBound)
            let prefix = "@methods-progress "
            if line.hasPrefix(prefix), let bytes = line.dropFirst(prefix.count).data(using: .utf8),
               let value = try? JSONSerialization.jsonObject(with: bytes) as? [String: Any] {
                stage = value["stage"] as? String ?? stage
                if let fraction = value["fraction"] as? Double, fraction.isFinite {
                    progress = max(progress, min(1, max(0, fraction)))
                }
            }
        }
        pending = String(pending.suffix(24000))
    }
    func cancel() {
        guard busy, !cancelling, let process else { return }
        cancelling = true; stage = "Stopping after saving completed work…"
        // The Python supervisor terminates its current child before exiting.
        process.terminate()
    }
}

struct MissingMethodsControls: View {
    @EnvironmentObject var store: ExplorerStore
    @ObservedObject var calculation: MissingMethodsStore
    var body: some View {
        Button {
            if calculation.busy { calculation.showing = true }
            else if let folder = store.folder {
                calculation.start(folder) { store.reloadMethodResults($0) }
            }
        } label: {
            Label(calculation.busy ? "Calculating methods…" : "Calculate missing methods", systemImage: "sparkles")
        }
        .disabled(store.folder == nil || store.loading || store.membershipBusy || store.dataset?.syntheticDemo == true)
        .help(store.dataset?.syntheticDemo == true ? "All 36 demo methods are already calculated and included in the app." : "Calculate missing representations for all cells in the loaded dataset using available frozen models. Existing vectors are retained.")
        .sheet(isPresented: $calculation.showing) {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Text("Calculate missing methods").font(.title2.bold())
                    Spacer()
                    Button("Close") { calculation.showing = false }
                }
                Text(calculation.datasetFolder?.lastPathComponent ?? "Loaded dataset").foregroundStyle(.secondary)
                Text("Calculates representations for the full dataset, regardless of plot filters. Existing vectors, segmentation and measurements are retained.")
                    .font(.callout)
                Text(calculation.stage).font(.headline)
                if calculation.busy {
                    ProgressView(value: calculation.progress)
                    HStack {
                        Text("One method family at a time · completed work is saved").font(.caption).foregroundStyle(.secondary)
                        Spacer()
                        Button(calculation.cancelling ? "Stopping…" : "Stop") { calculation.cancel() }.disabled(calculation.cancelling)
                    }
                }
                if let report = calculation.report {
                    let skipped = report.groups.filter { !$0.available }
                    if !skipped.isEmpty {
                        Text("Unavailable methods").font(.headline)
                        ForEach(skipped) { group in
                            Text(group.names.joined(separator: ", ") + "\n" + group.reason).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    ForEach(report.failed ?? [], id: \.group) { failure in
                        Text("\(failure.group): calculation failed. Details are in \(failure.reason)").font(.caption).foregroundStyle(.orange).textSelection(.enabled)
                    }
                }
                if let error = calculation.error {
                    Text(error).font(.caption).foregroundStyle(.orange).textSelection(.enabled).lineLimit(8)
                }
                DisclosureGroup("Processing log") {
                    ScrollView { Text(calculation.log).font(.caption.monospaced()).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }.frame(height: 160)
                }
                Text("Uses the existing frozen encoders and descriptors. No training is started. Original cohort-specific validation references are not recalculated.")
                    .font(.caption).foregroundStyle(.secondary)
            }.padding(22).frame(width: 650)
        }
    }
}

extension ExplorerStore {
    func reloadMethodResults(_ url: URL) {
        guard folder == url else { return }
        Task {
            do {
                let updated = try await Task.detached { () -> Dataset in
                    let data = try JSONDecoder().decode(Dataset.self, from: Data(contentsOf: url.appendingPathComponent("dataset.json")))
                    try data.validate(); return data
                }.value
                guard folder == url else { return }
                dataset = updated; analysisRevision = UUID()
            } catch { self.error = error.localizedDescription }
        }
    }
}
