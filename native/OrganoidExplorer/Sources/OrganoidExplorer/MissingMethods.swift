import SwiftUI
import Darwin

struct MissingMethodGroup: Codable, Identifiable {
    let id: String
    let methods, names: [String]
    let available: Bool
    let reason: String
    var title: String {
        ["spatial": "Morphometry & spatial descriptors", "harmonics": "Spherical harmonics",
         "lite": "ShapeEmbedLite3D", "pointnet": "PointNet++", "point-mae": "Point-MAE",
         "allen": "Allen rotation-invariant encoder", "image": "Microscopy representations", "cajal": "CAJAL"][id] ?? id
    }
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
    @Published var inspecting = false
    @Published var selectedMethods: Set<String> = []
    private var process: Process?
    private var pending = ""
    private var token = UUID()
    let root: URL

    init(root: URL? = nil) {
        self.root = root ?? URL(fileURLWithPath: Bundle.main.object(forInfoDictionaryKey: "ExplorerProjectPath") as? String ?? FileManager.default.currentDirectoryPath)
    }
    var selectableMethods: Set<String> {
        Set(report?.groups.filter(\.available).flatMap(\.methods) ?? [])
    }
    var selectedCount: Int { selectedMethods.intersection(selectableMethods).count }
    func present(_ folder: URL) {
        showing = true
        guard !busy, !inspecting else { return }
        launch(folder, operation: "inspect", completion: { _ in })
    }
    func start(_ folder: URL, methods: [String]? = nil, completion: @escaping (URL) -> Void) {
        showing = true
        guard !busy, !inspecting else { return }
        guard methods?.isEmpty != true else { return }
        launch(folder, operation: "run", methods: methods, completion: completion)
    }
    private func launch(_ folder: URL, operation: String, methods: [String]? = nil, completion: @escaping (URL) -> Void) {
        let preview = operation == "inspect"
        do {
            let python = root.appendingPathComponent(".venv/bin/python")
            guard FileManager.default.isExecutableFile(atPath: python.path) else {
                throw DataError.invalid("The analysis environment is missing. Restore the project's Python environment to calculate representations.")
            }
            let destination = FileManager.default.temporaryDirectory.appendingPathComponent("atlas-missing-methods-\(UUID().uuidString).json")
            let p = Process(); p.executableURL = python; p.currentDirectoryURL = root
            p.arguments = ["-u", root.appendingPathComponent("scripts/atlas_missing_methods.py").path,
                           operation, "--dataset", folder.path, "--output", destination.path]
            for method in methods ?? [] { p.arguments?.append(contentsOf: ["--method", method]) }
            var environment = ProcessInfo.processInfo.environment
            environment["PYTHONUNBUFFERED"] = "1"; p.environment = environment
            let pipe = Pipe(); p.standardOutput = pipe; p.standardError = pipe
            let generation = UUID(); token = generation
            busy = !preview; inspecting = preview; cancelling = false; progress = 0
            stage = preview ? "Checking available methods…" : "Calculating selected methods…"
            error = nil; log = ""; pending = ""; report = nil; datasetFolder = folder
            if preview { selectedMethods = [] }
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
                    self.busy = false; self.inspecting = false; self.process = nil
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
                        if preview {
                            self.selectedMethods = self.selectableMethods
                            self.stage = report.missing == 0 ? "All transferable methods are already calculated" : "Choose the methods to calculate"
                        } else {
                            self.progress = 1
                            let count = report.completed?.count ?? 0
                            self.stage = count > 0 ? "Added \(count) methods for \(report.cells) cells" : report.missing == 0 ? "All transferable methods are already calculated" : "No selected methods could be calculated"
                        }
                    } else {
                        self.error = "Could not read the method availability report. See the processing log."
                    }
                    // Also reload after cancellation/failure: earlier families
                    // may already have been published successfully.
                    if !preview { completion(folder) }
                }
            }
            process = p
            try p.run()
        } catch {
            busy = false; inspecting = false; process = nil; self.error = error.localizedDescription
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
                calculation.present(folder)
            }
        } label: {
            Label(calculation.busy ? "Calculating methods…" : "Calculate missing methods", systemImage: "sparkles")
        }
        .disabled(store.folder == nil || store.loading || store.membershipBusy || store.dataset?.syntheticDemo == true)
        .help(store.dataset?.syntheticDemo == true ? "All 36 demo methods are already calculated and included in the app." : "Choose which missing representations to calculate for the loaded dataset.")
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
                if calculation.inspecting {
                    ProgressView().controlSize(.small)
                } else if !calculation.busy, calculation.report?.status == nil, let report = calculation.report {
                    selection(report)
                    HStack {
                        Text("\(calculation.selectedCount) of \(calculation.selectableMethods.count) available methods selected")
                            .font(.caption).foregroundStyle(.secondary)
                        Spacer()
                        Button("Calculate selected (\(calculation.selectedCount))") {
                            guard let folder = store.folder, folder == calculation.datasetFolder else { return }
                            let selected = calculation.selectedMethods.intersection(calculation.selectableMethods).sorted()
                            calculation.start(folder, methods: selected) { store.reloadMethodResults($0) }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(calculation.selectedCount == 0 || store.folder != calculation.datasetFolder || store.loading || store.membershipBusy)
                    }
                } else if calculation.busy {
                    ProgressView(value: calculation.progress)
                    HStack {
                        Text("One method family at a time · completed work is saved").font(.caption).foregroundStyle(.secondary)
                        Spacer()
                        Button(calculation.cancelling ? "Stopping…" : "Stop") { calculation.cancel() }.disabled(calculation.cancelling)
                    }
                }
                if let report = calculation.report {
                    let skipped = report.groups.filter { !$0.available }
                    if !skipped.isEmpty, report.status != nil {
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
            }.padding(22).frame(width: 720)
        }
    }
    private func selection(_ report: MissingMethodsReport) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Missing methods").font(.headline)
                Spacer()
                Button("Select all") { calculation.selectedMethods = calculation.selectableMethods }
                Button("Clear") { calculation.selectedMethods = [] }
            }
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 16) {
                    ForEach(report.groups) { group in
                        VStack(alignment: .leading, spacing: 6) {
                            Toggle(group.title, isOn: Binding(
                                get: { Set(group.methods).isSubset(of: calculation.selectedMethods) },
                                set: { checked in
                                    if checked { calculation.selectedMethods.formUnion(group.methods) }
                                    else { calculation.selectedMethods.subtract(group.methods) }
                                }))
                                .font(.headline).disabled(!group.available)
                            ForEach(Array(group.methods.enumerated()), id: \.element) { index, method in
                                Toggle(group.names.indices.contains(index) ? group.names[index] : method, isOn: Binding(
                                    get: { calculation.selectedMethods.contains(method) },
                                    set: { checked in
                                        if checked { calculation.selectedMethods.insert(method) }
                                        else { calculation.selectedMethods.remove(method) }
                                    }))
                                    .disabled(!group.available).padding(.leading, 20)
                            }
                            if !group.available {
                                Text(group.reason).font(.caption).foregroundStyle(.secondary).textSelection(.enabled).padding(.leading, 20)
                            }
                        }
                    }
                }.padding(10).frame(maxWidth: .infinity, alignment: .leading)
            }.frame(height: report.groups.isEmpty ? 0 : 280)
                .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
            Text("Only selected methods are added. Related methods may share preprocessing and cached encoder outputs.")
                .font(.caption).foregroundStyle(.secondary)
        }.toggleStyle(.checkbox)
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
