import SwiftUI
import AppKit

struct AtlasWorkerEndpoint: Codable, Identifiable, Equatable {
    var id, name, host, fingerprint: String
    var port: Int
}
struct AtlasExecution: Codable {
    var useLocal = true
    var workers: [AtlasWorkerEndpoint] = []
    var splitStacks = false
    var planesPerBatch = 16
    init(useLocal: Bool = true, workers: [AtlasWorkerEndpoint] = [], splitStacks: Bool = false, planesPerBatch: Int = 16) {
        self.useLocal = useLocal; self.workers = workers; self.splitStacks = splitStacks; self.planesPerBatch = planesPerBatch
    }
    enum CodingKeys: String, CodingKey { case useLocal, workers, splitStacks, planesPerBatch }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        useLocal = try c.decodeIfPresent(Bool.self, forKey: .useLocal) ?? true
        workers = try c.decodeIfPresent([AtlasWorkerEndpoint].self, forKey: .workers) ?? []
        splitStacks = try c.decodeIfPresent(Bool.self, forKey: .splitStacks) ?? false
        planesPerBatch = try c.decodeIfPresent(Int.self, forKey: .planesPerBatch) ?? 16
    }
}
private struct WorkerReady: Decodable {
    let id, name, fingerprint, code: String
    let port: Int
}

@MainActor final class AtlasWorkers: NSObject, ObservableObject, @preconcurrency NetServiceBrowserDelegate, @preconcurrency NetServiceDelegate {
    @Published var nearby: [AtlasWorkerEndpoint] = []
    @Published var paired: [AtlasWorkerEndpoint] = []
    @Published var enabled: Set<String> = []
    @Published var useLocal = true
    @Published var splitStacks = false
    @Published var planesPerBatch = 16
    @Published var browsing = false
    @Published var sharing = false
    @Published var starting = false
    @Published var busy = false
    @Published var code = ""
    @Published var fingerprint = ""
    @Published var message = "Find and pair another Mac to share segmentation jobs."
    @Published var log = ""
    @Published var pairingEndpoint: AtlasWorkerEndpoint?
    @Published var pairingCode = ""
    private let root: URL
    private var browser: NetServiceBrowser?
    private var services: [NetService] = []
    private var publishedService: NetService?
    private var server: Process?
    private var pending = ""
    private var localID = ""
    var execution: AtlasExecution? { enabled.isEmpty && !splitStacks ? nil : AtlasExecution(useLocal: useLocal, workers: paired.filter { enabled.contains($0.id) }, splitStacks: splitStacks, planesPerBatch: planesPerBatch) }
    var runtimeInstalled: Bool { FileManager.default.isExecutableFile(atPath: root.appendingPathComponent(".venv/bin/python").path) }
    private var saved: URL { root.appendingPathComponent(".cache/atlas-workers/endpoints.json") }
    init(root: URL? = nil) {
        self.root = root ?? URL(fileURLWithPath: Bundle.main.object(forInfoDictionaryKey: "ExplorerProjectPath") as? String ?? FileManager.default.currentDirectoryPath)
        super.init()
        NotificationCenter.default.addObserver(self, selector: #selector(willTerminate), name: NSApplication.willTerminateNotification, object: nil)
        if let data = try? Data(contentsOf: saved), let values = try? JSONDecoder().decode([AtlasWorkerEndpoint].self, from: data) { paired = values }
    }
    func discover() {
        browser?.stop(); services = []; nearby = []; browsing = true
        let browser = NetServiceBrowser(); browser.delegate = self; self.browser = browser
        browser.searchForServices(ofType: "_atlas-worker._tcp.", inDomain: "local.")
        message = "Looking for Atlas Workers on your local network…"
    }
    func stopDiscovery() { browser?.stop(); browser = nil; browsing = false }
    func netServiceBrowser(_ browser: NetServiceBrowser, didFind service: NetService, moreComing: Bool) {
        service.delegate = self; services.append(service); service.resolve(withTimeout: 5)
    }
    func netServiceBrowser(_ browser: NetServiceBrowser, didRemove service: NetService, moreComing: Bool) {
        if let data = service.txtRecordData() {
            let txt = NetService.dictionary(fromTXTRecord: data)
            if let id = txt["id"].flatMap({ String(data: $0, encoding: .utf8) }) { nearby.removeAll { $0.id == id } }
        }
        services.removeAll { $0 === service }
    }
    func netServiceBrowser(_ browser: NetServiceBrowser, didNotSearch errorDict: [String: NSNumber]) {
        browsing = false; message = "Discovery failed. Allow Atlas local-network access in System Settings and check both Macs are on the same network."
    }
    func netService(_ sender: NetService, didNotPublish errorDict: [String: NSNumber]) { message = "Worker started, but Bonjour could not advertise it. Check local-network permission and restart sharing." }
    func netServiceDidResolveAddress(_ sender: NetService) {
        guard let data = sender.txtRecordData(), let host = sender.hostName else { return }
        let txt = NetService.dictionary(fromTXTRecord: data)
        func value(_ key: String) -> String? { txt[key].flatMap { String(data: $0, encoding: .utf8) } }
        guard let id = value("id"), id != localID, let fp = value("fp"), fp.count == 64, value("protocol") == "1" else { return }
        let endpoint = AtlasWorkerEndpoint(id: id, name: sender.name, host: host, fingerprint: fp, port: sender.port)
        nearby.removeAll { $0.id == id }; nearby.append(endpoint)
        if let index = paired.firstIndex(where: { $0.id == id && $0.fingerprint == fp }) { paired[index] = endpoint; persist() }
        message = "Found \(nearby.count) nearby worker(s). Select a worker and pair it once."
    }
    func restore(_ execution: AtlasExecution?) {
        useLocal = execution?.useLocal ?? true; enabled = []
        splitStacks = execution?.splitStacks ?? false; planesPerBatch = execution?.planesPerBatch ?? 16
        for endpoint in execution?.workers ?? [] {
            if !paired.contains(where: { $0.id == endpoint.id && $0.fingerprint == endpoint.fingerprint }) {
                paired.removeAll { $0.id == endpoint.id }; paired.append(endpoint)
            }
            enabled.insert(endpoint.id)
        }
        persist()
        if !enabled.isEmpty { message = "Saved worker selection restored. Open discovery to refresh addresses; pair again if this Mac has not trusted the worker yet." }
    }
    func beginPair(_ endpoint: AtlasWorkerEndpoint) {
        pairingEndpoint = endpoint; pairingCode = ""
    }
    func pair() {
        guard let endpoint = pairingEndpoint else { return }
        request("pair", value: ["endpoint": endpointJSON(endpoint), "code": pairingCode]) { [weak self] _ in
            guard let self else { return }
            self.paired.removeAll { $0.id == endpoint.id }; self.paired.append(endpoint); self.persist()
            self.enabled.insert(endpoint.id); self.pairingEndpoint = nil; self.pairingCode = ""
            self.message = "Paired with \(endpoint.name). It is now available for dataset preparation."
        }
    }
    func remove(_ endpoint: AtlasWorkerEndpoint) {
        request("forget", value: ["endpoint": endpointJSON(endpoint)]) { [weak self] _ in
            guard let self else { return }
            self.paired.removeAll { $0.id == endpoint.id }; self.enabled.remove(endpoint.id); self.persist()
        }
    }
    func fingerprintLabel(_ value: String) -> String { String(value.prefix(16)).uppercased() }
    private func endpointJSON(_ endpoint: AtlasWorkerEndpoint) -> Any {
        (try? JSONSerialization.jsonObject(with: JSONEncoder().encode(endpoint))) ?? [:]
    }
    private func persist() {
        try? FileManager.default.createDirectory(at: saved.deletingLastPathComponent(), withIntermediateDirectories: true)
        if let data = try? JSONEncoder().encode(paired) { try? data.write(to: saved, options: .atomic) }
    }
    private func request(_ operation: String, value: [String: Any], done: @escaping ([String: Any]) -> Void) {
        guard !busy else { return }; busy = true
        do {
            let req = FileManager.default.temporaryDirectory.appendingPathComponent("atlas-worker-\(UUID().uuidString).json")
            let out = req.appendingPathExtension("result")
            try JSONSerialization.data(withJSONObject: value).write(to: req)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: req.path)
            let p = Process(); p.executableURL = root.appendingPathComponent(".venv/bin/python"); p.currentDirectoryURL = root
            p.arguments = [root.appendingPathComponent("scripts/atlas_worker_client.py").path, operation, "--request", req.path, "--output", out.path]
            let pipe = Pipe(); p.standardOutput = pipe; p.standardError = pipe
            p.terminationHandler = { [weak self] p in
                let output = pipe.fileHandleForReading.readDataToEndOfFile()
                Task { @MainActor in
                    guard let self else { return }; self.busy = false
                    defer { try? FileManager.default.removeItem(at: req); try? FileManager.default.removeItem(at: out) }
                    if p.terminationStatus == 0, let data = try? Data(contentsOf: out), let result = try? JSONSerialization.jsonObject(with: data) as? [String: Any] { done(result) }
                    else { self.message = String(decoding: output, as: UTF8.self).suffix(1200).description }
                }
            }
            try p.run()
        } catch { busy = false; message = error.localizedDescription }
    }
    func startSharing() {
        guard server == nil, !starting else { return }
        guard runtimeInstalled else { message = "Set up the worker runtime first."; return }
        do {
            starting = true; pending = ""; log = ""
            let p = Process(); p.executableURL = root.appendingPathComponent(".venv/bin/python"); p.currentDirectoryURL = root
            p.arguments = ["-u", root.appendingPathComponent("scripts/atlas_worker.py").path, "serve", "--state", root.appendingPathComponent(".cache/atlas-worker").path]
            let pipe = Pipe(); p.standardOutput = pipe; p.standardError = pipe
            pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
                let bytes = handle.availableData
                if !bytes.isEmpty { Task { @MainActor in self?.consume(String(decoding: bytes, as: UTF8.self)) } }
            }
            p.terminationHandler = { [weak self] _ in
                pipe.fileHandleForReading.readabilityHandler = nil
                Task { @MainActor in
                    self?.publishedService?.stop(); self?.publishedService = nil; self?.server = nil
                    self?.starting = false; self?.sharing = false; self?.code = ""
                    self?.message = "Worker stopped. Jobs retain their completed checkpoints."
                }
            }
            server = p; try p.run()
            message = "Starting worker…"
        } catch { server = nil; starting = false; message = error.localizedDescription }
    }
    private func consume(_ text: String) {
        log = String((log+text).suffix(16000)); pending += text
        while let end = pending.firstIndex(of: "\n") {
            let line = String(pending[..<end]); pending.removeSubrange(...end)
            if line.hasPrefix("@worker-ready "), let data = line.dropFirst(14).data(using: .utf8), let ready = try? JSONDecoder().decode(WorkerReady.self, from: data) {
                starting = false; sharing = true; code = ready.code; fingerprint = ready.fingerprint; localID = ready.id
                let service = NetService(domain: "local.", type: "_atlas-worker._tcp.", name: ready.name, port: Int32(ready.port))
                service.setTXTRecord(NetService.data(fromTXTRecord: ["id": Data(ready.id.utf8), "fp": Data(ready.fingerprint.utf8), "protocol": Data("1".utf8)]))
                service.delegate = self; service.publish(); publishedService = service
                message = "Accepting jobs · one stack at a time. Keep this Mac awake and Atlas Worker open."
            }
        }
    }
    func stopSharing() { publishedService?.stop(); publishedService = nil; server?.terminate() }
    func installRuntime() {
        guard !busy, !sharing else { return }
        busy = true; log = ""; message = "Installing Python and processing packages. First setup requires internet and several GB of disk space."
        do {
            let p = Process(); p.executableURL = URL(fileURLWithPath: "/bin/zsh"); p.arguments = [root.appendingPathComponent("scripts/setup_atlas_worker.sh").path]
            let pipe = Pipe(); p.standardOutput = pipe; p.standardError = pipe
            pipe.fileHandleForReading.readabilityHandler = { [weak self] h in let data = h.availableData; if !data.isEmpty { Task { @MainActor in self?.log = String(((self?.log ?? "")+String(decoding: data, as: UTF8.self)).suffix(16000)) } } }
            p.terminationHandler = { [weak self] p in pipe.fileHandleForReading.readabilityHandler = nil; Task { @MainActor in self?.busy = false; self?.message = p.terminationStatus == 0 ? "Runtime ready. Press Accept jobs to start sharing." : "Setup failed; see the worker log." } }
            try p.run()
        } catch { busy = false; message = error.localizedDescription }
    }
    @objc private func willTerminate() { shutdown() }
    func shutdown() { stopDiscovery(); stopSharing() }
}

struct AtlasWorkersView: View {
    @ObservedObject var workers: AtlasWorkers
    var workerOnly = false
    var close: (() -> Void)?
    @State private var showLog = false
    var body: some View {
        ScrollView {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text(workerOnly ? "Atlas Worker" : "Nearby Macs").font(.title2.bold())
                Spacer()
                if let close { Button("Done", action: close) }
            }
            Text(workers.message).font(.callout).textSelection(.enabled)
            if !workerOnly {
                GroupBox("Find processing workers") {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Button("Find nearby workers") { workers.discover() }
                            if workers.browsing { ProgressView().controlSize(.small); Button("Stop searching") { workers.stopDiscovery() } }
                        }
                        ForEach(workers.nearby) { endpoint in
                            HStack {
                                Text(endpoint.name); Spacer()
                                Text(workers.fingerprintLabel(endpoint.fingerprint)).font(.caption.monospaced())
                                Button(workers.paired.contains(where: { $0.id == endpoint.id && $0.fingerprint == endpoint.fingerprint }) ? "Pair again" : "Pair…") { workers.beginPair(endpoint) }
                            }
                        }
                        if let endpoint = workers.pairingEndpoint {
                            Divider()
                            Text("Compare fingerprint \(workers.fingerprintLabel(endpoint.fingerprint)) with the one displayed on \(endpoint.name). If it matches, enter that worker's pairing code.").font(.caption)
                            HStack {
                                TextField("6-digit pairing code", text: $workers.pairingCode).frame(width: 150)
                                Button("Pair worker") { workers.pair() }.disabled(workers.pairingCode.count != 6 || workers.busy)
                                Button("Cancel") { workers.pairingEndpoint = nil }
                            }
                        }
                    }.padding(6)
                }
                GroupBox("Use for this preparation") {
                    VStack(alignment: .leading, spacing: 8) {
                        Toggle("This Mac", isOn: $workers.useLocal)
                        ForEach(workers.paired) { endpoint in
                            HStack {
                                Toggle(endpoint.name, isOn: Binding(get: { workers.enabled.contains(endpoint.id) }, set: { if $0 { workers.enabled.insert(endpoint.id) } else { workers.enabled.remove(endpoint.id) } }))
                                Spacer(); Button("Remove") { workers.remove(endpoint) }
                            }
                        }
                        Text("One stack per Mac. Raw files are cached on the worker; results return to this Mac. Both Macs must stay awake. RAM is not pooled.").font(.caption).foregroundStyle(.secondary)
                    }.padding(6)
                }
            }
            GroupBox("Share this Mac") {
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        if !workers.runtimeInstalled { Button("Set up worker runtime…") { workers.installRuntime() }.disabled(workers.busy) }
                        Button(workers.sharing || workers.starting ? "Stop accepting jobs" : "Accept jobs") { if workers.sharing || workers.starting { workers.stopSharing() } else { workers.startSharing() } }.disabled(workers.busy)
                        if workers.starting || workers.busy { ProgressView().controlSize(.small) }
                    }
                    if workers.sharing {
                        Text("Pairing code: \(workers.code)").font(.title.monospaced())
                        Text("Fingerprint: \(workers.fingerprintLabel(workers.fingerprint))").font(.body.monospaced()).textSelection(.enabled)
                        Text("Code expires after 10 minutes or one pairing. Restart sharing for another code. Only pair Macs you control.").font(.caption).foregroundStyle(.secondary)
                    }
                    Text("On the other Mac, open Atlas → New Dataset → Nearby Macs, find this worker and pair it. Stopping sharing cancels its active job and retains checkpoints.").font(.caption).foregroundStyle(.secondary)
                }.padding(6)
            }
            DisclosureGroup("Worker log", isExpanded: $showLog) { ScrollView { Text(workers.log).font(.system(size: 10, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }.frame(height: 130) }
            Spacer(minLength: 0)
        }.padding(22)
        }.frame(minWidth: 640, idealWidth: 720, minHeight: 500)
            .onAppear { if !workerOnly && !workers.browsing { workers.discover() } }
    }
}

struct AtlasWorkerSelection: View {
    @ObservedObject var workers: AtlasWorkers
    @Binding var showWorkers: Bool
    var allowPlaneBatches = true
    var body: some View {
        GroupBox("Processing Macs") {
          VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(workers.enabled.isEmpty ? "This Mac" : "\(workers.useLocal ? "This Mac + " : "")\(workers.enabled.count) paired worker(s)")
                    Text(workers.splitStacks ? "One inference batch runs at a time on each selected Mac." : "Whole stacks run one at a time on each selected Mac.").font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Button("Nearby Macs…") { showWorkers = true }
            }
            Toggle("Split large stacks into distributed batches", isOn: $workers.splitStacks).disabled(!allowPlaneBatches)
            if !allowPlaneBatches { Text("Native Cellpose 3D runs each complete stack on one selected Mac. Plane batches and streamed u-Segment3D linking are unavailable in this mode.").font(.caption).foregroundStyle(.secondary) }
            if workers.splitStacks {
                Picker("Planes per batch", selection: $workers.planesPerBatch) {
                    Text("8").tag(8); Text("16 (recommended)").tag(16); Text("32").tag(32); Text("64").tag(64)
                }.frame(maxWidth: 320)
                Text("Cellpose plane batches are shared across the selected Macs. Cells are linked once across the complete volume, so batch edges do not cut cells. Image preparation and final 3D linking still run on this Mac and need enough local memory. Imported labels skip inference.").font(.caption).foregroundStyle(.secondary)
            }
          }.padding(6)
        }
    }
}
