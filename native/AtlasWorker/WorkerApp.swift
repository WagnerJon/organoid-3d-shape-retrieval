import SwiftUI
import AppKit

@main struct AtlasWorkerApp: App {
    var body: some Scene {
        WindowGroup("Atlas Worker") {
            WorkerLocationView()
                .onAppear { NSApp.setActivationPolicy(.regular); NSApp.activate(ignoringOtherApps: true) }
        }.defaultSize(width: 720, height: 540)
    }
}

private struct WorkerLocationView: View {
    @State private var root: URL? = {
        let bundled = Bundle.main.bundleURL.deletingLastPathComponent()
        if FileManager.default.fileExists(atPath: bundled.appendingPathComponent("scripts/setup_atlas_worker.sh").path) { return bundled }
        if let saved = UserDefaults.standard.string(forKey: "workerPackageFolder"), FileManager.default.fileExists(atPath: URL(fileURLWithPath: saved).appendingPathComponent("scripts/setup_atlas_worker.sh").path) { return URL(fileURLWithPath: saved) }
        return nil
    }()
    @State private var error: String?
    var body: some View {
        Group {
            if let root { WorkerSession(root: root) }
            else {
                VStack(spacing: 16) {
                    Text("Locate your Worker folder").font(.title2.bold())
                    Text("Choose the extracted Atlas Worker for Mac folder containing the app, scripts and vendor folders.")
                    Button("Choose Worker folder…") {
                        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false
                        if panel.runModal() == .OK, let url = panel.url {
                            if FileManager.default.fileExists(atPath: url.appendingPathComponent("scripts/setup_atlas_worker.sh").path) {
                                UserDefaults.standard.set(url.path, forKey: "workerPackageFolder"); root = url
                            } else { error = "Choose the entire extracted Worker package, not just the application." }
                        }
                    }
                    if let error { Text(error).foregroundStyle(.red) }
                }.padding(30).frame(minWidth: 640, minHeight: 400)
            }
        }
    }
}

private struct WorkerSession: View {
    @StateObject private var workers: AtlasWorkers
    init(root: URL) { _workers = StateObject(wrappedValue: AtlasWorkers(root: root)) }
    var body: some View { AtlasWorkersView(workers: workers, workerOnly: true) }
}
