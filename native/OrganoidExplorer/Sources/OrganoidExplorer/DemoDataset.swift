import Foundation
import CryptoKit

enum DemoDataset {
    static func fingerprint(_ bytes: Data) -> String {
        SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    }

    /// Extract the trusted bundled archive into a writable, versioned copy.
    /// Neither exploration caches nor membership edits modify the app bundle.
    static func install(archive: URL, base: URL) throws -> URL {
        let fm = FileManager.default
        let identity = fingerprint(try Data(contentsOf: archive, options: .mappedIfSafe))
        let destination = base.appendingPathComponent("shape-gallery-" + identity.prefix(16), isDirectory: true)
        if fm.fileExists(atPath: destination.appendingPathComponent("dataset.json").path) { return destination }
        try fm.createDirectory(at: base, withIntermediateDirectories: true)
        let staging = base.appendingPathComponent("install-" + UUID().uuidString)
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: staging) }
        let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        process.arguments = ["-x", "-k", archive.path, staging.path]
        let pipe = Pipe(); process.standardOutput = pipe; process.standardError = pipe
        try process.run()
        let log = pipe.fileHandleForReading.readDataToEndOfFile(); process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw DataError.invalid("Could not open the bundled demo: " + String(decoding: log, as: UTF8.self))
        }
        let manifestURL = staging.appendingPathComponent("demo-manifest.json")
        guard let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as? [String: Any],
              let files = manifest["files"] as? [String: String] else {
            throw DataError.invalid("The bundled demo inventory is missing")
        }
        for (relative, expected) in files {
            guard !relative.hasPrefix("/"), !relative.split(separator: "/").contains("..") else {
                throw DataError.invalid("Invalid bundled demo path")
            }
            let bytes = try Data(contentsOf: staging.appendingPathComponent(relative), options: .mappedIfSafe)
            guard fingerprint(bytes) == expected else { throw DataError.invalid("Bundled demo asset is damaged: " + relative) }
        }
        let path = staging.appendingPathComponent("dataset.json")
        let bytes = try Data(contentsOf: path)
        let dataset = try JSONDecoder().decode(Dataset.self, from: bytes)
        try dataset.validate()
        guard dataset.syntheticDemo == true, dataset.cells.count == manifest["cells"] as? Int,
              dataset.methods.count == manifest["methods"] as? Int else { throw DataError.invalid("Invalid demo dataset") }
        // Relative paths make the shipped data portable; absolute source paths
        // keep existing segmentation tools working after extraction.
        var document = try JSONSerialization.jsonObject(with: bytes) as! [String: Any]
        var samples = document["samples"] as! [[String: Any]]
        for index in samples.indices {
            for key in ["sourceImage", "sourceLabels"] {
                guard let relative = samples[index][key] as? String, !relative.hasPrefix("/"),
                      !relative.split(separator: "/").contains("..") else { throw DataError.invalid("Invalid demo source path") }
                samples[index][key] = destination.appendingPathComponent(relative).path
            }
            let sample = dataset.samples[index]
            let runRelative = URL(fileURLWithPath: sample.sourceLabels).deletingLastPathComponent().lastPathComponent
            let configURL = staging.appendingPathComponent("sources/" + runRelative + "/run.json")
            var config = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as! [String: Any]
            config["source"] = samples[index]["sourceImage"]
            try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys]).write(to: configURL, options: .atomic)
        }
        document["samples"] = samples
        let installed = try JSONSerialization.data(withJSONObject: document, options: [.sortedKeys])
        try installed.write(to: path, options: .atomic)
        try Data(fingerprint(installed).utf8).write(to: staging.appendingPathComponent("demo-analysis-state.sha256"))
        do { try fm.moveItem(at: staging, to: destination) }
        catch {
            // A second app window may have installed the same bundle first.
            guard fm.fileExists(atPath: destination.appendingPathComponent("dataset.json").path) else { throw error }
        }
        return destination
    }

    static func cachedAnalysis(folder: URL, method: String, level: String, action: String,
                               ids: [String]?, minimum: Int, samples: Int, selection: String,
                               allowSingle: Bool) throws -> ShapeAnalysisResult? {
        let state = folder.appendingPathComponent("demo-analysis-state.sha256")
        guard let expected = try? String(contentsOf: state, encoding: .utf8),
              expected == fingerprint(try Data(contentsOf: folder.appendingPathComponent("dataset.json"))),
              ["projection", "groups"].contains(action), ["Cells", "Organoids"].contains(level),
              !method.contains("/"), !method.contains("..") else { return nil }
        if action == "groups" && (minimum != 10 || samples != 5 || selection != "eom" || allowSingle) { return nil }
        let cache = folder.appendingPathComponent("demo-analysis/\(level)/\(method).\(action).json")
        guard let bytes = try? Data(contentsOf: cache) else { return nil }
        let result = try JSONDecoder().decode(ShapeAnalysisResult.self, from: bytes)
        guard result.method == method, result.level == level, ids == nil || Set(ids!) == Set(result.ids) else { return nil }
        return result
    }
}

extension ExplorerStore {
    func openDemo() {
        guard !loading else { return }
        guard let archive = Bundle.main.url(forResource: "DemoShapes", withExtension: "atlasdemo") else {
            error = "This build does not contain the demo dataset. Rebuild the app with its bundled demo resource."
            return
        }
        loading = true; error = nil
        Task {
            do {
                let folder = try await Task.detached(priority: .userInitiated) {
                    let support = try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask,
                                                              appropriateFor: nil, create: true)
                    return try DemoDataset.install(archive: archive, base: support.appendingPathComponent("Organoid Atlas/Demo"))
                }.value
                load(folder)
            } catch { self.error = error.localizedDescription; loading = false }
        }
    }
}
