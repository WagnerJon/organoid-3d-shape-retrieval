import Foundation

struct ShapeMatch: Codable, Identifiable {
    let id: String
    let distance: Double
}
struct DiscoveredShapeGroup: Codable, Identifiable {
    let label: Int
    let members: [String]
    let representative: String
    let nearest: [ShapeMatch]
    let membershipStrength: Double
    var id: Int { label }
}
struct ShapeAnalysisResult: Codable {
    let ids: [String]
    let method, level, metric: String
    var points: [[Double]]?
    var variance: [Double]?
    var projection: String?
    var labels: [Int]?
    var groups: [DiscoveredShapeGroup]?
    var unassigned: Int?
    var strengths: [Double]?
    var selection: String?
    var allowSingle: Bool?
}

enum ShapeAnalysis {
    static func run(folder: URL, method: String, level: String, action: String,
                    ids: [String]? = nil, minimum: Int = 10, samples: Int = 5,
                    selection: String = "eom", allowSingle: Bool = false) async throws -> ShapeAnalysisResult {
        let root = URL(fileURLWithPath: Bundle.main.object(forInfoDictionaryKey: "ExplorerProjectPath") as? String ?? FileManager.default.currentDirectoryPath)
        return try await Task.detached(priority: .userInitiated) {
            let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("atlas-shape-" + UUID().uuidString)
            try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: temporary) }
            var request: [String: Any] = ["dataset": folder.path, "method": method, "level": level,
                                          "action": action, "minimum": minimum, "samples": samples,
                                          "selection": selection, "allowSingle": allowSingle]
            if let ids { request["ids"] = ids }
            let input = temporary.appendingPathComponent("request.json"), output = temporary.appendingPathComponent("result.json")
            try JSONSerialization.data(withJSONObject: request).write(to: input)
            let log = temporary.appendingPathComponent("analysis.log")
            FileManager.default.createFile(atPath: log.path, contents: nil)
            let handle = try FileHandle(forWritingTo: log)
            defer { try? handle.close() }
            let process = Process()
            process.executableURL = root.appendingPathComponent(".venv/bin/python")
            process.arguments = [root.appendingPathComponent("scripts/atlas_interactive.py").path,
                                 "--request", input.path, "--output", output.path]
            var environment = ProcessInfo.processInfo.environment
            environment["OPENBLAS_NUM_THREADS"] = "1"; environment["OMP_NUM_THREADS"] = "1"
            process.environment = environment; process.currentDirectoryURL = root
            process.standardOutput = handle; process.standardError = handle
            try process.run(); process.waitUntilExit()
            guard process.terminationStatus == 0 else {
                let details = (try? String(contentsOf: log, encoding: .utf8)) ?? "Analysis failed"
                throw DataError.invalid(String(details.suffix(1500)))
            }
            return try JSONDecoder().decode(ShapeAnalysisResult.self, from: Data(contentsOf: output))
        }.value
    }
}
