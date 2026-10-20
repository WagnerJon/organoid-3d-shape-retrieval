import Foundation

struct PansegSettings: Codable, Equatable {
    var modelSource = "zoo"
    var model = "generic_confocal_3D_unet"
    var modelID = ""
    var configPath = ""
    var weightsPath = ""
    var rescaleZ = 1.0, rescaleY = 1.0, rescaleX = 1.0
    var gaussianSigma = 0.0
    var medianRadius = 0
    var automaticPatch = true
    var patchZ = 80, patchY = 128, patchX = 128
    var haloMode = "tiled"
    var haloZ = 44, haloY = 44, haloX = 44
    var singleBatch = true
    var device = "auto"
    var gpu = true
    var boundaryChannel = 0
    var threshold = 0.5
    var minSize = 100
    var sigmaSeeds = 1.0, sigmaWeights = 2.0, alpha = 1.0
    var stacked = false
    var anisotropy = true
    var nonmax = false
    var blockwise = false
    var threads = 4
    var clustering = "multicut"
    var beta = 0.5
    var postMinSize = 100
    var linkage = "average"
    var foreground = 0.0

    func json() throws -> [String: Any] {
        try JSONSerialization.jsonObject(with: JSONEncoder().encode(self)) as! [String:Any]
    }
    static func fromJSON(_ json: [String: Any]) throws -> PansegSettings {
        var merged = try PansegSettings().json()
        for (key,value) in json where merged[key] != nil { merged[key] = value }
        if json["clustering"] == nil, let multicut = json["multicut"] as? Bool { merged["clustering"] = multicut ? "multicut" : "watershed"; merged["postMinSize"] = json["minSize"] ?? 100 }
        return try JSONDecoder().decode(PansegSettings.self, from: JSONSerialization.data(withJSONObject:merged))
    }
    var inferenceIdentity: String {
        var json = (try? self.json()) ?? [:]
        let keys: Set<String> = ["modelSource","model","modelID","configPath","weightsPath","rescaleZ","rescaleY","rescaleX","gaussianSigma","medianRadius","automaticPatch","patchZ","patchY","patchX","haloMode","haloZ","haloY","haloX","singleBatch","device","gpu","boundaryChannel"]
        json = json.filter { keys.contains($0.key) }
        let data = try? JSONSerialization.data(withJSONObject:json,options:.sortedKeys)
        return data.flatMap { String(data:$0,encoding:.utf8) } ?? ""
    }
    var summary: String { "\(modelSource == "zoo" ? model : modelSource == "bioimageio" ? modelID : URL(fileURLWithPath:configPath).lastPathComponent) · \(clustering) · threshold \(threshold) · seed σ \(sigmaSeeds)" }
}
struct PansegModel: Codable, Identifiable {
    let id, description: String
    let resolution: [Double]
}
struct PansegPreset: Codable, Identifiable {
    let id, name: String
    let settings: PansegSettings
}
struct SegPrediction: Codable {
    let key, hash: String
    let shape: [Int]
    let spacing: [Double]
    let device: String
}
