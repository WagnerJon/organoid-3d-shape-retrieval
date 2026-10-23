import Foundation

@main struct DemoDatasetWorker {
    @MainActor static func main() async throws {
        let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let archive = root.appendingPathComponent("native/OrganoidExplorer/Resources/DemoShapes.atlasdemo")
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("atlas-demo-native-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: base) }
        let folder = try DemoDataset.install(archive: archive, base: base)
        let original = try Data(contentsOf: folder.appendingPathComponent("dataset.json"))
        let data = try JSONDecoder().decode(Dataset.self, from: original)
        try data.validate()
        guard data.syntheticDemo == true, data.cells.count == 72, data.samples.count == 12,
              data.methods.count == 36, data.cells.allSatisfy({ $0.name != nil && $0.representations.count == 36 }) else {
            throw DataError.invalid("Invalid demo coverage")
        }
        for sample in data.samples {
            guard sample.sourceImage.hasPrefix(folder.path), sample.sourceLabels.hasPrefix(folder.path),
                  FileManager.default.fileExists(atPath: sample.sourceImage), FileManager.default.fileExists(atPath: sample.sourceLabels) else {
                throw DataError.invalid("Demo depends on an external source path")
            }
            let volume = try VolumeData(folder: folder.appendingPathComponent(sample.id), sample: sample)
            guard Set(volume.labels) == Set(UInt16(0)...UInt16(6)) else { throw DataError.invalid("Invalid demo label preview") }
        }
        for cell in data.cells {
            _ = try MeshData(url: folder.appendingPathComponent(cell.sample).appendingPathComponent("\(cell.label).mesh"))
            for relative in cell.pointReconstructions?.values ?? Dictionary<String, String>().values {
                _ = try PointCloudData(url: folder.appendingPathComponent(relative))
            }
        }
        // Force a location with no Python environment or model files. Both
        // three-axis plots and default shape discovery must remain available.
        guard FileManager.default.changeCurrentDirectoryPath(base.path) else { throw DataError.invalid("Could not isolate offline test") }
        defer { FileManager.default.changeCurrentDirectoryPath(root.path) }
        for method in data.methods {
            let projection = try await ShapeAnalysis.run(folder: folder, method: method.id, level: "Cells", action: "projection")
            let groups = try await ShapeAnalysis.run(folder: folder, method: method.id, level: "Cells", action: "groups")
            guard projection.ids.count == 72, projection.points?.allSatisfy({ $0.count == 3 }) == true,
                  groups.labels?.count == 72, (groups.groups ?? []).reduce(0, { $0 + $1.members.count }) + (groups.unassigned ?? 0) == 72 else {
                throw DataError.invalid("Invalid offline demo analysis: " + method.id)
            }
        }
        let previousDataset = UserDefaults.standard.object(forKey: "datasetPath")
        defer { UserDefaults.standard.set(previousDataset, forKey: "datasetPath") }
        let store = ExplorerStore()
        store.unifiedPlotColorLevels = ["Cells"]
        store.unifiedPlotColorSettings["Cells"] = PlotColorSettings(feature: "volume_um3", palette: .viridis)
        store.load(folder)
        for _ in 0..<100 where store.loading { try await Task.sleep(nanoseconds: 100_000_000) }
        guard store.dataset?.syntheticDemo == true, store.error == nil, store.unifiedPlotColors,
              store.plotColors("traditional").feature == "demo_shape_class" else { throw DataError.invalid("Demo did not load with linked family colors") }
        store.markPlotItems(Set(data.cells.prefix(3).map(\.id)))
        guard store.measurementItems.count == 3, store.selectedTableRows.count == 3 else { throw DataError.invalid("Demo plot selection did not mark rows") }
        let again = try DemoDataset.install(archive: archive, base: base)
        guard again.standardizedFileURL.path == folder.standardizedFileURL.path,
              original == (try Data(contentsOf: folder.appendingPathComponent("dataset.json"))) else {
            throw DataError.invalid("Reopening the demo overwrote the local copy")
        }
        // Cached analyses must not silently serve a changed cohort.
        guard try DemoDataset.cachedAnalysis(folder: folder, method: "traditional", level: "Cells", action: "groups",
                                              ids: [data.cells[0].id], minimum: 10, samples: 5, selection: "eom", allowSingle: false) == nil else {
            throw DataError.invalid("Offline discovery ignored a cohort filter")
        }
        var regular = try JSONSerialization.jsonObject(with: original) as! [String: Any]
        regular["syntheticDemo"] = false
        let regularFolder = base.appendingPathComponent("regular-dataset")
        try FileManager.default.createDirectory(at: regularFolder, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: regular).write(to: regularFolder.appendingPathComponent("dataset.json"))
        for sample in data.samples {
            try FileManager.default.createSymbolicLink(at: regularFolder.appendingPathComponent(sample.id),
                                                       withDestinationURL: folder.appendingPathComponent(sample.id))
        }
        store.load(regularFolder)
        for _ in 0..<100 where store.loading { try await Task.sleep(nanoseconds: 100_000_000) }
        guard store.plotColors("traditional").feature == "volume_um3", store.plotColors("traditional").palette == .viridis else {
            throw DataError.invalid("Demo colors leaked into a user dataset")
        }
        var edited = try JSONSerialization.jsonObject(with: original) as! [String: Any]
        edited["title"] = "My edited demo"
        let editedBytes = try JSONSerialization.data(withJSONObject: edited)
        try editedBytes.write(to: folder.appendingPathComponent("dataset.json"))
        let reopened = try DemoDataset.install(archive: archive, base: base)
        guard reopened.standardizedFileURL.path == folder.standardizedFileURL.path,
              editedBytes == (try Data(contentsOf: folder.appendingPathComponent("dataset.json"))),
              try DemoDataset.cachedAnalysis(folder: folder, method: "traditional", level: "Cells", action: "projection",
                                             ids: nil, minimum: 10, samples: 5, selection: "eom", allowSingle: false) == nil else {
            throw DataError.invalid("Edited demo was overwritten or served stale analysis")
        }
        print("PASS: portable bundled install, 72 named objects, 36 methods, raw/labels/meshes/reconstructions, all offline three-axis plots and default groups, family colors, linked rows, reusable writable copy and cohort checks.")
    }
}
