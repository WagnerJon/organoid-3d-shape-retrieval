import Foundation

@main struct MissingMethodsWorker {
    @MainActor static func main() async throws {
        let folder = URL(fileURLWithPath: CommandLine.arguments[1])
        let source = try Data(contentsOf: folder.appendingPathComponent("dataset.json"))
        let store = ExplorerStore()
        store.dataset = try JSONDecoder().decode(Dataset.self, from: source)
        store.folder = folder; store.level = "Organoids"; store.search = "Keep this filter"
        let selected = store.dataset!.organoids.first!.id
        store.selectedID = selected
        let calculation = store.missingMethods
        var callback = false
        calculation.start(folder) { url in
            callback = true; store.reloadMethodResults(url)
        }
        store.updateMembership(cellID: store.dataset!.cells.first!.id, included: false)
        guard !store.membershipBusy, store.error?.contains("method calculation") == true else {
            throw DataError.invalid("Membership edits could race the method calculation")
        }
        store.error = nil
        for _ in 0..<600 where calculation.busy {
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        guard !calculation.busy, calculation.error == nil, callback,
              calculation.report?.missing == 0, calculation.report?.completed?.isEmpty == true,
              calculation.progress == 1 else {
            throw DataError.invalid("Native missing-method calculation failed: \(calculation.error ?? calculation.log)")
        }
        try await Task.sleep(nanoseconds: 100_000_000)
        guard store.level == "Organoids", store.selectedID == selected, store.search == "Keep this filter",
              source == (try Data(contentsOf: folder.appendingPathComponent("dataset.json"))) else {
            throw DataError.invalid("No-op calculation changed the dataset or viewer state")
        }
        // Exercise the actual inspection → selection → run flow on an isolated
        // three-cell fixture, leaving the original experiment untouched.
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("atlas-method-selection-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        var fixture = try JSONSerialization.jsonObject(with: source) as! [String: Any]
        let sample = store.dataset!.cells.first!.sample
        var cells = (fixture["cells"] as! [[String: Any]]).filter { $0["sample"] as? String == sample }
        cells = Array(cells.prefix(3))
        for index in cells.indices {
            var reps = cells[index]["representations"] as! [String: Any]
            reps.removeValue(forKey: "morph_size"); reps.removeValue(forKey: "morph_shape")
            cells[index]["representations"] = reps
        }
        fixture["cells"] = cells
        fixture["samples"] = (fixture["samples"] as! [[String: Any]]).filter { $0["id"] as? String == sample }
        fixture["organoids"] = (fixture["organoids"] as! [[String: Any]]).filter { $0["sample"] as? String == sample }
        fixture["methods"] = (fixture["methods"] as! [[String: Any]]).filter { !["morph_size", "morph_shape"].contains($0["id"] as? String ?? "") }
        let fixtureData = try JSONSerialization.data(withJSONObject: fixture)
        try fixtureData.write(to: temporary.appendingPathComponent("dataset.json"))
        let surfaces = temporary.appendingPathComponent(sample)
        try FileManager.default.createDirectory(at: surfaces, withIntermediateDirectories: true)
        for cell in cells {
            let name = "\(cell["label"] as! Int).mesh"
            try FileManager.default.copyItem(at: folder.appendingPathComponent(sample).appendingPathComponent(name), to: surfaces.appendingPathComponent(name))
        }
        store.dataset = try JSONDecoder().decode(Dataset.self, from: fixtureData)
        store.folder = temporary
        calculation.present(temporary)
        for _ in 0..<600 where calculation.inspecting { try await Task.sleep(nanoseconds: 100_000_000) }
        guard !calculation.inspecting, !calculation.busy, calculation.error == nil,
              calculation.selectedMethods == Set(["morph_size", "morph_shape"]), calculation.selectedCount == 2,
              fixtureData == (try Data(contentsOf: temporary.appendingPathComponent("dataset.json"))) else {
            throw DataError.invalid("Method inspection did not produce a read-only checklist: \(calculation.error ?? calculation.log)")
        }
        calculation.selectedMethods = []
        calculation.start(temporary, methods: []) { _ in }
        guard !calculation.busy, calculation.selectedCount == 0 else { throw DataError.invalid("Empty selection started work") }
        calculation.selectedMethods = ["morph_shape"]
        callback = false
        calculation.start(temporary, methods: calculation.selectedMethods.sorted()) { url in
            callback = true; store.reloadMethodResults(url)
        }
        for _ in 0..<600 where calculation.busy { try await Task.sleep(nanoseconds: 100_000_000) }
        try await Task.sleep(nanoseconds: 200_000_000)
        guard !calculation.busy, calculation.error == nil, callback,
              calculation.report?.completed == ["morph_shape"],
              store.dataset!.cells.allSatisfy({ $0.representations["morph_shape"] != nil && $0.representations["morph_size"] == nil }),
              store.level == "Organoids", store.search == "Keep this filter",
              source == (try Data(contentsOf: folder.appendingPathComponent("dataset.json"))) else {
            throw DataError.invalid("Selected-method run changed siblings or failed to reload: \(calculation.error ?? calculation.log)")
        }
        print("PASS: read-only checklist, individual selection, empty-selection guard, selected-only native calculation and reload, preserved filters, unchanged source experiment.")
    }
}
