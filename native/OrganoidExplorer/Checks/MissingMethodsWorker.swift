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
        print("PASS: native method calculation, already-complete coverage, automatic reload, retained organoid selection and filters, unchanged dataset.")
    }
}
