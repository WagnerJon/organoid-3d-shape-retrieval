import Foundation

@main struct InteractiveShapeWorker {
    @MainActor static func main() async throws {
        let folder = URL(fileURLWithPath: CommandLine.arguments[1])
        let source = try Data(contentsOf: folder.appendingPathComponent("dataset.json"))
        let dataset = try JSONDecoder().decode(Dataset.self, from: source)
        let method = "shapeembed3d_invariant_raw"
        let cells = dataset.cells.filter { $0.includedInAnalysis && $0.qc && $0.representations[method] != nil }
        let projection = try await ShapeAnalysis.run(folder: folder, method: method, level: "Cells", action: "projection")
        guard let points = projection.points, projection.projection == "pca", projection.variance?.count == 3,
              points.count == projection.ids.count, points.allSatisfy({ $0.count == 3 && $0.allSatisfy(\.isFinite) }) else {
            throw DataError.invalid("Invalid native 3-PC analysis")
        }
        let discovered = try await ShapeAnalysis.run(folder: folder, method: method, level: "Cells", action: "groups", ids: cells.map(\.id))
        guard let labels = discovered.labels, labels.count == discovered.ids.count,
              Set(discovered.ids) == Set(cells.map(\.id)),
              (discovered.groups ?? []).reduce(0, { $0 + $1.members.count }) + (discovered.unassigned ?? 0) == cells.count else {
            throw DataError.invalid("Invalid native discovery coverage")
        }
        let cache = try await ShapeAnalysis.run(folder: folder, method: method, level: "Cells", action: "groups", ids: cells.map(\.id))
        guard cache.labels == discovered.labels else { throw DataError.invalid("Cached group labels changed") }
        guard discovered.selection == "eom", discovered.allowSingle == false else {
            throw DataError.invalid("Discovery default still allows the broad parent group")
        }
        let fine = try await ShapeAnalysis.run(folder: folder, method: method, level: "Cells", action: "groups",
                                               ids: cells.map(\.id), selection: "leaf")
        guard fine.selection == "leaf", fine.allowSingle == false,
              Set(fine.ids) == Set(discovered.ids) else { throw DataError.invalid("Fine grouping settings or cohort lost") }
        let distanceMap = try await ShapeAnalysis.run(folder: folder, method: "cajal_gw_validation", level: "Cells", action: "projection")
        guard distanceMap.ids.count == 48, distanceMap.projection == "mds" else { throw DataError.invalid("Wrong distance-map scope") }
        let store = ExplorerStore(); store.dataset = dataset; store.folder = folder
        if let group = discovered.groups?.first {
            store.markPlotItems(Set(group.members))
            guard store.selection == Set(group.members), store.measurementItems.count == group.members.count else {
                throw DataError.invalid("Discovered group failed to mark linked rows")
            }
        }
        guard source == (try Data(contentsOf: folder.appendingPathComponent("dataset.json"))) else {
            throw DataError.invalid("Interactive analysis modified the source dataset")
        }
        print("PASS: native Python worker, cached three-PC / three-axis MDS results, discovery coverage, group-to-table marking, unchanged source data.")
        print("Discovery: \(discovered.groups?.count ?? 0) groups / \(discovered.unassigned ?? 0) unassigned / \(cells.count) cells")
    }
}
