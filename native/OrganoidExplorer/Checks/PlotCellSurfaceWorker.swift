import Foundation
import AppKit
import SceneKit

@main struct PlotCellSurfaceChecks {
    @MainActor static func main() async throws {
        func require(_ condition: @autoclosure () -> Bool, _ message: String) {
            if !condition() { fatalError(message) }
        }
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("atlas-plot-surfaces-" + UUID().uuidString)
        let sample = folder.appendingPathComponent("sample")
        try FileManager.default.createDirectory(at: sample, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        var data = Data()
        let header: [UInt32] = [4, 4]
        let vertices: [Float] = [-2,0,0, 2,0,0, 0,3,0, 0,0,2]
        let normals: [Float] = [0,0,1, 0,0,1, 0,0,1, 0,0,1]
        let faces: [UInt32] = [0,2,1, 0,1,3, 1,2,3, 2,0,3]
        header.withUnsafeBytes { data.append(contentsOf: $0) }; vertices.withUnsafeBytes { data.append(contentsOf: $0) }
        normals.withUnsafeBytes { data.append(contentsOf: $0) }; faces.withUnsafeBytes { data.append(contentsOf: $0) }
        try data.write(to: sample.appendingPathComponent("1.mesh"))
        func cell(_ id: String, label: Int = 1) -> Specimen {
            Specimen(id:id, sample:"sample", split:"test", flags:"", label:label, qc:true,
                     metrics:[:], representations:[:], cellCount:nil)
        }
        let mesh = try PlotCellSurface(mesh: MeshData(url: sample.appendingPathComponent("1.mesh")))
        require(mesh.faces == faces, "Small original surfaces retain every triangle")
        let span = (0..<3).map { axis in mesh.vertices.map { $0[axis] }.max()! - mesh.vertices.map { $0[axis] }.min()! }.max()!
        require(abs(span - 1) < 1e-6, "Equal display size is independent of physical units")

        let first = PlotCellAssets(), second = PlotCellAssets()
        require(PlotCellAssets.requestKey(enabled:false, folder:folder, items:[cell("a")]) == "off", "Point mode does not schedule a mesh request")
        await first.load(folder:folder, items:[cell("a")], mode:.surfaces)
        require(first.surfaces.count == 1 && first.completed == 1 && !first.loading, "Active plot loads its surface")
        require(second.surfaces.isEmpty && second.thumbnails.isEmpty && second.total == 0, "Other plot has no loaded or scheduled assets")
        await first.load(folder:folder, items:[cell("missing", label:2)], mode:.surfaces)
        require(first.surfaces.isEmpty && first.failures["missing"] != nil && !first.loading, "Filtering removes hidden meshes and missing files remain recoverable")
        first.clear()
        let many = (0..<600).map { cell("old:\($0)") }
        let old = Task { @MainActor in await first.load(folder:folder, items:many, mode:.surfaces) }
        while !first.loading { await Task.yield() }
        first.clear()
        await first.load(folder:folder, items:[cell("new")], mode:.surfaces)
        await old.value
        require(Set(first.surfaces.keys) == ["new"] && first.completed == 1, "Cancelled loads cannot publish into a replacement plot request")
        first.clear()
        require(first.surfaces.isEmpty && first.thumbnails.isEmpty && first.total == 0, "Switching off releases this plot cache")

        let datasetFolder = URL(fileURLWithPath: CommandLine.arguments[1])
        let dataset = try JSONDecoder().decode(Dataset.self, from: Data(contentsOf:datasetFolder.appendingPathComponent("dataset.json")))
        let real = Array(dataset.cells.filter { $0.qc && $0.includedInAnalysis }.prefix(8))
        await first.load(folder:datasetFolder, items:real, mode:.surfaces)
        require(first.surfaces.count == real.count && first.failures.isEmpty, "Real reconstruction previews load")
        for surface in first.surfaces.values {
            require(surface.faces.count/3 <= 2500, "Preview geometry stays bounded for real cell meshes")
            require(surface.faces.allSatisfy { Int($0) < surface.vertices.count }, "Simplified triangle references remain valid")
        }
        first.clear()
        await first.load(folder:datasetFolder, items:Array(real.prefix(2)), mode:.thumbnails)
        require(first.thumbnails.count == 2 && first.surfaces.isEmpty && first.failures.isEmpty, "2D plots keep only rendered images, not full GPU meshes")
        let image = first.thumbnails[real[0].id]!
        var rect = NSRect(x:0,y:0,width:80,height:80)
        let cg = image.cgImage(forProposedRect:&rect, context:nil, hints:nil)!
        let bitmap = NSBitmapImageRep(cgImage:cg)
        var ink = 0, clear = 0
        for y in 0..<bitmap.pixelsHigh { for x in 0..<bitmap.pixelsWide {
            let alpha = bitmap.colorAt(x:x,y:y)?.alphaComponent ?? 0
            if alpha > 0.1 { ink += 1 } else { clear += 1 }
        } }
        require(ink > 50 && clear > 50, "Surface thumbnail has visible geometry and a transparent background")
        let output = datasetFolder.deletingLastPathComponent().appendingPathComponent("plot-surface-check.png")
        // The check artifact stays in the writable workspace, not the source dataset.
        if let destination = ProcessInfo.processInfo.environment["PLOT_SURFACE_CHECK_IMAGE"] {
            try bitmap.representation(using:.png,properties:[:])!.write(to:URL(fileURLWithPath:destination))
        }
        _ = output
        first.clear()
        print("PASS: plot-local loading; point mode; filter pruning; cancellation and replacement; bounded real geometry; transparent surface rendering; cache release.")
    }
}
