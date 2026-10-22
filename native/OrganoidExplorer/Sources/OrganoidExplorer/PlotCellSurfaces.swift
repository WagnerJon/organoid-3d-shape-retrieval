import SwiftUI
import SceneKit
import Metal

/// A small display mesh derived from the original surface, never a learned reconstruction.
struct PlotCellSurface {
    let vertices: [SIMD3<Float>]
    let normals: [SIMD3<Float>]
    let faces: [UInt32]

    init(mesh: MeshData) throws {
        guard !mesh.vertices.isEmpty, mesh.vertices.allSatisfy({ $0.x.isFinite && $0.y.isFinite && $0.z.isFinite }) else {
            throw DataError.invalid("Invalid cell surface")
        }
        var minimum = mesh.vertices[0], maximum = minimum
        for v in mesh.vertices { minimum = simd_min(minimum, v); maximum = simd_max(maximum, v) }
        let center = (minimum + maximum) / 2
        let extent = maximum - minimum, span = max(extent.x, extent.y, extent.z)
        guard span > 0 else { throw DataError.invalid("Empty cell surface extent") }
        let positions = mesh.vertices.map { ($0 - center) / span }
        // Glyphs occupy tens of pixels. Vertex clustering bounds GPU work even for
        // hundreds of large surfaces, while the inspector retains the full mesh.
        var output = (positions, mesh.normals, mesh.faces)
        if mesh.faces.count / 3 > 2_500 {
            for resolution in [32, 24, 20, 16, 12, 8] {
                let candidate = Self.cluster(positions, mesh.normals, mesh.faces, resolution: resolution)
                if candidate.2.isEmpty { break }
                output = candidate
                if output.2.count / 3 <= 2_500 { break }
            }
        }
        vertices = output.0; normals = output.1; faces = output.2
    }

    private static func cluster(_ vertices: [SIMD3<Float>], _ normals: [SIMD3<Float>], _ faces: [UInt32],
                                resolution: Int) -> ([SIMD3<Float>], [SIMD3<Float>], [UInt32]) {
        var buckets: [Int: Int] = [:], positions: [SIMD3<Float>] = [], sums: [SIMD3<Float>] = []
        var counts: [Float] = [], mapping: [UInt32] = []
        for (i, v) in vertices.enumerated() {
            let q = (v + SIMD3<Float>(repeating: 0.5)) * Float(resolution)
            let x = min(resolution-1, max(0, Int(q.x))), y = min(resolution-1, max(0, Int(q.y)))
            let z = min(resolution-1, max(0, Int(q.z))), key = (z * resolution + y) * resolution + x
            let index: Int
            if let found = buckets[key] { index = found }
            else { index = positions.count; buckets[key] = index; positions.append(.zero); sums.append(.zero); counts.append(0) }
            positions[index] += v; sums[index] += normals[i]; counts[index] += 1; mapping.append(UInt32(index))
        }
        for i in positions.indices {
            positions[i] /= counts[i]
            let length = simd_length(sums[i]); sums[i] = length > 1e-8 ? sums[i] / length : SIMD3<Float>(0, 0, 1)
        }
        var indices: [UInt32] = []
        for i in stride(from: 0, to: faces.count, by: 3) {
            let a = mapping[Int(faces[i])], b = mapping[Int(faces[i+1])], c = mapping[Int(faces[i+2])]
            if a != b && a != c && b != c { indices.append(contentsOf: [a, b, c]) }
        }
        return (positions, sums, indices)
    }

    func geometry() -> SCNGeometry {
        let element = SCNGeometryElement(data: faces.withUnsafeBufferPointer { Data(buffer: $0) },
            primitiveType: .triangles, primitiveCount: faces.count/3, bytesPerIndex: 4)
        return SCNGeometry(sources: [.init(vertices: vertices.map { SCNVector3($0.x, $0.y, $0.z) }),
                                     .init(normals: normals.map { SCNVector3($0.x, $0.y, $0.z) })], elements: [element])
    }
}

private final class PlotCellThumbnailRenderer {
    let renderer: SCNRenderer
    let scene = SCNScene()
    let surface = SCNNode()
    init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw DataError.invalid("3D preview rendering is unavailable") }
        renderer = SCNRenderer(device: device, options: nil)
        scene.background.contents = NSColor.clear
        let camera = SCNNode(); camera.camera = SCNCamera()
        camera.camera?.usesOrthographicProjection = true; camera.camera?.orthographicScale = 0.85
        camera.camera?.zNear = 0.01; camera.camera?.zFar = 20
        camera.position = SCNVector3(2, 1.4, 3); camera.look(at: SCNVector3Zero)
        scene.rootNode.addChildNode(camera); scene.rootNode.addChildNode(surface)
        let light = SCNNode(); light.light = SCNLight(); light.light?.type = .directional
        light.light?.intensity = 900; light.eulerAngles = SCNVector3(-0.6, -0.5, 0)
        scene.rootNode.addChildNode(light)
        let ambient = SCNNode(); ambient.light = SCNLight(); ambient.light?.type = .ambient; ambient.light?.intensity = 450
        scene.rootNode.addChildNode(ambient)
        renderer.scene = scene; renderer.pointOfView = camera
    }
    func image(_ mesh: PlotCellSurface) -> NSImage {
        let geometry = mesh.geometry(), material = SCNMaterial()
        material.diffuse.contents = NSColor.white; material.lightingModel = .lambert; material.isDoubleSided = true
        geometry.materials = [material]; surface.geometry = geometry
        return renderer.snapshot(atTime: 0, with: CGSize(width: 80, height: 80), antialiasingMode: .multisampling4X)
    }
}

enum PlotCellAssetMode: Equatable { case thumbnails, surfaces }
private struct PlotCellAsset {
    let id: String
    var surface: PlotCellSurface?
    var image: NSImage?
    var error: String?
}

/// Deliberately instance-owned: activating one plot cannot schedule loads in another.
@MainActor final class PlotCellAssets: ObservableObject {
    @Published private(set) var thumbnails: [String: NSImage] = [:]
    @Published private(set) var surfaces: [String: PlotCellSurface] = [:]
    @Published private(set) var completed = 0
    @Published private(set) var total = 0
    @Published private(set) var failures: [String: String] = [:]
    @Published private(set) var loading = false
    private var generation = UUID()
    private var worker: Task<Void, Never>?
    private var scope = ""

    static func requestKey(enabled: Bool, folder: URL?, items: [Specimen]) -> String {
        guard enabled, let folder else { return "off" }
        return folder.standardizedFileURL.path + "|" + items.map { "\($0.id):\($0.sample):\($0.label)" }.joined(separator: "|")
    }

    func clear() {
        worker?.cancel(); worker = nil; generation = UUID(); scope = ""
        thumbnails = [:]; surfaces = [:]; failures = [:]; completed = 0; total = 0; loading = false
    }

    func load(folder: URL, items: [Specimen], mode: PlotCellAssetMode) async {
        worker?.cancel(); generation = UUID(); let token = generation
        let newScope = folder.standardizedFileURL.path + (mode == .thumbnails ? "|images" : "|meshes")
        let ids = Set(items.map(\.id))
        if newScope != scope { thumbnails = [:]; surfaces = [:]; failures = [:] }
        else {
            thumbnails = thumbnails.filter { ids.contains($0.key) }; surfaces = surfaces.filter { ids.contains($0.key) }
            failures = failures.filter { ids.contains($0.key) }
        }
        scope = newScope; total = items.count
        let cached = Set(thumbnails.keys).union(surfaces.keys).union(failures.keys)
        completed = cached.count
        let entries = items.filter { !cached.contains($0.id) }.map {
            ($0.id, folder.appendingPathComponent($0.sample).appendingPathComponent("\($0.label).mesh"))
        }
        guard !entries.isEmpty else { loading = false; return }
        loading = true
        let job = Task.detached(priority: .utility) { [weak self] in
            do {
                let renderer = mode == .thumbnails ? try PlotCellThumbnailRenderer() : nil
                var batch: [PlotCellAsset] = []
                for (id, url) in entries {
                    try Task.checkCancellation()
                    let asset: PlotCellAsset = autoreleasepool {
                        do {
                            let mesh = try PlotCellSurface(mesh: MeshData(url: url))
                            return PlotCellAsset(id: id, surface: mode == .surfaces ? mesh : nil,
                                                 image: renderer?.image(mesh))
                        } catch { return PlotCellAsset(id: id, error: error.localizedDescription) }
                    }
                    batch.append(asset)
                    if batch.count == 8 {
                        try Task.checkCancellation(); await self?.receive(batch, generation: token); batch = []
                    }
                }
                try Task.checkCancellation()
                if !batch.isEmpty { await self?.receive(batch, generation: token) }
                await self?.finish(generation: token)
            } catch is CancellationError { }
            catch { await self?.unavailable(error.localizedDescription, generation: token) }
        }
        worker = job
        await withTaskCancellationHandler(operation: { await job.value }, onCancel: { job.cancel() })
    }
    private func receive(_ batch: [PlotCellAsset], generation token: UUID) {
        guard generation == token else { return }
        for item in batch {
            if let image = item.image { thumbnails[item.id] = image }
            if let surface = item.surface { surfaces[item.id] = surface }
            if let error = item.error { failures[item.id] = error }
        }
        completed += batch.count
    }
    private func finish(generation token: UUID) { if generation == token { loading = false; worker = nil } }
    private func unavailable(_ message: String, generation token: UUID) {
        guard generation == token else { return }; failures["Rendering"] = message; loading = false; worker = nil
    }
}

struct PlotCellLoadStatus: View {
    @ObservedObject var assets: PlotCellAssets
    var compact = false
    var body: some View {
        HStack(spacing: 6) {
            if assets.loading {
                ProgressView(value: Double(assets.completed), total: Double(max(assets.total, 1))).frame(width: 70)
                Text(compact ? "\(assets.completed)/\(assets.total) shapes" : "Loading shapes \(assets.completed)/\(assets.total)")
            } else { Text(compact ? "\(assets.thumbnails.count + assets.surfaces.count) cell shapes" : "\(assets.thumbnails.count + assets.surfaces.count) cell shapes · display size normalized") }
            if !assets.failures.isEmpty {
                Text("\(assets.failures.count) unavailable").foregroundStyle(.orange)
                    .help(assets.failures.values.sorted().first ?? "")
            }
            Spacer(minLength: 0)
        }.font(.caption2).foregroundStyle(.secondary).help("Surface previews from original cell reconstructions, normalized to the same display size.")
    }
}
