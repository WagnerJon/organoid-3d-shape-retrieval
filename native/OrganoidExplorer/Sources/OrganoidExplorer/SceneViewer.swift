import SwiftUI
import SceneKit

struct MeshData {
    let vertices: [SIMD3<Float>]
    let normals: [SIMD3<Float>]
    let faces: [UInt32]
    init(url: URL) throws {
        let data = try Data(contentsOf: url)
        guard data.count >= 8 else { throw DataError.invalid("Truncated mesh") }
        let counts: [UInt32] = data.prefix(8).withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
        let n = Int(counts[0]), f = Int(counts[1])
        guard n > 0, n < 10_000_000, f < 20_000_000, data.count == 8+n*24+f*12 else { throw DataError.invalid("Invalid mesh length") }
        let values: [Float] = data.subdata(in: 8..<(8+n*24)).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        vertices = (0..<n).map { SIMD3(values[$0*3], values[$0*3+1], values[$0*3+2]) }
        normals = (0..<n).map { SIMD3(values[n*3+$0*3], values[n*3+$0*3+1], values[n*3+$0*3+2]) }
        faces = data.suffix(f*12).withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
        guard faces.allSatisfy({ Int($0) < n }) else { throw DataError.invalid("Invalid triangle index") }
    }
    func geometry() -> SCNGeometry {
        let positions = vertices.map { SCNVector3($0.x, $0.y, $0.z) }
        let ns = normals.map { SCNVector3($0.x, $0.y, $0.z) }
        let indices = faces.withUnsafeBufferPointer { Data(buffer: $0) }
        let element = SCNGeometryElement(data: indices, primitiveType: .triangles, primitiveCount: faces.count/3, bytesPerIndex: 4)
        return SCNGeometry(sources: [.init(vertices: positions), .init(normals: ns)], elements: [element])
    }
}
@MainActor final class CameraLink: ObservableObject {
    var views: [UUID: SCNView] = [:]
    var enabled = true
    func propagate(from id: UUID) {
        guard enabled, let source = views[id]?.pointOfView else { return }
        for (key, view) in views where key != id {
            view.pointOfView?.transform = source.transform
            view.pointOfView?.camera?.orthographicScale = source.camera?.orthographicScale ?? 70
        }
    }
}
final class PickingSceneView: SCNView {
    var picked: ((Int) -> Void)?
    var acceptHit: ((SCNHitTestResult) -> Bool)?
    var cameraChanged: (() -> Void)?
    private var down: NSPoint = .zero
    override func mouseDown(with event: NSEvent) { down = event.locationInWindow; super.mouseDown(with: event) }
    override func mouseDragged(with event: NSEvent) { super.mouseDragged(with: event); cameraChanged?() }
    override func mouseUp(with event: NSEvent) {
        super.mouseUp(with: event)
        if hypot(event.locationInWindow.x-down.x, event.locationInWindow.y-down.y) < 4 {
            let p = convert(event.locationInWindow, from: nil)
            if let hit = hitTest(p).first(where: { acceptHit?($0) ?? true }), let name = hit.node.name, let label = Int(name) { picked?(label) }
        }
        cameraChanged?()
    }
    override func scrollWheel(with event: NSEvent) { super.scrollWheel(with: event); cameraChanged?() }
}
struct ReconstructionAppearance: Equatable {
    let selected: Int
    let isolated, hideExcluded, centered: Bool
    let excluded: Set<Int>
    let extent: [Double]
}
struct ReconstructionClip: Equatable {
    let enabled: Bool
    let axis: Int
    let position: Double
}

struct ReconstructionView: NSViewRepresentable {
    let folder: URL
    let sample: Sample
    let cells: [Specimen]
    let selectedLabel: Int
    var isolate = false
    var hideExcluded = false
    var clipping = false
    var axis = 0
    var slice = 0.5
    var reset = 0
    var centered = false
    var cameraSpan: Double? = nil
    var pointCloudURL: URL? = nil
    var link: CameraLink? = nil
    var pick: (Int) -> Void = { _ in }
    var onError: (String) -> Void = { _ in }
    var appearance: ReconstructionAppearance {
        ReconstructionAppearance(selected:selectedLabel, isolated:isolate, hideExcluded:hideExcluded, centered:centered,
                                 excluded:Set(cells.filter { !$0.includedInAnalysis }.map(\.label)), extent:sample.extent)
    }
    var clipState: ReconstructionClip {
        ReconstructionClip(enabled:clipping, axis:clipping ? axis : 0, position:clipping ? sample.extent[axis]*slice : 0)
    }
    func makeCoordinator() -> Coordinator { Coordinator() }
    func makeNSView(context: Context) -> PickingSceneView {
        let view = PickingSceneView()
        view.backgroundColor = NSColor(calibratedRed: 0.045, green: 0.065, blue: 0.09, alpha: 1)
        view.allowsCameraControl = true; view.autoenablesDefaultLighting = true
        view.antialiasingMode = .multisampling4X
        view.defaultCameraController.inertiaEnabled = false
        view.scene = SCNScene()
        context.coordinator.view = view
        if let link { link.views[context.coordinator.id] = view }
        let id = context.coordinator.id
        view.cameraChanged = { [weak link] in link?.propagate(from: id) }
        return view
    }
    func updateNSView(_ view: PickingSceneView, context: Context) {
        let c = context.coordinator
        c.current = self; view.picked = pick
        view.acceptHit = { hit in
            guard clipping else { return true }
            let p = hit.localCoordinates
            let coordinate = axis == 0 ? p.z : (axis == 1 ? p.y : p.x)
            return Double(coordinate) <= sample.extent[axis] * slice
        }
        let key = folder.path + sample.id + (pointCloudURL?.path ?? "")
        if c.key != key {
            c.key = key
            c.load(view: view, config: self)
        } else { c.updateMaterials(config: self) }
        if c.reset != reset || c.lastSpan != cameraSpan { c.reset = reset; c.lastSpan = cameraSpan; c.frame(config: self) }
    }
    static func dismantleNSView(_ view: PickingSceneView, coordinator: Coordinator) {
        coordinator.current?.link?.views.removeValue(forKey: coordinator.id)
    }
    @MainActor final class Coordinator {
        let id = UUID()
        var key = ""
        var reset = 0
        var lastSpan: Double?
        weak var view: PickingSceneView?
        var current: ReconstructionView?
        var nodes: [Int: SCNNode] = [:]
        private(set) var appliedAppearance: ReconstructionAppearance?
        private(set) var appliedClip: ReconstructionClip?
        func load(view: PickingSceneView, config: ReconstructionView) {
            nodes = [:]; appliedAppearance = nil; appliedClip = nil; view.scene = SCNScene()
            let loadingKey = key
            let urls: [(Int,URL,Bool)] = config.cells.map { cell in
                let isPointCloud = cell.label == config.selectedLabel && config.pointCloudURL != nil
                let url: URL
                if isPointCloud { url = config.pointCloudURL! }
                else { url = config.folder.appendingPathComponent(config.sample.id).appendingPathComponent("\(cell.label).mesh") }
                return (cell.label,url,isPointCloud)
            }
            Task {
                do {
                    let meshes = try await Task.detached { () -> [(Int,ReconstructionGeometry)] in
                        try urls.map { entry in
                            if entry.2 { return (entry.0,.points(try PointCloudData(url:entry.1))) }
                            return (entry.0,.mesh(try MeshData(url:entry.1)))
                        }
                    }.value
                    guard key == loadingKey, let current else { return }
                    let root = SCNNode(); root.name = "surfaces"
                    let extent = current.sample.extent
                    root.position = SCNVector3(-extent[2]/2, -extent[1]/2, -extent[0]/2)
                    for (label, mesh) in meshes {
                        let geometry = mesh.geometry()
                        // Pass original XYZ physical coordinates to fragment clipping.
                        geometry.shaderModifiers = [
                            .geometry: """
                            #pragma varyings
                            float3 physicalPosition;
                            #pragma body
                            out.physicalPosition = _geometry.position.xyz;
                            """,
                            .fragment: """
                            #pragma arguments
                            float clipEnabled;
                            float clipAxis;
                            float clipPosition;
                            #pragma body
                            float coordinate = clipAxis < 0.5 ? in.physicalPosition.z : (clipAxis < 1.5 ? in.physicalPosition.y : in.physicalPosition.x);
                            if (clipEnabled > 0.5 && coordinate > clipPosition) discard_fragment();
                            """
                        ]
                        let node = SCNNode(geometry: geometry); node.name = String(label)
                        nodes[label] = node; root.addChildNode(node)
                    }
                    view.scene?.rootNode.addChildNode(root)
                    frame(config: current); updateMaterials(config: current)
                } catch { if key == loadingKey { config.onError(error.localizedDescription) } }
            }
        }
        func frame(config: ReconstructionView) {
            guard let view else { return }
            view.scene?.rootNode.childNode(withName: "camera", recursively: false)?.removeFromParentNode()
            let node = SCNNode(); node.name = "camera"; node.camera = SCNCamera()
            let span = config.cameraSpan ?? config.sample.extent.max() ?? 80
            node.camera?.usesOrthographicProjection = true
            node.camera?.orthographicScale = span * 0.68
            node.camera?.zNear = 0.01; node.camera?.zFar = 10000
            node.position = SCNVector3(span*1.1, -span*1.4, span*1.1)
            node.look(at: SCNVector3Zero, up: SCNVector3(0,0,1), localFront: SCNVector3(0,0,-1))
            view.scene?.rootNode.addChildNode(node); view.pointOfView = node
            view.defaultCameraController.target = SCNVector3Zero
        }
        func updateMaterials(config: ReconstructionView) {
            let appearance = config.appearance, clipping = config.clipState
            let appearanceChanged = appliedAppearance != appearance, clipChanged = appliedClip != clipping
            guard appearanceChanged || clipChanged else { return }
            SCNTransaction.begin(); SCNTransaction.animationDuration = 0
            if appearanceChanged {
                if let root = view?.scene?.rootNode.childNode(withName: "surfaces", recursively: false) {
                    let extent = config.sample.extent
                    root.position = SCNVector3(-extent[2]/2, -extent[1]/2, -extent[0]/2)
                    if config.centered && config.isolate, let node = nodes[config.selectedLabel] {
                        let bounds = node.boundingBox
                        root.position = SCNVector3(-(bounds.min.x+bounds.max.x)/2, -(bounds.min.y+bounds.max.y)/2, -(bounds.min.z+bounds.max.z)/2)
                    }
                }
                for (label, node) in nodes {
                    let selected = label == config.selectedLabel
                    let excluded = appearance.excluded.contains(label)
                    node.isHidden = (config.hideExcluded && excluded) || (config.isolate && config.selectedLabel > 0 && !selected)
                    let material = node.geometry?.firstMaterial ?? SCNMaterial()
                    material.lightingModel = config.pointCloudURL != nil && selected ? .constant : .physicallyBased
                    material.diffuse.contents = excluded ? NSColor.systemRed : selected ? NSColor.systemOrange : NSColor(calibratedHue: CGFloat(label % 11)/11, saturation: 0.55, brightness: 0.83, alpha: 1)
                    material.roughness.contents = 0.65
                    material.emission.contents = selected ? NSColor(calibratedRed: 0.22, green: 0.09, blue: 0.01, alpha: 1) : NSColor.black
                    material.isDoubleSided = true
                    node.geometry?.materials = [material]
                }
                appliedAppearance = appearance
            }
            if clipChanged {
                for node in nodes.values {
                    node.geometry?.setValue(clipping.enabled ? 1.0 : 0.0, forKey: "clipEnabled")
                    node.geometry?.setValue(Double(clipping.axis), forKey: "clipAxis")
                    node.geometry?.setValue(clipping.position, forKey: "clipPosition")
                }
                appliedClip = clipping
            }
            SCNTransaction.commit()
        }
    }
}
