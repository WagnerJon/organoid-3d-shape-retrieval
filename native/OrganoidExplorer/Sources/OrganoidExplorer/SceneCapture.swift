import SwiftUI
import AppKit
import SceneKit
import Metal

struct ReconstructionSceneSeed {
    let identity: String
    let scene: SCNScene
    let camera: SCNNode
}

/// Each expanded window retains only a weak reference to its own SceneKit view.
/// A render clones visible geometry and the current camera; it performs no data or mesh loading.
@MainActor final class PlotSceneCapture: ObservableObject {
    private weak var view: SCNView?
    private var plainColors: [String: NSColor] = [:]
    private var baseScales: [String: Float] = [:]
    private var colorMap: PlotColorMap?
    private var groups: Set<Int> = []
    private var reconstruction = false
    private var identity = ""
    private var categoricalLegend: String?
    private var axisEndpoints = [SCNVector3(4.7,0,0),SCNVector3(0,4.7,0),SCNVector3(0,0,4.7)]
    func seed() -> ReconstructionSceneSeed? {
        guard !loading, let view, let scene = view.scene, let camera = view.pointOfView, !identity.isEmpty else { return nil }
        let copy = Self.copyScene(scene)
        let pointOfView = SCNNode(); pointOfView.camera = camera.camera?.copy() as? SCNCamera
        pointOfView.transform = camera.worldTransform; copy.rootNode.addChildNode(pointOfView)
        return ReconstructionSceneSeed(identity: identity, scene: copy, camera: pointOfView)
    }
    private static func copyScene(_ source: SCNScene) -> SCNScene {
        let scene = SCNScene()
        scene.rootNode.transform = source.rootNode.transform
        for child in source.rootNode.childNodes { scene.rootNode.addChildNode(child.clone()) }
        scene.lightingEnvironment.contents = source.lightingEnvironment.contents
        scene.lightingEnvironment.intensity = source.lightingEnvironment.intensity
        scene.rootNode.enumerateChildNodes { node, _ in
            if let geometry = node.geometry?.copy() as? SCNGeometry {
                geometry.materials = geometry.materials.compactMap { $0.copy() as? SCNMaterial }; node.geometry = geometry
            }
        }
        return scene
    }
    @Published private(set) var loading = false
    func setLoading(_ value: Bool) { if loading != value { loading = value } }
    var aspectRatio: CGFloat {
        guard let view, view.bounds.height > 0 else { return 1.5 }
        return max(0.5, min(3, view.bounds.width / view.bounds.height))
    }

    func update(view: SCNView, colors: [String: NSColor], baseScales: [String: Float],
                map: PlotColorMap, groups: Set<Int>, reconstruction: Bool = false, identity: String = "",
                axisEndpoints: [SCNVector3]? = nil, categoricalLegend: String? = nil) {
        self.view = view; plainColors = colors; self.baseScales = baseScales
        colorMap = map; self.groups = groups; self.reconstruction = reconstruction; self.identity = identity
        self.categoricalLegend = categoricalLegend
        if let axisEndpoints { self.axisEndpoints = axisEndpoints }
    }

    func snapshot(options: PlotRenderOptions, size: CGSize) throws -> NSImage {
        guard size.width.isFinite, size.height.isFinite, size.width >= 64, size.height >= 64,
              size.width <= 16_384, size.height <= 16_384, size.width * size.height <= 64_000_000 else {
            throw DataError.invalid("Choose a 3D image size of at least 64 pixels per side and no more than 64 megapixels.")
        }
        guard let view, let source = view.scene, let currentCamera = view.pointOfView,
              view.bounds.width > 0, view.bounds.height > 0 else {
            throw DataError.invalid("The 3D plot is not ready to render. Open the plot and try again.")
        }
        guard !loading else { throw DataError.invalid("Wait for the cell surfaces to finish loading, then try again.") }
        var hasCells = false
        source.rootNode.enumerateChildNodes { node, _ in
            if !node.isHidden && (self.reconstruction ? node.name.flatMap(Int.init) != nil : node.name?.hasPrefix("cell:") == true) { hasCells = true }
        }
        guard hasCells else {
            throw DataError.invalid("No cells are visible in this plot. Adjust its filters and try again.")
        }
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw DataError.invalid("3D plot rendering is unavailable on this Mac.")
        }
        let width = Int(size.width.rounded()), height = Int(size.height.rounded())
        guard width > 0, height > 0,
              let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height,
                bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0),
              let graphics = NSGraphicsContext(bitmapImageRep: bitmap) else {
            throw DataError.invalid("Could not allocate the plot rendering image.")
        }
        let scene = SCNScene()
        let root = source.rootNode.clone()
        scene.rootNode.addChildNode(root)
        scene.lightingEnvironment.contents = source.lightingEnvironment.contents
        scene.lightingEnvironment.intensity = source.lightingEnvironment.intensity
        scene.background.contents = options.backgroundColor
        // SceneKit node clones share materials by default. Copy them before styling so
        // hiding highlights, changing axes, or previewing a background cannot change the plot.
        root.enumerateChildNodes { node, _ in
            node.removeAllActions()
            if let geometry = node.geometry?.copy() as? SCNGeometry {
                geometry.materials = geometry.materials.compactMap { $0.copy() as? SCNMaterial }
                node.geometry = geometry
            }
            if node.name?.hasPrefix("plot-axis:") == true {
                node.isHidden = !options.showAxes
                node.geometry?.materials.forEach { $0.diffuse.contents = NSColor(options.axisColor) }
            }
            if node.name == "plot-grid" {
                node.isHidden = !options.showGrid
                node.childNodes.forEach { line in
                    if let geometry = line.geometry?.copy() as? SCNGeometry {
                        geometry.materials = geometry.materials.compactMap { $0.copy() as? SCNMaterial }
                        line.geometry = geometry
                    }
                    line.geometry?.materials.forEach {
                        $0.diffuse.contents = NSColor(options.axisColor).withAlphaComponent(0.2)
                        $0.transparency = 0.45
                    }
                }
            }
            if !options.highlightSelection, let name = node.name,
               (self.reconstruction ? Int(name) != nil : name.hasPrefix("cell:")) {
                let id = self.reconstruction ? name : String(name.dropFirst(5)), scale = self.baseScales[id] ?? 1
                node.opacity = 1; node.scale = SCNVector3(scale, scale, scale)
                node.geometry?.materials.forEach {
                    $0.diffuse.contents = self.plainColors[id] ?? .gray
                    $0.emission.contents = NSColor.black
                }
            }
        }
        let camera = SCNNode()
        camera.camera = currentCamera.camera?.copy() as? SCNCamera
        camera.transform = currentCamera.worldTransform
        scene.rootNode.addChildNode(camera)
        let renderer = SCNRenderer(device: device, options: nil)
        renderer.scene = scene; renderer.pointOfView = camera
        renderer.autoenablesDefaultLighting = view.autoenablesDefaultLighting
        renderer.sceneTime = view.sceneTime

        let scale = max(0.5, CGFloat(min(width, height)) / 800)
        let margin = 22 * scale
        let titleHeight: CGFloat = options.showTitle && !options.title.isEmpty ? 48 * scale : 0
        let legendHeight = options.showLegend ? self.legendHeight(width: CGFloat(width), scale: scale) : 0
        let plotRect = CGRect(x: 0, y: legendHeight, width: CGFloat(width),
                              height: max(1, CGFloat(height) - titleHeight - legendHeight))
        let image = renderer.snapshot(atTime: view.sceneTime, with: plotRect.size, antialiasingMode: .multisampling4X)
        NSGraphicsContext.saveGraphicsState(); NSGraphicsContext.current = graphics
        defer { NSGraphicsContext.restoreGraphicsState() }
        graphics.cgContext.clear(CGRect(x: 0, y: 0, width: width, height: height))
        options.backgroundColor.setFill()
        NSBezierPath(rect: CGRect(x: 0, y: 0, width: width, height: height)).fill()
        image.draw(in: plotRect, from: .zero, operation: .sourceOver, fraction: 1)
        let foreground = options.foregroundColor
        if titleHeight > 0 {
            drawText(options.title, rect: CGRect(x: margin, y: CGFloat(height) - titleHeight + 8 * scale,
                                                width: CGFloat(width) - 2 * margin, height: 32 * scale),
                     color: foreground, size: 22 * scale, bold: true)
        }
        if options.showAxes {
            let labels = [options.xLabel, options.yLabel, options.zLabel]
            for axis in 0..<3 where !labels[axis].isEmpty {
                let endpoint = axisEndpoints[axis]
                let point = renderer.projectPoint(endpoint)
                guard point.x.isFinite, point.y.isFinite else { continue }
                let measured = (labels[axis] as NSString).size(withAttributes: [.font: NSFont.boldSystemFont(ofSize: 16 * scale)]).width
                let labelWidth = min(CGFloat(width) - 2 * margin, max(12 * scale, measured + 8 * scale))
                let x = max(margin, min(CGFloat(width) - margin - labelWidth, CGFloat(point.x) - labelWidth / 2))
                let y = max(plotRect.minY + margin, min(plotRect.maxY - 28 * scale, plotRect.minY + CGFloat(point.y) + 6 * scale))
                drawText(labels[axis], rect: CGRect(x: x, y: y, width: labelWidth, height: 26 * scale),
                         color: NSColor(options.axisColor), size: 16 * scale, bold: true)
            }
        }
        if options.showLegend { drawLegend(width: CGFloat(width), scale: scale, foreground: foreground) }
        let output = NSImage(size: CGSize(width: width, height: height))
        bitmap.size = output.size; output.addRepresentation(bitmap)
        return output
    }

    private func drawText(_ text: String, rect: CGRect, color: NSColor, size: CGFloat, bold: Bool = false) {
        let paragraph = NSMutableParagraphStyle(); paragraph.lineBreakMode = .byTruncatingTail
        (text as NSString).draw(in: rect, withAttributes: [
            .font: bold ? NSFont.boldSystemFont(ofSize: size) : NSFont.systemFont(ofSize: size),
            .foregroundColor: color, .paragraphStyle: paragraph
        ])
    }
    private func legendHeight(width: CGFloat, scale: CGFloat) -> CGFloat {
        guard let map = colorMap else { return 0 }
        if !map.settings.feature.isEmpty { return 74 * scale }
        if categoricalLegend != nil { return 40 * scale }
        let columns = max(1, Int((width - 44 * scale) / (112 * scale)))
        return CGFloat(max(1, (groups.count + columns - 1) / columns)) * 25 * scale + 18 * scale
    }
    private func drawLegend(width: CGFloat, scale: CGFloat, foreground: NSColor) {
        guard let map = colorMap else { return }
        let margin = 22 * scale
        if !map.settings.feature.isEmpty {
            let label = plotFeatureName(map.settings.feature) + " · " + map.settings.palette.rawValue
            drawText(label, rect: CGRect(x: margin, y: 47 * scale, width: width - 2 * margin, height: 22 * scale),
                     color: foreground, size: 14 * scale, bold: true)
            let bar = CGRect(x: margin, y: 29 * scale, width: min(width - 2 * margin, 300 * scale), height: 8 * scale)
            if let lo = map.minimum, let hi = map.maximum {
                for step in 0..<256 {
                    map.settings.palette.color(hi > lo ? Double(step) / 255 : 0.5).setFill()
                    NSBezierPath(rect: CGRect(x: bar.minX + CGFloat(step) / 256 * bar.width, y: bar.minY,
                                              width: bar.width / 256 + 1, height: bar.height)).fill()
                }
                drawText(String(format: "%.4g", lo), rect: CGRect(x: bar.minX, y: 5 * scale, width: bar.width / 2, height: 20 * scale),
                         color: foreground, size: 12 * scale)
                let high = String(format: "%.4g", hi)
                let highWidth = (high as NSString).size(withAttributes: [.font: NSFont.systemFont(ofSize: 12 * scale)]).width
                drawText(high, rect: CGRect(x: bar.maxX - highWidth, y: 5 * scale, width: highWidth + 2, height: 20 * scale),
                         color: foreground, size: 12 * scale)
                if map.missing > 0 {
                    drawText("Gray: missing value", rect: CGRect(x: bar.maxX + 16 * scale, y: 24 * scale,
                                                               width: width - bar.maxX - margin, height: 20 * scale),
                             color: foreground, size: 12 * scale)
                }
            } else {
                drawText("No values available · gray cells", rect: CGRect(x: margin, y: 18 * scale,
                                                                       width: width - 2 * margin, height: 22 * scale),
                         color: foreground, size: 12 * scale)
            }
        } else if let categoricalLegend {
            drawText(categoricalLegend, rect: CGRect(x: margin, y: 10 * scale, width: width - 2 * margin, height: 22 * scale),
                     color: foreground, size: 13 * scale)
        } else {
            let columns = max(1, Int((width - 2 * margin) / (112 * scale)))
            let sorted = groups.sorted()
            let rows = max(1, (sorted.count + columns - 1) / columns)
            for (index, group) in sorted.enumerated() {
                let x = margin + CGFloat(index % columns) * 112 * scale
                let y = 10 * scale + CGFloat(rows - 1 - index / columns) * 25 * scale
                map.settings.palette.groupColor(group).setFill()
                NSBezierPath(ovalIn: CGRect(x: x, y: y + 5 * scale, width: 10 * scale, height: 10 * scale)).fill()
                drawText(group < 0 ? "Unassigned" : "Group \(group + 1)",
                         rect: CGRect(x: x + 16 * scale, y: y, width: 94 * scale, height: 21 * scale),
                         color: foreground, size: 12 * scale)
            }
        }
    }
}
