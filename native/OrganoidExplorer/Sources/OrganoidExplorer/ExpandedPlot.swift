import SwiftUI
import SceneKit
import AppKit
import Metal

/// Each plot owns a native window, so macOS full-screen and multiple monitors work.
final class PlotWindows: NSObject, NSWindowDelegate {
    static let shared = PlotWindows()
    private var windows: [ObjectIdentifier: NSWindowController] = [:]
    @MainActor func open(method: Method, store: ExplorerStore, threeD: Bool = false,
                         cohort: Set<String>? = nil, groups: [String: Int]? = nil,
                         colorMethod: String? = nil, featureColoring: Bool = false) {
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1200, height: 800),
                              styleMask: [.titled, .closable, .resizable, .miniaturizable], backing: .buffered, defer: false)
        window.title = method.name
        window.collectionBehavior = [.fullScreenPrimary]
        window.contentViewController = NSHostingController(rootView:
            ExpandedPlotView(method: method, initialThreeD: threeD, cohort: cohort, groups: groups, colorMethod: colorMethod, featureColoring: featureColoring,
                             close: { [weak window] in window?.performClose(nil) },
                             toggleFullScreen: { [weak window] in window?.toggleFullScreen(nil) }).environmentObject(store))
        let controller = NSWindowController(window: window)
        windows[ObjectIdentifier(window)] = controller
        window.delegate = self; window.center(); controller.showWindow(nil)
        window.makeKeyAndOrderFront(nil)
        DispatchQueue.main.async { [weak window] in window?.toggleFullScreen(nil) }
    }
    func windowWillClose(_ notification: Notification) {
        if let window = notification.object as? NSWindow { windows.removeValue(forKey: ObjectIdentifier(window)) }
    }
}

struct ExpandedPlotView: View {
    @EnvironmentObject var store: ExplorerStore
    let method: Method
    let initialThreeD: Bool
    var cohort: Set<String>?
    var groups: [String: Int]?
    var colorMethod: String?
    var featureColoring = false
    let close: () -> Void
    let toggleFullScreen: () -> Void
    @State private var threeD = false
    @State private var result: ShapeAnalysisResult?
    @State private var error: String?
    @State private var loading = false
    @State private var reset = 0
    @State private var cellShapes = false
    @StateObject private var sceneCapture = PlotSceneCapture()
    var usesFeatureColors: Bool { featureColoring || store.unifiedPlotColors }
    var isDistanceMap: Bool { ["precomputed", "shape_hybrid95"].contains(method.metric) }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text(method.name).font(.title2.bold())
                    Text(groups == nil ? "Linked selection · groups from the comparison view" : "Discovered groups · gray cells are unassigned")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Picker("Dimensions", selection: $threeD) {
                    Text(isDistanceMap ? "2 axes" : "2 PCs").tag(false)
                    Text(isDistanceMap ? "3 axes" : "3 PCs").tag(true)
                }.pickerStyle(.segmented).labelsHidden().frame(width: 170)
                Button { cellShapes.toggle() } label: {
                    Label(cellShapes ? "Points" : "Cell shapes", systemImage: cellShapes ? "circle.grid.3x3" : "cube")
                }.disabled(store.level != "Cells" || store.folder == nil)
                    .help("Load reconstructed cell surfaces in this window only")
                if threeD {
                    PlotRenderButton(title: method.name,
                                     xLabel: isDistanceMap ? "Axis 1" : "PC 1",
                                     yLabel: isDistanceMap ? "Axis 2" : "PC 2",
                                     zLabel: isDistanceMap ? "Axis 3" : "PC 3",
                                     aspectRatio: sceneCapture.aspectRatio, threeD: true) { options, size in
                        try sceneCapture.snapshot(options: options, size: size)
                    }.disabled(result == nil || sceneCapture.loading)
                    Button("Reset camera") { reset += 1 }
                }
                Button(action: toggleFullScreen) { Image(systemName: "arrow.up.left.and.arrow.down.right") }.help("Toggle macOS full screen")
                Button("Done", action: close).keyboardShortcut(.cancelAction)
            }
            if usesFeatureColors && (threeD || store.unifiedPlotColors) {
                let items = store.filtered.filter { cohort?.contains($0.id) ?? true }
                PlotColorControls(settings: store.plotColorBinding(method.id), features: plotFeatureKeys(store.allItems),
                                  map: PlotColorMap(settings: store.plotColors(method.id), items: store.plotColorDomain(items)),
                                  scope: store.unifiedPlotColors ? .allMethods : .thisPlot).frame(maxWidth: 420)
            }
            if threeD {
                if let result {
                    let items = store.filtered.filter { cohort?.contains($0.id) ?? true }
                    PlotCellScatter3D(result: result, items: items, colorMethod: colorMethod ?? method.id, groups: groups,
                                   selected: store.selectedID, marked: store.selection, reset: reset,
                                   folder: store.folder, cellShapes: cellShapes && store.level == "Cells",
                                   coloring: usesFeatureColors ? store.plotColors(method.id) : PlotColorSettings(),
                                   colorDomain: store.plotColorDomain(items), capture: sceneCapture) { id in
                        store.markPlotItems([id]); store.select(id)
                    }.clipShape(RoundedRectangle(cornerRadius: 12))
                    HStack(spacing: 24) {
                        ForEach(0..<3) { index in
                            Text("\(isDistanceMap ? "Axis" : "PC") \(index + 1): \(Int((result.variance?[index] ?? 0) * 100))%")
                                .foregroundStyle([Color.red, .green, .blue][index])
                        }
                        Spacer()
                        Text(isDistanceMap ? "Positive inertia of the distance map" : "Explained variance")
                    }.font(.caption)
                } else if loading {
                    ProgressView("Preparing the third axis…").frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    ContentUnavailableView("Plot unavailable", systemImage: "chart.xyaxis.line", description: Text(error ?? "No results"))
                }
            } else {
                ScatterPanel(methodID: method.id, title: method.name, colorMethodID: colorMethod,
                             cohortIDs: cohort, groupLabels: groups, featureColoring: featureColoring, showActions: false, cellShapes: $cellShapes)
            }
            Text(threeD ? "Drag to rotate · scroll to zoom · click a cell to mark its table row. Axes retain a shared scale. Three axes are for display; similarity and grouping use the full descriptors."
                 : "Click to select · drag a rectangle to mark cells across plots and tables.")
                .font(.caption).foregroundStyle(.secondary)
            HStack {
                Text(store.selectedID ?? "Select a point").font(.caption.monospaced()).foregroundStyle(.teal)
                Button("Inspect selection in 3D") { store.activeTab = "explore"; close() }.disabled(store.analysisSelected == nil)
                Spacer()
                if groups != nil { Text("Group labels are a snapshot from when this window opened.").foregroundStyle(.secondary) }
            }.font(.caption)
            if isDistanceMap {
                Text("This method uses a distance map (MDS), not principal components of packed distance rows.").font(.caption).foregroundStyle(.secondary)
            }
        }.padding(20).frame(minWidth: 820, minHeight: 600)
            .onAppear { threeD = initialThreeD }
            .onChange(of: store.level) { if store.level != "Cells" { cellShapes = false } }
            .task(id: "\(store.folder?.path ?? "")|\(store.level)|\(store.analysisRevision)") {
                result = nil; error = nil; loading = true
                guard let folder = store.folder else { loading = false; return }
                do {
                    let value = try await ShapeAnalysis.run(folder: folder, method: method.id, level: store.level, action: "projection")
                    guard !Task.isCancelled else { return }
                    result = value
                } catch { if !Task.isCancelled { self.error = error.localizedDescription } }
                if !Task.isCancelled { loading = false }
            }
    }
}

/// Owns only this expanded plot's mesh assets; point mode performs no mesh I/O.
struct PlotCellScatter3D: View {
    let result: ShapeAnalysisResult
    let items: [Specimen]
    let colorMethod: String
    var groups: [String: Int]?
    let selected: String?
    let marked: Set<String>
    let reset: Int
    let folder: URL?
    let cellShapes: Bool
    var coloring = PlotColorSettings()
    var colorDomain: [Specimen]? = nil
    var capture: PlotSceneCapture? = nil
    let pick: (String) -> Void
    @StateObject private var assets = PlotCellAssets()
    @State private var shapeScale = 0.22
    var plottedItems: [Specimen] { let ids = Set(result.ids); return items.filter { ids.contains($0.id) } }
    var body: some View {
        VStack(spacing: 6) {
            if cellShapes {
                HStack {
                    PlotCellLoadStatus(assets: assets)
                    Text("Shape size").font(.caption2).foregroundStyle(.secondary)
                    Slider(value: $shapeScale, in: 0.08...0.6).frame(width: 130)
                        .accessibilityLabel("Cell surface size in this plot")
                }
            }
            ShapeScatter3D(result: result, items: items, colorMethod: colorMethod, groups: groups,
                           selected: selected, marked: marked, reset: reset, cellShapes: cellShapes,
                           surfaces: assets.surfaces, shapeScale: Float(shapeScale), coloring: coloring, colorDomain: colorDomain, capture: capture, pick: pick)
        }
        .task(id: PlotCellAssets.requestKey(enabled: cellShapes, folder: folder, items: plottedItems)) {
            guard cellShapes, let folder else { assets.clear(); return }
            await assets.load(folder: folder, items: plottedItems, mode: .surfaces)
        }
        .onChange(of: assets.loading) { capture?.setLoading(cellShapes && assets.loading) }
        .onChange(of: cellShapes) { capture?.setLoading(cellShapes && assets.loading) }
        .onDisappear { assets.clear(); capture?.setLoading(false) }
    }
}

final class ShapePlotSceneView: SCNView {
    var pick: ((String) -> Void)?
    private var down = NSPoint.zero
    override func mouseDown(with event: NSEvent) { down = event.locationInWindow; super.mouseDown(with: event) }
    override func mouseUp(with event: NSEvent) {
        super.mouseUp(with: event)
        guard hypot(event.locationInWindow.x - down.x, event.locationInWindow.y - down.y) < 5 else { return }
        let location = convert(event.locationInWindow, from: nil)
        if let id = hitTest(location).compactMap({ $0.node.name }).first(where: { $0.hasPrefix("cell:") }) {
            pick?(String(id.dropFirst(5)))
        }
    }
}

struct ShapeScatter3D: NSViewRepresentable {
    let result: ShapeAnalysisResult
    let items: [Specimen]
    let colorMethod: String
    var groups: [String: Int]?
    let selected: String?
    let marked: Set<String>
    let reset: Int
    var cellShapes = false
    var surfaces: [String: PlotCellSurface] = [:]
    var shapeScale: Float = 0.22
    var coloring = PlotColorSettings()
    var colorDomain: [Specimen]? = nil
    var capture: PlotSceneCapture? = nil
    let pick: (String) -> Void
    func makeCoordinator() -> Coordinator { Coordinator() }
    func makeNSView(context: Context) -> ShapePlotSceneView {
        let view = ShapePlotSceneView()
        view.allowsCameraControl = true; view.backgroundColor = NSColor(calibratedWhite: 0.06, alpha: 1)
        view.antialiasingMode = .multisampling4X
        return view
    }
    func updateNSView(_ view: ShapePlotSceneView, context: Context) {
        view.pick = pick
        context.coordinator.update(view, config: self)
    }
    @MainActor final class Coordinator {
        var signature: Int?
        var nodes: [String: SCNNode] = [:]
        var colors: [String: NSColor] = [:]
        var surfaceNodes: Set<String> = []
        func update(_ view: ShapePlotSceneView, config: ShapeScatter3D) {
            var hash = Hasher(); hash.combine(config.reset); hash.combine(config.result.method)
            hash.combine(config.cellShapes)
            let allowed = Dictionary(uniqueKeysWithValues: config.items.map { ($0.id, $0) })
            for (i, id) in config.result.ids.enumerated() where allowed[id] != nil {
                hash.combine(id); hash.combine(config.result.points?[i])
                hash.combine(config.groups?[id] ?? allowed[id]?.representations[config.colorMethod]?.cluster ?? -1)
            }
            let stamp = hash.finalize()
            if signature != stamp {
                signature = stamp; nodes = [:]; colors = [:]; surfaceNodes = []
                let scene = SCNScene()
                let ambient = SCNNode(); ambient.light = SCNLight(); ambient.light?.type = .ambient; ambient.light?.intensity = 500
                scene.rootNode.addChildNode(ambient)
                let light = SCNNode(); light.light = SCNLight(); light.light?.type = .directional; light.light?.intensity = 900
                light.eulerAngles = SCNVector3(-0.5, -0.6, 0); scene.rootNode.addChildNode(light)
                let camera = SCNNode(); camera.camera = SCNCamera(); camera.position = SCNVector3(10, 8, 12)
                camera.camera?.zNear = 0.01; camera.camera?.zFar = 200
                camera.camera?.usesOrthographicProjection = true; camera.camera?.orthographicScale = 5.8
                camera.look(at: SCNVector3Zero)
                scene.rootNode.addChildNode(camera)
                let all = config.result.points ?? []
                let visible = config.result.ids.enumerated().compactMap { allowed[$0.element] != nil && $0.offset < all.count ? all[$0.offset] : nil }
                let minimum = (0..<3).map { axis in visible.map { $0[axis] }.min() ?? 0 }
                let maximum = (0..<3).map { axis in visible.map { $0[axis] }.max() ?? 0 }
                let span = max(zip(minimum, maximum).map { $1 - $0 }.max() ?? 1, 1e-12)
                for (i, id) in config.result.ids.enumerated() {
                    guard let item = allowed[id], i < all.count else { continue }
                    let sphere = SCNSphere(radius: 0.047); sphere.segmentCount = 8
                    let material = SCNMaterial(); material.lightingModel = .constant; sphere.materials = [material]
                    let node = SCNNode(geometry: sphere); node.name = "cell:" + id
                    let p = (0..<3).map { axis in Float((all[i][axis] - (minimum[axis] + maximum[axis])/2) / span * 8) }
                    node.position = SCNVector3(p[0], p[1], p[2])
                    let group = config.groups.map { $0[id] ?? -1 } ?? item.representations[config.colorMethod]?.cluster ?? -1
                    colors[id] = group < 0 ? .gray : NSColor(clusterColors[group % clusterColors.count])
                    nodes[id] = node; scene.rootNode.addChildNode(node)
                }
                for axis in 0..<3 {
                    let lengths: [CGFloat] = (0..<3).map { $0 == axis ? 9 : 0.014 }
                    let line = SCNBox(width: lengths[0], height: lengths[1], length: lengths[2], chamferRadius: 0)
                    line.firstMaterial?.diffuse.contents = [NSColor.systemRed, .systemGreen, .systemBlue][axis]
                    line.firstMaterial?.lightingModel = .constant
                    let axisNode = SCNNode(geometry: line); axisNode.name = "plot-axis:\(axis)"
                    scene.rootNode.addChildNode(axisNode)
                }
                // The grid is export-only, so rendering options never alter the interactive plot.
                let grid = SCNNode(); grid.name = "plot-grid"; grid.isHidden = true
                for coordinate in -4...4 where coordinate != 0 {
                    for direction in 0..<2 {
                        let line = SCNBox(width: direction == 0 ? 8 : 0.008,
                                          height: direction == 0 ? 0.008 : 8, length: 0.008, chamferRadius: 0)
                        line.firstMaterial?.lightingModel = .constant
                        let node = SCNNode(geometry: line)
                        node.position = direction == 0 ? SCNVector3(0, Float(coordinate), 0) : SCNVector3(Float(coordinate), 0, 0)
                        grid.addChildNode(node)
                    }
                }
                scene.rootNode.addChildNode(grid)
                view.scene = scene; view.pointOfView = camera
                view.defaultCameraController.target = SCNVector3Zero
                view.defaultCameraController.interactionMode = .orbitTurntable
            }
            let colorMap = PlotColorMap(settings: config.coloring, items: config.colorDomain ?? config.items)
            for (id, node) in nodes {
                if let item = allowed[id] {
                    let group = config.groups.map { $0[id] ?? -1 } ?? item.representations[config.colorMethod]?.cluster ?? -1
                    colors[id] = colorMap.color(item: item, group: group)
                }
                if config.cellShapes, !surfaceNodes.contains(id), let surface = config.surfaces[id] {
                    let geometry = surface.geometry(), material = SCNMaterial()
                    material.lightingModel = .lambert; material.isDoubleSided = true
                    geometry.materials = [material]; node.geometry = geometry; surfaceNodes.insert(id)
                }
                node.geometry?.firstMaterial?.diffuse.contents = id == config.selected && config.coloring.feature.isEmpty ? NSColor.systemOrange : colors[id]
                node.opacity = config.marked.isEmpty || config.marked.contains(id) || id == config.selected ? 1 : 0.2
                let base: Float = surfaceNodes.contains(id) ? config.shapeScale : 1
                let scale: Float = base * (id == config.selected ? 1.9 : config.marked.contains(id) ? 1.35 : 1)
                node.scale = SCNVector3(scale, scale, scale)
            }
            config.capture?.update(view: view, colors: colors,
                                   baseScales: Dictionary(uniqueKeysWithValues: nodes.keys.map { id in
                                       (id, surfaceNodes.contains(id) ? config.shapeScale : Float(1))
                                   }), map: colorMap,
                                   groups: Set(config.result.ids.compactMap { id in
                                       guard let item = allowed[id] else { return nil }
                                       return config.groups.map { $0[id] ?? -1 } ?? item.representations[config.colorMethod]?.cluster ?? -1
                                   }))
        }
    }
}

/// Each expanded window retains only a weak reference to its own SceneKit view.
/// A render clones visible geometry and the current camera; it performs no data or mesh loading.
@MainActor final class PlotSceneCapture: ObservableObject {
    private weak var view: ShapePlotSceneView?
    private var plainColors: [String: NSColor] = [:]
    private var baseScales: [String: Float] = [:]
    private var colorMap: PlotColorMap?
    private var groups: Set<Int> = []
    @Published private(set) var loading = false
    func setLoading(_ value: Bool) { if loading != value { loading = value } }
    var aspectRatio: CGFloat {
        guard let view, view.bounds.height > 0 else { return 1.5 }
        return max(0.5, min(3, view.bounds.width / view.bounds.height))
    }

    func update(view: ShapePlotSceneView, colors: [String: NSColor], baseScales: [String: Float],
                map: PlotColorMap, groups: Set<Int>) {
        self.view = view; plainColors = colors; self.baseScales = baseScales
        colorMap = map; self.groups = groups
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
        guard source.rootNode.childNodes.contains(where: { $0.name?.hasPrefix("cell:") == true }) else {
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
            if !options.highlightSelection, let name = node.name, name.hasPrefix("cell:") {
                let id = String(name.dropFirst(5)), scale = self.baseScales[id] ?? 1
                node.opacity = 1; node.scale = SCNVector3(scale, scale, scale)
                node.geometry?.materials.forEach { $0.diffuse.contents = self.plainColors[id] ?? .gray }
            }
        }
        let camera = SCNNode()
        camera.camera = currentCamera.camera?.copy() as? SCNCamera
        camera.transform = currentCamera.worldTransform
        scene.rootNode.addChildNode(camera)
        let renderer = SCNRenderer(device: device, options: nil)
        renderer.scene = scene; renderer.pointOfView = camera
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
                let endpoint = SCNVector3(axis == 0 ? 4.7 : 0, axis == 1 ? 4.7 : 0, axis == 2 ? 4.7 : 0)
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
