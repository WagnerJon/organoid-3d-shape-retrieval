import SwiftUI
import AppKit

@MainActor final class ReconstructionWindows: NSObject, NSWindowDelegate {
    static let shared = ReconstructionWindows()
    private var windows: [ObjectIdentifier: NSWindowController] = [:]
    func open(store: ExplorerStore, seed: ReconstructionSceneSeed? = nil, link: CameraLink? = nil) {
        guard store.folder != nil, store.sample != nil else { return }
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1250, height: 850),
                              styleMask: [.titled,.closable,.resizable,.miniaturizable], backing: .buffered, defer: false)
        window.title = "Reconstruction · Organoid Atlas"
        window.minSize = NSSize(width: 850, height: 600)
        window.collectionBehavior = [.fullScreenPrimary]
        window.contentViewController = NSHostingController(rootView:
            ExpandedReconstructionView(navigation: store.sliceNavigation, seed: seed, link: link,
                close: { [weak window] in window?.performClose(nil) },
                toggleFullScreen: { [weak window] in window?.toggleFullScreen(nil) }).environmentObject(store))
        let controller = NSWindowController(window: window)
        windows[ObjectIdentifier(window)] = controller
        window.delegate = self; window.center(); controller.showWindow(nil); window.makeKeyAndOrderFront(nil)
        DispatchQueue.main.async { [weak window] in window?.toggleFullScreen(nil) }
    }
    func windowWillClose(_ notification: Notification) {
        if let window = notification.object as? NSWindow { windows.removeValue(forKey: ObjectIdentifier(window)) }
    }
}

struct ExpandedReconstructionView: View {
    @EnvironmentObject var store: ExplorerStore
    @ObservedObject var navigation: SliceNavigation
    let seed: ReconstructionSceneSeed?
    let link: CameraLink?
    let close: () -> Void
    let toggleFullScreen: () -> Void
    @StateObject private var capture = PlotSceneCapture()
    @State private var meshError: String?
    var cells: [Specimen] { store.dataset?.cells.filter { $0.sample == store.sample?.id } ?? [] }
    var colorMap: ReconstructionColorMap { ReconstructionColorMap(cells: cells, settings: store.reconstructionColors, hideExcluded: store.hideExcluded) }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Reconstruction").font(.title2.bold())
                    Text(store.sample?.displayName ?? "Select an object in Atlas").font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                PlotRenderButton(title: store.sample?.displayName ?? "Reconstruction", xLabel: "X (µm)", yLabel: "Y (µm)", zLabel: "Z (µm)",
                                 aspectRatio: capture.aspectRatio, threeD: true) { options, size in
                    try capture.snapshot(options: options, size: size)
                }.disabled(store.sample == nil || capture.loading)
                Button("Reset camera") { store.cameraReset += 1 }
                Button(action: toggleFullScreen) { Image(systemName: "arrow.up.left.and.arrow.down.right") }.help("Toggle macOS full screen")
                Button("Done", action: close).keyboardShortcut(.cancelAction)
            }
            if let folder = store.folder, let sample = store.sample {
                HStack(alignment: .top, spacing: 24) {
                    PlotColorControls(settings: $store.reconstructionColors, features: plotFeatureKeys(cells), map: colorMap.map,
                                      emptyFeatureLabel: "Random by cell ID", showGroupPalette: false,
                                      helpOverride: "Colors apply to the reconstruction only. Feature ranges cover the organoid’s visible cells and do not change its analyses.")
                        .frame(width: 330)
                    VStack(alignment: .leading, spacing: 7) {
                        Toggle("Isolate selected cell", isOn: $store.isolate).disabled(store.level != "Cells")
                        Toggle("Hide excluded cells", isOn: $store.hideExcluded)
                        Toggle("Highlight selection", isOn: $store.reconstructionHighlight)
                    }.toggleStyle(.checkbox).font(.caption)
                    VStack(alignment: .leading, spacing: 7) {
                        Toggle("Slice 3D", isOn: $store.clip).toggleStyle(.checkbox)
                        Picker("Plane", selection: $store.axis) { Text("XY").tag(0); Text("XZ").tag(1); Text("YZ").tag(2) }
                            .pickerStyle(.segmented).labelsHidden().frame(width: 150)
                        HStack {
                            Slider(value: $navigation.fraction, in: 0...1).frame(minWidth: 100, maxWidth: 220)
                            Text(String(format: "%.1f µm", sample.extent[store.axis]*navigation.fraction)).monospacedDigit()
                        }.disabled(!store.clip)
                    }.font(.caption)
                    Spacer(minLength: 0)
                }
                ZStack(alignment: .bottomLeading) {
                    ReconstructionView(folder: folder, sample: sample, cells: cells, selectedLabel: store.selected?.label ?? 0,
                        isolate: store.isolate && store.level == "Cells", hideExcluded: store.hideExcluded,
                        clipping: store.clip, axis: store.axis, slice: navigation.fraction, reset: store.cameraReset,
                        link: link, coloring: store.reconstructionColors, highlightSelection: store.reconstructionHighlight, capture: capture, seed: seed,
                        pick: { label in if store.level == "Cells" { store.select("\(sample.id):L\(label)") } },
                        onError: { meshError = $0 })
                    if let meshError { Text(meshError).font(.caption).foregroundStyle(.orange).padding(12) }
                }.frame(maxWidth: .infinity, maxHeight: .infinity).clipShape(RoundedRectangle(cornerRadius: 12))
                HStack {
                    Text("Drag to orbit · scroll to zoom · click to select").foregroundStyle(.secondary)
                    Spacer()
                    if let selected = store.selected { Text(selected.id).monospaced() }
                    if !store.reconstructionColors.feature.isEmpty, let selected = store.selected,
                       let value = plotFeatureValue(selected, key: store.reconstructionColors.feature) {
                        Text("\(plotFeatureName(store.reconstructionColors.feature)): \(value, specifier: "%.4g")")
                    }
                }.font(.caption)
                if store.clip { Text("Slicing clips surfaces; cut faces are not capped.").font(.caption2).foregroundStyle(.secondary) }
            } else { ContentUnavailableView("Select an object", systemImage: "cube", description: Text("Choose a row or plot point in Atlas to open its reconstruction here.")) }
        }.padding(20).frame(minWidth: 810, minHeight: 540)
            .onChange(of: store.sample?.id) { meshError = nil }
    }
}
