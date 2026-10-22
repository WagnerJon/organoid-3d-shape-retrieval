import SwiftUI
import AppKit
import UniformTypeIdentifiers

enum PlotRenderBackground: String, CaseIterable, Identifiable {
    case dark = "Dark", light = "Light", transparent = "Transparent", custom = "Custom"
    var id: String { rawValue }
}

/// Export settings belong to one plot and never change its analysis or on-screen appearance.
struct PlotRenderOptions: Hashable {
    var showAxes = true
    var showGrid = true
    var showTitle = true
    var showLegend = true
    var highlightSelection = true
    var background: PlotRenderBackground = .dark
    var customBackground: Color = .white
    var axisColor: Color = .gray
    var title = ""
    var xLabel = "PC 1"
    var yLabel = "PC 2"
    var zLabel = "PC 3"
    var outputWidth = 2400

    var backgroundColor: NSColor {
        switch background {
        case .dark: return NSColor(srgbRed: 0.055, green: 0.072, blue: 0.094, alpha: 1)
        case .light: return .white
        case .transparent: return .clear
        case .custom: return NSColor(customBackground).withAlphaComponent(1)
        }
    }

    var foregroundColor: NSColor {
        if background == .transparent { return NSColor(srgbRed: 0.12, green: 0.14, blue: 0.17, alpha: 1) }
        let rgb = backgroundColor.usingColorSpace(.sRGB) ?? .white
        let luminance = 0.2126 * rgb.redComponent + 0.7152 * rgb.greenComponent + 0.0722 * rgb.blueComponent
        return luminance > 0.55 ? NSColor(srgbRed: 0.12, green: 0.14, blue: 0.17, alpha: 1) : .white
    }
}

enum PlotImageExport {
    enum ExportError: LocalizedError {
        case noBitmap, pngEncoding
        var errorDescription: String? {
            switch self {
            case .noBitmap: return "The plot could not be rendered as an image."
            case .pngEncoding: return "The rendered plot could not be saved as a PNG."
            }
        }
    }

    /// Use the renderer's pixels directly. Drawing through an NSImage focus context would
    /// apply display scaling and can flatten transparent backgrounds.
    static func pngData(_ image: NSImage) throws -> Data {
        let bitmap: NSBitmapImageRep
        if let original = image.representations.compactMap({ $0 as? NSBitmapImageRep })
            .max(by: { $0.pixelsWide * $0.pixelsHigh < $1.pixelsWide * $1.pixelsHigh }) {
            bitmap = original
        } else {
            guard let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
                throw ExportError.noBitmap
            }
            bitmap = NSBitmapImageRep(cgImage: cgImage)
        }
        guard let data = bitmap.representation(using: .png, properties: [:]) else {
            throw ExportError.pngEncoding
        }
        return data
    }

    static func writePNG(_ image: NSImage, to url: URL) throws {
        try pngData(image).write(to: url, options: .atomic)
    }
}

/// Shared rendering menu for a single 2D or 3D plot. Renderers receive explicit pixel
/// dimensions, so the resulting image is independent of the user's display scale.
struct PlotRenderButton: View {
    let title: String
    let aspectRatio: CGFloat
    let threeD: Bool
    let snapshot: @MainActor (PlotRenderOptions, CGSize) throws -> NSImage
    @State private var presented = false
    @State private var options: PlotRenderOptions
    @State private var preview: NSImage?
    @State private var error: String?
    @State private var notice: String?
    @State private var rendering = false

    init(title: String, xLabel: String = "PC 1", yLabel: String = "PC 2", zLabel: String = "PC 3",
         aspectRatio: CGFloat = 1.5, threeD: Bool = false,
         snapshot: @escaping @MainActor (PlotRenderOptions, CGSize) throws -> NSImage) {
        self.title = title
        self.aspectRatio = aspectRatio.isFinite && aspectRatio > 0 ? aspectRatio : 1.5
        self.threeD = threeD
        self.snapshot = snapshot
        var initial = PlotRenderOptions()
        initial.title = title; initial.xLabel = xLabel; initial.yLabel = yLabel; initial.zLabel = zLabel
        _options = State(initialValue: initial)
    }

    var body: some View {
        Button { presented.toggle() } label: { Image(systemName: "camera") }
            .accessibilityLabel("Render \(title)")
            .help("Render this plot as a PNG image")
            .popover(isPresented: $presented, arrowEdge: .bottom) { menu }
    }

    private var menu: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Render plot").font(.headline)
                Spacer()
                Button { presented = false } label: { Image(systemName: "xmark") }
                    .buttonStyle(.plain).accessibilityLabel("Close rendering menu")
            }
            ZStack {
                PlotTransparencyPreview()
                if let preview {
                    Image(nsImage: preview).resizable().interpolation(.high).scaledToFit().padding(6)
                } else if error != nil {
                    Label("Preview unavailable", systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.secondary)
                } else { ProgressView("Preparing preview…") }
            }
            .frame(height: 190)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(.gray.opacity(0.25)))
            HStack(alignment: .top, spacing: 24) {
                VStack(alignment: .leading, spacing: 8) {
                    Toggle("Show axes", isOn: $options.showAxes)
                    if !threeD { Toggle("Show grid", isOn: $options.showGrid) }
                    Toggle("Show title", isOn: $options.showTitle)
                    Toggle("Show legend", isOn: $options.showLegend)
                    Toggle("Highlight selected cells", isOn: $options.highlightSelection)
                }.frame(maxWidth: .infinity, alignment: .leading)
                VStack(alignment: .leading, spacing: 8) {
                    Picker("Background", selection: $options.background) {
                        ForEach(PlotRenderBackground.allCases) { background in Text(background.rawValue).tag(background) }
                    }
                    if options.background == .custom {
                        ColorPicker("Background color", selection: $options.customBackground, supportsOpacity: false)
                    }
                    if options.showAxes || (!threeD && options.showGrid) {
                        ColorPicker("Axis / grid color", selection: $options.axisColor, supportsOpacity: false)
                    }
                    Picker("Image width", selection: $options.outputWidth) {
                        ForEach([1200, 2400, 3600, 4800], id: \.self) { width in Text("\(width) px").tag(width) }
                    }
                    Text("\(options.outputWidth) × \(Int(exportSize.height)) px · PNG")
                        .font(.caption).foregroundStyle(.secondary)
                }.frame(maxWidth: .infinity, alignment: .leading)
            }.toggleStyle(.checkbox)
            if options.showTitle {
                TextField("Title", text: $options.title).textFieldStyle(.roundedBorder)
            }
            if options.showAxes {
                HStack(spacing: 8) {
                    TextField("X-axis label", text: $options.xLabel).accessibilityLabel("X-axis label")
                    TextField("Y-axis label", text: $options.yLabel).accessibilityLabel("Y-axis label")
                    if threeD { TextField("Z-axis label", text: $options.zLabel).accessibilityLabel("Z-axis label") }
                }.textFieldStyle(.roundedBorder)
            }
            HStack {
                if let error {
                    Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled)
                } else if let notice {
                    Text(notice).font(.caption).foregroundStyle(.secondary)
                } else {
                    Text("Exports only this plot.").font(.caption).foregroundStyle(.secondary)
                }
                Spacer(minLength: 8)
                if rendering { ProgressView().controlSize(.small) }
                Button("Copy image") { export(save: false) }.disabled(rendering)
                Button("Save PNG…") { export(save: true) }.buttonStyle(.borderedProminent).disabled(rendering)
            }
        }
        .padding(16).frame(width: 570)
        .controlSize(.small)
        .onChange(of: options) { notice = nil }
        .task(id: options) {
            do { try await Task.sleep(nanoseconds: 180_000_000) } catch { return }
            guard !Task.isCancelled else { return }
            renderPreview()
        }
    }

    private var exportSize: CGSize {
        CGSize(width: options.outputWidth, height: max(1, Int((CGFloat(options.outputWidth) / aspectRatio).rounded())))
    }

    @MainActor private func renderPreview() {
        do {
            preview = try snapshot(options, CGSize(width: 720, height: max(1, (720 / aspectRatio).rounded())))
            error = nil
        } catch { self.error = error.localizedDescription }
    }

    @MainActor private func export(save: Bool) {
        var destination: URL?
        if save {
            let panel = NSSavePanel()
            panel.allowedContentTypes = [.png]
            panel.canCreateDirectories = true
            panel.isExtensionHidden = false
            panel.nameFieldStringValue = suggestedName + ".png"
            panel.title = "Save plot image"
            guard panel.runModal() == .OK, let url = panel.url else { return }
            destination = url
        }
        rendering = true; error = nil; notice = nil
        Task { @MainActor in
            await Task.yield()
            defer { rendering = false }
            do {
                let image = try snapshot(options, exportSize)
                let data = try PlotImageExport.pngData(image)
                if let destination {
                    try data.write(to: destination, options: .atomic)
                    notice = "Saved \(destination.lastPathComponent)"
                } else {
                    let pasteboard = NSPasteboard.general
                    pasteboard.clearContents()
                    guard pasteboard.setData(data, forType: .png) else {
                        throw NSError(domain: "PlotImageExport", code: 1,
                                      userInfo: [NSLocalizedDescriptionKey: "The image could not be copied to the clipboard."])
                    }
                    notice = "Image copied"
                }
            } catch { self.error = error.localizedDescription }
        }
    }

    private var suggestedName: String {
        let base = options.title.trimmingCharacters(in: .whitespacesAndNewlines)
        let name = (base.isEmpty ? title : base).components(separatedBy: CharacterSet(charactersIn: "/:\\").union(.newlines)).joined(separator: "-")
        return name.isEmpty ? "Organoid Atlas plot" : String(name.prefix(150))
    }
}

private struct PlotTransparencyPreview: View {
    var body: some View {
        Canvas { context, size in
            let tile: CGFloat = 10
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(Color(white: 0.94)))
            for row in 0..<Int(ceil(size.height / tile)) {
                for column in 0..<Int(ceil(size.width / tile)) where (row + column).isMultiple(of: 2) {
                    context.fill(Path(CGRect(x: CGFloat(column) * tile, y: CGFloat(row) * tile, width: tile, height: tile)),
                                 with: .color(Color(white: 0.84)))
                }
            }
        }.accessibilityHidden(true)
    }
}
