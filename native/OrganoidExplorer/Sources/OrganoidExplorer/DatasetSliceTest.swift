import SwiftUI
import AppKit

struct PreparationSliceResult: Decodable {
    let path, plane, note: String
    let index: Int
    let nativeShape: [Int]
    let spacing: [Double]
    let settings: CellposeSettings?
    let cells: Int?
    let device: String?
    let seconds: Double?
}

enum PreparationDiameterGuide {
    static func imageRect(image: CGSize, canvas: CGSize) -> CGRect {
        guard image.width > 0, image.height > 0 else { return .zero }
        let factor = min(canvas.width / image.width, canvas.height / image.height)
        let size = CGSize(width: image.width * factor, height: image.height * factor)
        return CGRect(x: (canvas.width-size.width)/2, y: (canvas.height-size.height)/2, width: size.width, height: size.height)
    }
    static func displayDiameter(result: PreparationSliceResult, imageRect: CGRect, scale: Double, diameter: Double) -> CGFloat? {
        guard result.nativeShape.count == 2, result.spacing.count == 3,
              result.nativeShape.allSatisfy({ $0 > 0 }), result.spacing.allSatisfy({ $0.isFinite && $0 > 0 }),
              scale.isFinite, (0.05...1).contains(scale), diameter.isFinite, (2...500).contains(diameter) else { return nil }
        let widthAxis: Int
        switch result.plane { case "xy", "xz": widthAxis = 2; case "yz": widthAxis = 1; default: return nil }
        // The preview has physical aspect ratio; Cellpose's grid uses scale * spacing/min(spacing).
        let physicalDiameter = diameter * result.spacing.min()! / scale
        let physicalWidth = Double(result.nativeShape[1]) * result.spacing[widthAxis]
        return physicalDiameter / physicalWidth * imageRect.width
    }
}

struct DatasetSliceTestView: View {
    @ObservedObject var preparation: DatasetPreparationStore
    @State private var selectedPath = ""
    @State private var plane = "xy"
    @State private var slice = 0.0
    @State private var overlay = true
    @State private var opacity = 0.8
    @State private var showDiameter = false
    var input: PreparationInput? { preparation.inputs.first { $0.id == selectedPath } }
    var axis: Int { plane == "xy" ? 0 : plane == "xz" ? 1 : 2 }
    var count: Int { input?.shape[axis] ?? 1 }
    var index: Int { min(max(0, Int(slice.rounded())), max(0, count-1)) }
    var canPreview: Bool { input?.spacing.allSatisfy { $0.isFinite && $0 > 0 } == true }
    var canTest: Bool { preparation.settings.validModel && canPreview && preparation.settings.scale.isFinite && (0.05...1).contains(preparation.settings.scale) && preparation.settings.diameter.isFinite && (2...500).contains(preparation.settings.diameter) && preparation.settings.flowThreshold.isFinite && preparation.settings.flowThreshold >= 0 }
    var stale: Bool {
        guard let result = preparation.sliceResult, result.cells != nil else { return false }
        return result.settings != preparation.settings || result.spacing != input?.spacing
    }
    var body: some View {
        GroupBox(preparation.usesFocus ? "Inspect the raw stack" : "Try settings on one slice") {
            VStack(alignment: .leading, spacing: 8) {
                if preparation.inputs.isEmpty {
                    Text("Add a TIFF, NumPy or OME-Zarr volume above to test segmentation before preparing the dataset.").font(.caption).foregroundStyle(.secondary)
                } else {
                    HStack {
                        Picker("Stack", selection: $selectedPath) {
                            ForEach(preparation.inputs) { input in Text(input.name).tag(input.id) }
                        }.frame(maxWidth: 350)
                        Picker("Plane", selection: $plane) {
                            Text("XY").tag("xy"); Text("XZ").tag("xz"); Text("YZ").tag("yz")
                        }.pickerStyle(.segmented).frame(width: 160)
                    }
                    HStack {
                        Text("Slice \(index+1)/\(count)").font(.caption.monospacedDigit()).frame(width: 100, alignment: .leading)
                        Slider(value: $slice, in: 0...Double(max(1, count-1)), step: 1)
                        Stepper("Slice", value: $slice, in: 0...Double(max(1, count-1)), step: 1).labelsHidden()
                    }
                    HStack {
                        Button("Load raw slice") { if let input { preparation.previewSlice(input: input, plane: plane, index: index, segment: false) } }.disabled(!canPreview)
                        if !preparation.usesFocus {
                            Button("Test segmentation on this slice") { if let input { preparation.previewSlice(input: input, plane: plane, index: index, segment: true) } }.disabled(!canTest)
                        }
                        Spacer()
                        if preparation.sliceLabels != nil {
                            Toggle("Overlay", isOn: $overlay)
                            Slider(value: $opacity, in: 0...1).frame(width: 90).accessibilityLabel("Cell label opacity")
                        }
                    }
                    if !preparation.usesFocus {
                    HStack {
                        Toggle("Show diameter guide", isOn: $showDiameter).toggleStyle(.checkbox)
                        Text("Purple disc · \(preparation.settings.diameter, specifier: "%.1f") processed px at the current scale")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    HStack {
                        Text("Flow threshold (slice test)")
                        TextField("Flow threshold", value: $preparation.settings.flowThreshold, format: .number).frame(width: 65)
                    }
                    Text("Default 0.4. Lower values reject more inconsistent 2D masks; 0 disables this filter. Full-volume u-Segment3D linking uses raw flows, so this setting only affects the slice test.")
                        .font(.caption).foregroundStyle(.secondary)
                    if !preparation.settings.flowThreshold.isFinite || preparation.settings.flowThreshold < 0 {
                        Text("Flow threshold must be a finite number of 0 or greater.").font(.caption).foregroundStyle(.orange)
                    }
                    }
                    if !canPreview { Text("Enter valid voxel spacing above before loading a slice.").font(.caption).foregroundStyle(.orange) }
                    if let raw = preparation.sliceRaw {
                        HStack(spacing: 10) {
                            preview(title: "Raw microscopy", raw: raw, labels: nil)
                            if preparation.sliceLabels != nil {
                                preview(title: "Cellpose · selected slice", raw: raw, labels: overlay ? preparation.sliceLabels : nil)
                            } else if !preparation.usesFocus {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text("Cellpose · selected slice").font(.caption.bold())
                                    Text("Run the slice test to see detected cells here.").font(.caption).foregroundStyle(.secondary)
                                        .frame(maxWidth: .infinity, minHeight: 220).background(.black, in: RoundedRectangle(cornerRadius: 6))
                                }.frame(maxWidth: .infinity)
                            }
                        }
                    }
                    if let result = preparation.sliceResult, let cells = result.cells {
                        Text("\(cells) cells detected · \(result.settings?.model ?? "Cellpose") · \(result.device ?? "") · \(String(format: "%.1f", result.seconds ?? 0)) s")
                            .font(.caption).foregroundStyle(.secondary)
                        if stale { Text("Settings or calibration changed. Run the slice test again to see their effect.").font(.caption).foregroundStyle(.orange) }
                    }
                    Text(preparation.usesFocus ? "FOCUS-3D requires a full 3D stack. Inspect raw slices here, then prepare the dataset with the selected FOCUS-3D settings." : "Uses the settings above. This is a 2D Cellpose test with slice-local preprocessing; full-volume preprocessing and u-Segment3D linking can produce different results. The test does not create or change a dataset.")
                        .font(.caption).foregroundStyle(.secondary)
                    if !preparation.usesFocus && preparation.settings.native3D { Text("Native 3D settings apply when preparing the complete stack. This single-slice test remains 2D and cannot preview 3D mask formation.").font(.caption).foregroundStyle(.orange) }
                }
            }.padding(6)
        }
        .onChange(of: selectedPath) { centerSlice() }
        .onChange(of: plane) { centerSlice() }
        .onChange(of: index) { preparation.clearSlice() }
        .onChange(of: input?.spacing) { preparation.clearSlice() }
        .onChange(of: preparation.inputs.map(\.id)) { chooseInput() }
        .onAppear { chooseInput() }
    }
    func chooseInput() {
        if !preparation.inputs.contains(where: { $0.id == selectedPath }) {
            selectedPath = preparation.inputs.first?.id ?? ""; centerSlice()
        }
    }
    func centerSlice() { slice = Double(max(0, count/2)); preparation.clearSlice() }
    func preview(title: String, raw: NSImage, labels: NSImage?) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption.bold())
            ZStack {
                Color.black
                Image(nsImage: raw).resizable().interpolation(.high).scaledToFit()
                if let labels { Image(nsImage: labels).resizable().interpolation(.none).scaledToFit().opacity(opacity) }
                if showDiameter && !preparation.usesFocus, let result = preparation.sliceResult {
                    GeometryReader { geometry in
                        let rect = PreparationDiameterGuide.imageRect(image: raw.size, canvas: geometry.size)
                        if let diameter = PreparationDiameterGuide.displayDiameter(result: result, imageRect: rect, scale: preparation.settings.scale, diameter: preparation.settings.diameter) {
                            Circle().fill(Color.purple.opacity(0.35))
                                .overlay(Circle().stroke(Color.purple, lineWidth: 2))
                                .frame(width: diameter, height: diameter)
                                .position(x: min(diameter/2+10, rect.width/2), y: rect.height - min(diameter/2+10, rect.height/2))
                                .frame(width: rect.width, height: rect.height).clipped()
                                .offset(x: rect.minX, y: rect.minY)
                        }
                    }.allowsHitTesting(false).accessibilityLabel("Purple Cellpose diameter guide")
                }
            }.frame(height: 220).clipShape(RoundedRectangle(cornerRadius: 6))
        }.frame(maxWidth: .infinity)
    }
}
