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

struct DatasetSliceTestView: View {
    @ObservedObject var preparation: DatasetPreparationStore
    @State private var selectedPath = ""
    @State private var plane = "xy"
    @State private var slice = 0.0
    @State private var overlay = true
    @State private var opacity = 0.8
    var input: PreparationInput? { preparation.inputs.first { $0.id == selectedPath } }
    var axis: Int { plane == "xy" ? 0 : plane == "xz" ? 1 : 2 }
    var count: Int { input?.shape[axis] ?? 1 }
    var index: Int { min(max(0, Int(slice.rounded())), max(0, count-1)) }
    var canPreview: Bool { input?.spacing.allSatisfy { $0.isFinite && $0 > 0 } == true }
    var canTest: Bool { canPreview && preparation.settings.scale.isFinite && (0.05...1).contains(preparation.settings.scale) && preparation.settings.diameter.isFinite && (2...500).contains(preparation.settings.diameter) }
    var stale: Bool {
        guard let result = preparation.sliceResult, result.cells != nil else { return false }
        return result.settings != preparation.settings || result.spacing != input?.spacing
    }
    var body: some View {
        GroupBox("Try settings on one slice") {
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
                        Button("Test segmentation on this slice") { if let input { preparation.previewSlice(input: input, plane: plane, index: index, segment: true) } }.disabled(!canTest)
                        Spacer()
                        if preparation.sliceLabels != nil {
                            Toggle("Overlay", isOn: $overlay)
                            Slider(value: $opacity, in: 0...1).frame(width: 90).accessibilityLabel("Cell label opacity")
                        }
                    }
                    if !canPreview { Text("Enter valid voxel spacing above before loading a slice.").font(.caption).foregroundStyle(.orange) }
                    if let raw = preparation.sliceRaw {
                        HStack(spacing: 10) {
                            preview(title: "Raw microscopy", raw: raw, labels: nil)
                            if preparation.sliceLabels != nil {
                                preview(title: "Cellpose · selected slice", raw: raw, labels: overlay ? preparation.sliceLabels : nil)
                            } else {
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
                    Text("Uses the settings above. This is a 2D Cellpose test with slice-local preprocessing; full-volume preprocessing and u-Segment3D linking can produce different results. The test does not create or change a dataset.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }.padding(6)
        }
        .onChange(of: selectedPath) { centerSlice() }
        .onChange(of: plane) { centerSlice() }
        .onChange(of: index) { preparation.clearSlice() }
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
            }.frame(height: 220).clipShape(RoundedRectangle(cornerRadius: 6))
        }.frame(maxWidth: .infinity)
    }
}
