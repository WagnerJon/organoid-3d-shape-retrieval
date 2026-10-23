import SwiftUI

struct CellposeSettings: Codable, Equatable {
    static let models = ["cpsam_v2","cpdino","cpdino-vitb"]
    var model = "cpsam_v2", scale = 0.5, diameter = 30.0, background = true, gpu = true
    var flowThreshold = 0.4
    var segmentationMode = "usegment3d"
    var flow3DSmooth = 0.0, cellprobThreshold = 0.0
    var minSize3D = 15
    var native3D: Bool { segmentationMode == "cellpose3d" }
    var modeName: String { native3D ? "Native Cellpose 3D" : "Cellpose + u-Segment3D" }
    var valid3DSettings: Bool { ["usegment3d","cellpose3d"].contains(segmentationMode) && flow3DSmooth.isFinite && (0...20).contains(flow3DSmooth) && cellprobThreshold.isFinite && (0...10_000_000).contains(minSize3D) }
    func json() throws -> [String:Any] { try JSONSerialization.jsonObject(with:JSONEncoder().encode(self)) as! [String:Any] }
    static func fromJSON(_ settings:[String:Any]) throws -> CellposeSettings {
        var merged = try CellposeSettings().json()
        for (key,value) in settings where merged[key] != nil { merged[key] = value }
        if let background = settings["background_correction"] { merged["background"] = background }
        if let background = settings["backgroundCorrection"] { merged["background"] = background }
        if let diameter = settings["diameterProcessedPx"] { merged["diameter"] = diameter }
        if let threshold = settings["flow_threshold"] { merged["flowThreshold"] = threshold }
        let value = try JSONDecoder().decode(CellposeSettings.self,from:JSONSerialization.data(withJSONObject:merged))
        guard models.contains(value.model) else { throw DataError.invalid("Unknown Cellpose model") }
        return value
    }
}

extension CellposeSettings {
    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        model = try values.decode(String.self, forKey: .model)
        scale = try values.decode(Double.self, forKey: .scale)
        diameter = try values.decode(Double.self, forKey: .diameter)
        background = try values.decode(Bool.self, forKey: .background)
        gpu = try values.decode(Bool.self, forKey: .gpu)
        flowThreshold = try values.decodeIfPresent(Double.self, forKey: .flowThreshold) ?? 0.4
        segmentationMode = try values.decodeIfPresent(String.self, forKey: .segmentationMode) ?? "usegment3d"
        flow3DSmooth = try values.decodeIfPresent(Double.self, forKey: .flow3DSmooth) ?? 0
        cellprobThreshold = try values.decodeIfPresent(Double.self, forKey: .cellprobThreshold) ?? 0
        minSize3D = try values.decodeIfPresent(Int.self, forKey: .minSize3D) ?? 15
    }
}

struct CellposeModeControls: View {
    @Binding var settings: CellposeSettings
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Picker("3D segmentation", selection: $settings.segmentationMode) {
                Text("Cellpose + u-Segment3D").tag("usegment3d")
                Text("Native Cellpose 3D").tag("cellpose3d")
            }
            if settings.native3D {
                HStack {
                    Text("3D flow smoothing")
                    TextField("3D flow smoothing", value: $settings.flow3DSmooth, format: .number).frame(width: 55)
                    Text("px (0–20)").font(.caption).foregroundStyle(.secondary)
                }
                HStack {
                    Text("Cell probability threshold")
                    TextField("Cell probability threshold", value: $settings.cellprobThreshold, format: .number).frame(width: 55)
                }
                HStack {
                    Text("Minimum cell size")
                    TextField("Minimum 3D cell size", value: $settings.minSize3D, format: .number).frame(width: 65)
                    Text("processed voxels").font(.caption).foregroundStyle(.secondary)
                }
                Text("Cellpose predicts all three views and forms masks with native 3D dynamics. Calibration is already applied to the processing grid. Mac GPU prediction uses MPS; 3D masks run on CPU. The 2D flow threshold is not used.")
                    .font(.caption).foregroundStyle(.secondary)
                if !settings.valid3DSettings { Text("Enter valid 3D settings: smoothing 0–20, a finite probability threshold, and a whole-number minimum size of 0–10,000,000.").font(.caption).foregroundStyle(.orange) }
            }
        }
    }
}

struct CellposeControls: View {
    @EnvironmentObject var store: SegmentationStore
    let side: Int
    var settings: Binding<CellposeSettings> { side == 1 ? $store.cellposeA : $store.cellposeB }
    var body: some View {
        VStack(alignment:.leading,spacing:8) {
            Picker("Model",selection:settings.model) { ForEach(CellposeSettings.models,id:\.self) { Text($0).tag($0) } }
            HStack { Text("Scale"); TextField("Scale",value:settings.scale,format:.number).frame(width:55); Text("0.05–1").font(.caption).foregroundStyle(.secondary) }
            HStack { Text("Diameter"); TextField("Diameter",value:settings.diameter,format:.number).frame(width:55); Text("processed px").font(.caption).foregroundStyle(.secondary) }
            CellposeModeControls(settings: settings)
            Toggle("Correct background",isOn:settings.background)
            Toggle("GPU when available",isOn:settings.gpu)
            Text("Model weights download on first use.").font(.caption2).foregroundStyle(.secondary)
            HStack {
                Button("Run Cellpose \(side == 1 ? "A" : "B")") { store.run([side == 1 ? "cellpose" : "cellposeB"]) }.disabled(!store.canRun || !settings.wrappedValue.valid3DSettings)
                Button(side == 1 ? "Load existing…" : "Import labels…") { store.importLabels(method:"cellpose",existing:side == 1,side:side) }.disabled(!store.canRun)
            }
        }.frame(maxWidth:.infinity,alignment:.leading).padding(6)
    }
}
