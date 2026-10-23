import SwiftUI
import AppKit

struct FocusSettings: Codable, Equatable {
    var checkpoint = "", config = "", device = "auto"
    var cell_radius = 15.0
    var automatic_background = true, background_threshold = 0.0
    var min_size = 0, max_size = 0
    var lower_percentile = 1.0, upper_percentile = 99.0
    var batch_size = 1, stride_z = 24, stride_y = 64, stride_x = 64
    var score_threshold = 0.5, mask_threshold = 0.5
    var min_edge_area = 20, stitch_iom = 0.2, stitch_core = 64, stitch_contact = 10, stitch_ratio = 0.5
    func json() throws -> [String:Any] {
        try JSONSerialization.jsonObject(with:JSONEncoder().encode(self)) as! [String:Any]
    }
    static func fromJSON(_ json: [String:Any]) throws -> FocusSettings {
        var merged = try FocusSettings().json()
        for (key,value) in json where merged[key] != nil { merged[key] = value }
        return try JSONDecoder().decode(FocusSettings.self,from:JSONSerialization.data(withJSONObject:merged))
    }
    var valid: Bool {
        let finite = [cell_radius, background_threshold, lower_percentile, upper_percentile, score_threshold, mask_threshold, stitch_iom, stitch_ratio].allSatisfy(\.isFinite)
        return finite && (1...1000).contains(cell_radius) && lower_percentile >= 0 && upper_percentile <= 100 && lower_percentile < upper_percentile &&
            min_size >= 0 && max_size >= 0 && (max_size == 0 || max_size >= min_size) && (1...16).contains(batch_size) &&
            (1..<32).contains(stride_z) && (1..<96).contains(stride_y) && (1..<96).contains(stride_x) &&
            [score_threshold, mask_threshold, stitch_iom, stitch_ratio].allSatisfy { (0...1).contains($0) } &&
            min_edge_area >= 0 && stitch_core >= 1 && stitch_contact >= 1 && [min_size,max_size,min_edge_area,stitch_core,stitch_contact].allSatisfy { $0 <= Int(Int32.max) } &&
            ["auto","cpu","mps","cuda:0"].contains(device)
    }
}

/// Same upstream parameters as the comparison workflow, bound to a preparation.
struct FocusPreparationControls: View {
    @Binding var settings: FocusSettings
    @State private var advanced = false
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(settings.checkpoint.isEmpty ? "Choose a FOCUS-3D checkpoint" : URL(fileURLWithPath: settings.checkpoint).lastPathComponent)
                    .font(.caption).lineLimit(1).help(settings.checkpoint)
                Button("Choose checkpoint…") { chooseFile(config: false) }
            }
            HStack {
                Text("Cell radius"); TextField("Cell radius", value: $settings.cell_radius, format: .number).frame(width: 70)
                Text("pixels in the original XY image").font(.caption).foregroundStyle(.secondary)
            }
            Text("Z/XY ratio comes from each stack's calibration. X and Y spacing must match. Labels are restored to the original grid before measurements; Cellpose scale and diameter settings do not apply.")
                .font(.caption).foregroundStyle(.secondary)
            Toggle("Automatic background (5th percentile)", isOn: $settings.automatic_background)
            if !settings.automatic_background { number("Background intensity", value: $settings.background_threshold) }
            HStack { integer("Min size", value: $settings.min_size); integer("Max size", value: $settings.max_size) }
            Text("Sizes are in resampled inference voxels; max 0 means unlimited.").font(.caption).foregroundStyle(.secondary)
            HStack { number("Detection score threshold", value: $settings.score_threshold); number("Mask threshold", value: $settings.mask_threshold) }
            Picker("Device", selection: $settings.device) {
                Text("Auto (GPU preferred)").tag("auto"); Text("Apple GPU (MPS)").tag("mps"); Text("CPU").tag("cpu")
            }.frame(maxWidth: 420)
            DisclosureGroup("Advanced FOCUS-3D settings", isExpanded: $advanced) {
                VStack(alignment: .leading, spacing: 8) {
                    HStack { number("Low percentile", value: $settings.lower_percentile); number("High percentile", value: $settings.upper_percentile) }
                    integer("Batch size", value: $settings.batch_size)
                    Text("Batch 1 uses less memory. Model patch size is fixed at 32 × 96 × 96 (ZYX).").font(.caption).foregroundStyle(.secondary)
                    HStack { integer("Stride Z", value: $settings.stride_z); integer("Y", value: $settings.stride_y); integer("X", value: $settings.stride_x) }
                    Text("Smaller strides increase patch overlap and processing time. Valid ranges: Z 1–31, Y/X 1–95.").font(.caption).foregroundStyle(.secondary)
                    integer("Minimum edge area", value: $settings.min_edge_area)
                    number("Stitch face IoM", value: $settings.stitch_iom)
                    integer("Minimum core voxels", value: $settings.stitch_core)
                    integer("Minimum contact pixels", value: $settings.stitch_contact)
                    number("Contact component ratio", value: $settings.stitch_ratio)
                    HStack {
                        Text(settings.config.isEmpty ? "Official model configuration" : URL(fileURLWithPath: settings.config).lastPathComponent).font(.caption).lineLimit(1)
                        Button("Choose YAML…") { chooseFile(config: true) }
                        if !settings.config.isEmpty { Button("Reset") { settings.config = "" } }
                    }
                }.padding(.top, 6)
            }
            if !settings.valid { Text("Review the parameter ranges. Thresholds must be 0–1, percentiles must increase, batch size 1–16, and sizes cannot be negative.").font(.caption).foregroundStyle(.orange) }
        }
    }
    private func chooseFile(config: Bool) {
        let panel = NSOpenPanel(); panel.message = config ? "Choose a FOCUS-3D model YAML configuration" : "Choose downloaded FOCUS-3D weights (.pth)"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        if config { settings.config = url.path }
        else { settings.checkpoint = url.path; UserDefaults.standard.set(url.path, forKey: "focus3dCheckpoint") }
    }
    private func number(_ title: String, value: Binding<Double>) -> some View {
        HStack { Text(title); TextField(title, value: value, format: .number).frame(width: 70) }
    }
    private func integer(_ title: String, value: Binding<Int>) -> some View {
        HStack { Text(title); TextField(title, value: value, format: .number).frame(width: 70) }
    }
}

struct FocusControls: View {
    @EnvironmentObject var store: SegmentationStore
    @State private var advanced = false
    var body: some View {
        VStack(alignment:.leading,spacing:8) {
            HStack {
                Text(store.focusSettings.checkpoint.isEmpty ? "No checkpoint selected" : URL(fileURLWithPath:store.focusSettings.checkpoint).lastPathComponent)
                    .font(.caption).lineLimit(1).help(store.focusSettings.checkpoint)
                Spacer(); Button("Choose checkpoint…") { store.openFocusFile() }
            }
            if !store.focusCheckpointAvailable {
                Link("Get official model weights ↗",destination:URL(string:"https://huggingface.co/Qinghua-thu/FOCUS-3D")!).font(.caption)
                Text("Request access on Hugging Face, download the appropriate .pth model, then select it here.").font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal:false,vertical:true)
            }
            HStack { Text("Cell radius"); TextField("Radius",value:$store.focusSettings.cell_radius,format:.number).frame(width:55); Text("raw XY px").font(.caption2).foregroundStyle(.secondary) }
            Text("Z/XY ratio from stack calibration: \(store.spacingZ / store.spacingY,specifier:"%.3f")").font(.caption2).foregroundStyle(.secondary)
            Toggle("Automatic background (5th percentile)",isOn:$store.focusSettings.automatic_background).font(.caption)
            if !store.focusSettings.automatic_background { number("Background intensity",value:$store.focusSettings.background_threshold) }
            HStack { integer("Min size",value:$store.focusSettings.min_size); integer("Max size",value:$store.focusSettings.max_size) }
            Text("Size in resampled inference voxels · max 0 = unlimited").font(.caption2).foregroundStyle(.secondary)
            DisclosureGroup("Advanced inference & stitching",isExpanded:$advanced) {
                VStack(alignment:.leading,spacing:8) {
                    Picker("Device",selection:$store.focusSettings.device) { Text("Auto (GPU preferred)").tag("auto"); Text("Apple GPU (MPS)").tag("mps"); Text("CPU").tag("cpu"); Text("CUDA 0").tag("cuda:0") }
                    Text("Auto prefers CUDA, then Apple MPS, then CPU. Batch size 1 uses less memory.").font(.caption2).foregroundStyle(.secondary)
                    HStack { number("Low percentile",value:$store.focusSettings.lower_percentile); number("High",value:$store.focusSettings.upper_percentile) }
                    integer("Batch size",value:$store.focusSettings.batch_size)
                    Text("Fixed model patch: 32 × 96 × 96 (ZYX)").font(.caption2).foregroundStyle(.secondary)
                    HStack { integer("Stride Z",value:$store.focusSettings.stride_z); integer("Y",value:$store.focusSettings.stride_y); integer("X",value:$store.focusSettings.stride_x) }
                    HStack { number("Score threshold",value:$store.focusSettings.score_threshold); number("Mask",value:$store.focusSettings.mask_threshold) }
                    integer("Minimum edge area",value:$store.focusSettings.min_edge_area)
                    number("Stitch face IoM",value:$store.focusSettings.stitch_iom)
                    integer("Minimum core voxels",value:$store.focusSettings.stitch_core)
                    integer("Minimum contact pixels",value:$store.focusSettings.stitch_contact)
                    number("Contact component ratio",value:$store.focusSettings.stitch_ratio)
                    HStack { Text(store.focusSettings.config.isEmpty ? "Official model configuration" : URL(fileURLWithPath:store.focusSettings.config).lastPathComponent).font(.caption).lineLimit(1); Button("Choose YAML…") { store.openFocusFile(config:true) }; if !store.focusSettings.config.isEmpty { Button("Reset") {store.focusSettings.config=""} } }
                }.padding(.top,6)
            }.font(.caption)
            HStack {
                Button("Run FOCUS-3D") { store.run(["focus3d"]) }.disabled(!store.canRun || !store.focusReady)
                Button("Import labels…") { store.importLabels(method:"focus3d") }.disabled(!store.canRun)
            }
            if !store.focusInstalled { Text("Install with zsh scripts/setup_focus3d.sh").font(.caption).foregroundStyle(.orange) }
        }.frame(maxWidth:.infinity,alignment:.leading).padding(6)
    }
    func number(_ title:String,value:Binding<Double>) -> some View { HStack { Text(title); TextField(title,value:value,format:.number).frame(width:55) } }
    func integer(_ title:String,value:Binding<Int>) -> some View { HStack { Text(title); TextField(title,value:value,format:.number).frame(width:55) } }
}
