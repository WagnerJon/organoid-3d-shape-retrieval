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
