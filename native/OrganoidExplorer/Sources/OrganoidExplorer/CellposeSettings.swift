import SwiftUI

struct CellposeSettings: Codable, Equatable {
    static let models = ["cpsam_v2","cpdino","cpdino-vitb"]
    var model = "cpsam_v2", scale = 0.5, diameter = 30.0, background = true, gpu = true
    func json() throws -> [String:Any] { try JSONSerialization.jsonObject(with:JSONEncoder().encode(self)) as! [String:Any] }
    static func fromJSON(_ settings:[String:Any]) throws -> CellposeSettings {
        var merged = try CellposeSettings().json()
        for (key,value) in settings where merged[key] != nil { merged[key] = value }
        if let background = settings["background_correction"] { merged["background"] = background }
        let value = try JSONDecoder().decode(CellposeSettings.self,from:JSONSerialization.data(withJSONObject:merged))
        guard models.contains(value.model) else { throw DataError.invalid("Unknown Cellpose model") }
        return value
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
            Toggle("Correct background",isOn:settings.background)
            Toggle("GPU when available",isOn:settings.gpu)
            Text("XY / XZ / YZ → u-Segment3D aggregation").font(.caption2).foregroundStyle(.secondary)
            Text("Model weights download on first use.").font(.caption2).foregroundStyle(.secondary)
            HStack {
                Button("Run Cellpose \(side == 1 ? "A" : "B")") { store.run([side == 1 ? "cellpose" : "cellposeB"]) }.disabled(!store.canRun)
                Button(side == 1 ? "Load existing…" : "Import labels…") { store.importLabels(method:"cellpose",existing:side == 1,side:side) }.disabled(!store.canRun)
            }
        }.frame(maxWidth:.infinity,alignment:.leading).padding(6)
    }
}
