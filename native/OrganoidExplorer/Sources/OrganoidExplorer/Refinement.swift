import SwiftUI

struct RefinementSettings: Codable {
    var iterations = 30
    var membraneWeight = 1.0, shapeWeight = 0.5, smoothness = 0.5, ellipsoidWeight = 0.0
    var maxDisplacement = 0.5, controlSpacing = 2.0, psfZ = 0.7, psfXY = 0.25
    var learningRate = 0.03, maxVolumeChange = 0.1
    var maxRoughnessIncrease = 0.05, maxChangedFraction = 0.1
    var device = "auto"
    var preserveConnectivity = true
    var preserveTopology = true
    func json() throws -> [String:Any] { try JSONSerialization.jsonObject(with:JSONEncoder().encode(self)) as! [String:Any] }
    static func fromJSON(_ settings:[String:Any]) throws -> RefinementSettings {
        var merged = try RefinementSettings().json()
        for (key,value) in settings where merged[key] != nil { merged[key] = value }
        return try JSONDecoder().decode(RefinementSettings.self,from:JSONSerialization.data(withJSONObject:merged))
    }
}
struct RefinementInfo: Codable {
    let parentID, parentFolder, parentMethod: String
    let changedVoxels: Int
    let initialLoss, finalLoss: Double
    let rejectedCells: [Int]
    let algorithmVersion: Int?
}
struct RefinementControls: View {
    @EnvironmentObject var store: SegmentationStore
    var body: some View {
            VStack(alignment:.leading,spacing:12) {
                Text("Membrane-guided refinement").font(.headline)
                Text("Make small, smooth adjustments to the original shapes. Unsafe changes keep the original cell.").font(.caption).foregroundStyle(.secondary)
                GroupBox("Fitting & shape") { VStack(spacing:8) {
                    number("Membrane fit",value:$store.refinementSettings.membraneWeight)
                    number("Shape preservation",value:$store.refinementSettings.shapeWeight)
                    number("Smoothness",value:$store.refinementSettings.smoothness)
                    number("Ellipsoid prior",value:$store.refinementSettings.ellipsoidWeight)
                }.padding(4) }
                GroupBox("Movement & optical blur") { VStack(spacing:8) {
                    number("Max movement µm",value:$store.refinementSettings.maxDisplacement)
                    number("Control spacing µm",value:$store.refinementSettings.controlSpacing)
                    number("PSF σ Z µm",value:$store.refinementSettings.psfZ)
                    number("PSF σ XY µm",value:$store.refinementSettings.psfXY)
                }.padding(4) }
                GroupBox("Safety") { VStack(alignment:.leading,spacing:8) {
                    number("Max volume change",value:$store.refinementSettings.maxVolumeChange)
                    number("Max changed fraction",value:$store.refinementSettings.maxChangedFraction)
                    number("Max roughness increase",value:$store.refinementSettings.maxRoughnessIncrease)
                    Toggle("Prevent new fragmentation",isOn:$store.refinementSettings.preserveConnectivity).font(.caption)
                    Toggle("Prevent new holes / handles",isOn:$store.refinementSettings.preserveTopology).font(.caption)
                }.padding(4) }
                GroupBox("Optimization") { VStack(spacing:8) {
                    integer("Iterations",value:$store.refinementSettings.iterations)
                    number("Learning rate",value:$store.refinementSettings.learningRate)
                    Picker("Device",selection:$store.refinementSettings.device) { Text("Auto").tag("auto"); Text("Apple GPU").tag("mps"); Text("CPU").tag("cpu") }
                }.padding(4) }
                Text("Use membrane-stained raw data. PSF values are estimates. Ellipsoid prior 0 preserves irregular shapes; fraction 0.1 means 10%. Already good segmentations may stay unchanged. This does not repair cell identities.").font(.caption2).foregroundStyle(.secondary)
                Text("The images show saved results. Edits to these settings apply to your next run.").font(.caption2).foregroundStyle(.secondary)
            }.textFieldStyle(.roundedBorder).disabled(store.busy)
    }
    func number(_ label:String,value:Binding<Double>) -> some View { HStack { Text(label).font(.caption);Spacer();TextField(label,value:value,format:.number).frame(width:65) } }
    func integer(_ label:String,value:Binding<Int>) -> some View { HStack { Text(label).font(.caption);Spacer();TextField(label,value:value,format:.number).frame(width:65) } }
}
