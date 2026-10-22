import Foundation
import AppKit

@main struct PlotColorChecks {
    @MainActor static func main() throws {
        func require(_ condition: @autoclosure () -> Bool, _ message: String) { if !condition() { fatalError(message) } }
        func item(_ id: String, _ metrics: [String:Double], count: Int? = nil) -> Specimen {
            Specimen(id:id, sample:"sample", split:"train", flags:"", label:1, qc:true, metrics:metrics, representations:[:], cellCount:count)
        }
        func close(_ a: NSColor, _ b: NSColor) -> Bool {
            let x = a.usingColorSpace(.sRGB)!, y = b.usingColorSpace(.sRGB)!
            return abs(x.redComponent-y.redComponent)+abs(x.greenComponent-y.greenComponent)+abs(x.blueComponent-y.blueComponent) < 1e-6
        }
        let cells = [item("a",["volume_um3":1,"custom_descriptor":7,"constant":4]), item("b",["volume_um3":3,"constant":4]), item("missing",["volume_um3":.nan])]
        let keys = plotFeatureKeys(cells)
        require(keys.contains("custom_descriptor") && keys.contains("constant"), "Feature choices include dataset-specific measurements")
        let settings = PlotColorSettings(feature:"volume_um3",palette:.viridis)
        let map = PlotColorMap(settings:settings,items:cells)
        require(map.minimum == 1 && map.maximum == 3 && map.missing == 1, "Nonfinite and missing measurements do not distort the color domain")
        require(close(map.color(item:cells[0],group:5),PlotPalette.viridis.color(0)), "Small values use the lower palette endpoint")
        require(close(map.color(item:cells[1],group:0),PlotPalette.viridis.color(1)), "Large values use the upper endpoint regardless of cluster")
        require(close(map.color(item:cells[2],group:0),.gray), "Unavailable values render gray")
        let constant = PlotColorMap(settings:PlotColorSettings(feature:"constant",palette:.magma),items:cells)
        require(close(constant.color(item:cells[0],group:0),PlotPalette.magma.color(0.5)), "Constant features use a defined midpoint")
        let unavailable = PlotColorMap(settings:PlotColorSettings(feature:"unknown",palette:.viridis),items:cells)
        require(unavailable.minimum == nil && close(unavailable.color(item:cells[0],group:2),.gray), "Entirely missing features remain recoverable")
        let stack = item("stack",[:],count:14)
        require(plotFeatureKeys([stack]).contains("@cellCount") && plotFeatureValue(stack,key:"@cellCount") == 14, "Organoid plots can color by cell count")
        let store = ExplorerStore()
        store.plotColorBinding("method-A").wrappedValue = settings
        require(store.plotColors("method-A") == settings && store.plotColors("method-B") == PlotColorSettings(), "Changing one method leaves another plot untouched")
        store.level = "Organoids"
        require(store.plotColors("method-A") == PlotColorSettings(), "Cell and organoid plot settings are independent")
        store.level = "Cells"
        require(store.plotColors("method-A") == settings, "Returning to a plot restores its settings")
        let individualB = PlotColorSettings(feature:"constant",palette:.magma)
        store.plotColorBinding("method-B").wrappedValue = individualB
        let originalAnalysis = (store.method, store.referenceMethod, store.selection, store.clusterFilter)
        require(!store.unifiedPlotColors, "Unified colors are optional and initially disabled")
        store.setUnifiedPlotColors(true, seedMethod:"method-A")
        require(store.unifiedPlotColors && store.plotColors("method-B") == settings,
                "Enabling unified colors starts from the reference plot and reaches every method")
        let shared = PlotColorSettings(feature:"custom_descriptor",palette:.turbo)
        store.plotColorBinding("method-B").wrappedValue = shared
        require(store.plotColors("method-A") == shared && store.plotColors("method-not-yet-opened") == shared,
                "Editing any unified plot updates open and future method plots")
        store.unifiedPlotColorBinding().wrappedValue = settings
        require(store.plotColors("method-B") == settings, "The comparison control edits the same settings as expanded plots and export")
        store.level = "Organoids"
        require(!store.unifiedPlotColors && store.plotColors("method-A") == PlotColorSettings(),
                "Cell and organoid unified states are independent")
        store.setUnifiedPlotColors(true, seedMethod:"method-A")
        let organoidColors = PlotColorSettings(feature:"@cellCount",palette:.inferno)
        store.unifiedPlotColorBinding().wrappedValue = organoidColors
        store.level = "Cells"
        require(store.unifiedPlotColors && store.plotColors("method-A") == settings,
                "Returning to cells restores their unified feature and palette")
        store.setUnifiedPlotColors(false, seedMethod:"method-A")
        require(store.plotColors("method-A") == settings && store.plotColors("method-B") == individualB,
                "Turning off unified colors restores each previous per-plot preference")
        store.setUnifiedPlotColors(true, seedMethod:"method-B")
        require(store.plotColors("method-A") == settings,
                "Re-enabling unified colors retains the shared choice rather than replacing it with another plot")
        store.level = "Organoids"
        require(store.unifiedPlotColors && store.plotColors("method-A") == organoidColors,
                "Organoid shared settings survive cell-level edits")
        store.level = "Cells"
        require(store.method == originalAnalysis.0 && store.referenceMethod == originalAnalysis.1 && store.selection == originalAnalysis.2 && store.clusterFilter == originalAnalysis.3,
                "Color controls do not change analysis, selection or grouping")
        let rep = Representation(vector:[0],point:[0,0],cluster:0)
        let cohort = [Specimen(id:"both",sample:"s",split:"train",flags:"",label:1,qc:true,
                              metrics:["volume_um3":1],representations:["method-A":rep,"method-B":rep],cellCount:nil),
                      Specimen(id:"only-A",sample:"s",split:"train",flags:"",label:2,qc:true,
                               metrics:["volume_um3":3],representations:["method-A":rep],cellCount:nil),
                      Specimen(id:"no-value",sample:"s",split:"train",flags:"",label:3,qc:true,
                               metrics:[:],representations:["method-B":rep],cellCount:nil)]
        store.dataset = Dataset(version:1,title:"colors",provenance:"test",methods:[],samples:[],cells:cohort,organoids:[])
        let partial = cohort.filter { $0.representations["method-B"] != nil }
        require(store.plotColorDomain(partial).count == cohort.count,
                "Expanded and partial-cohort plots use the same filtered color range while unified")
        let methodAColors = PlotColorMap(settings:store.plotColors("method-A"),items:store.plotColorDomain(cohort))
        let methodBColors = PlotColorMap(settings:store.plotColors("method-B"),items:store.plotColorDomain(partial))
        require(methodAColors.minimum == methodBColors.minimum && methodAColors.maximum == methodBColors.maximum && methodBColors.maximum == 3,
                "Shared color domains include the comparison cohort even when a method has missing representations")
        require(close(methodAColors.color(item:cohort[0],group:0),methodBColors.color(item:cohort[0],group:5)),
                "The same cell has identical feature colors across methods, independent of clustering")
        require(close(methodAColors.color(item:cohort[2],group:0),.gray) && close(methodBColors.color(item:cohort[2],group:5),.gray),
                "Missing feature values remain gray in every shared plot")
        store.setUnifiedPlotColors(false, seedMethod:"method-A")
        require(store.plotColorDomain(partial).count == partial.count,
                "Turning off shared colors restores each plot’s original cohort domain")
        for palette in PlotPalette.allCases {
            for fraction in [-1.0,0,0.5,1,2] {
                let c = palette.color(fraction).usingColorSpace(.sRGB)!
                require([c.redComponent,c.greenComponent,c.blueComponent].allSatisfy { $0.isFinite && (0...1).contains($0) }, "Palettes clamp finite display colors")
            }
        }
        let dataset = try JSONDecoder().decode(Dataset.self,from:Data(contentsOf:URL(fileURLWithPath:CommandLine.arguments[1]).appendingPathComponent("dataset.json")))
        let realKeys = plotFeatureKeys(dataset.cells)
        require(Set(dataset.cells.flatMap { $0.metrics.keys }).isSubset(of:Set(realKeys)), "Every real dataset feature is selectable")
        print("PASS: arbitrary features; shared domains and palettes; missing/nonfinite/constant values; unified edits; per-plot restoration; per-level isolation; unchanged analyses; organoid cell count; real dataset coverage.")
    }
}
