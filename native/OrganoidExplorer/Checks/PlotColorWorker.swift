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
        for palette in PlotPalette.allCases {
            for fraction in [-1.0,0,0.5,1,2] {
                let c = palette.color(fraction).usingColorSpace(.sRGB)!
                require([c.redComponent,c.greenComponent,c.blueComponent].allSatisfy { $0.isFinite && (0...1).contains($0) }, "Palettes clamp finite display colors")
            }
        }
        let dataset = try JSONDecoder().decode(Dataset.self,from:Data(contentsOf:URL(fileURLWithPath:CommandLine.arguments[1]).appendingPathComponent("dataset.json")))
        let realKeys = plotFeatureKeys(dataset.cells)
        require(Set(dataset.cells.flatMap { $0.metrics.keys }).isSubset(of:Set(realKeys)), "Every real dataset feature is selectable")
        print("PASS: arbitrary features; shared domains; missing/nonfinite/constant values; palette endpoints; per-method and per-level isolation; organoid cell count; real dataset coverage.")
    }
}
