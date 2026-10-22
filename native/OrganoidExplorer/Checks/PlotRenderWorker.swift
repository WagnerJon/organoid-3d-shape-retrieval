import Foundation
import AppKit
import SwiftUI

@main struct PlotRenderChecks {
    @MainActor static func main() throws {
        func require(_ condition: @autoclosure () -> Bool, _ message: String) {
            if !condition() { fatalError(message) }
        }
        func item(_ id: String, x: Double, y: Double, group: Int, value: Double) -> Specimen {
            Specimen(id:id, sample:"sample", split:"test", flags:"", label:group+1, qc:true,
                     metrics:["sphericity":value], representations:[
                        "shown":Representation(vector:[x,y], point:[x,y], cluster:group),
                        "color":Representation(vector:[y,x], point:[y,x], cluster:5-group)
                     ], cellCount:nil)
        }
        func bitmap(_ image: NSImage) throws -> NSBitmapImageRep {
            guard let result = NSBitmapImageRep(data:try PlotImageExport.pngData(image)) else {
                throw DataError.invalid("Exported PNG could not be read")
            }
            require(result.bitsPerSample == 8 && result.samplesPerPixel >= 3,
                    "Exported PNG uses 8-bit RGB samples")
            require(result.cgImage?.colorSpace?.name == CGColorSpace.sRGB,
                    "Exported PNG declares the sRGB color space")
            return result
        }
        func close(_ a: NSColor, _ b: NSColor, tolerance: CGFloat = 0.025) -> Bool {
            guard let x = a.usingColorSpace(.sRGB), let y = b.usingColorSpace(.sRGB) else { return false }
            return abs(x.redComponent-y.redComponent) < tolerance &&
                abs(x.greenComponent-y.greenComponent) < tolerance &&
                abs(x.blueComponent-y.blueComponent) < tolerance &&
                abs(x.alphaComponent-y.alphaComponent) < tolerance
        }
        func pixel(_ bitmap: NSBitmapImageRep, x: Int, y: Int) -> NSColor {
            // colorAt creates an NSCalibratedRGB NSColor even for an sRGB PNG,
            // so converting that result to sRGB would alter already-correct RGB.
            // Inspect the actual decoded samples in the PNG's declared space.
            var values = [UInt](repeating:0,count:bitmap.samplesPerPixel)
            bitmap.getPixel(&values,atX:x,y:y)
            return NSColor(srgbRed:CGFloat(values[0])/255,green:CGFloat(values[1])/255,
                           blue:CGFloat(values[2])/255,alpha:bitmap.hasAlpha ? CGFloat(values.last!)/255 : 1)
        }
        func countColor(_ bitmap: NSBitmapImageRep, _ color: NSColor) -> Int {
            var count = 0
            for y in 0..<bitmap.pixelsHigh { for x in 0..<bitmap.pixelsWide {
                // Plot marks intentionally retain their on-screen opacity. Compare
                // unassociated RGB, while requiring a visible interior pixel.
                let sample = pixel(bitmap,x:x,y:y)
                if sample.alphaComponent > 0.5,
                   close(sample,color.withAlphaComponent(sample.alphaComponent)) { count += 1 }
            } }
            return count
        }
        // Count disconnected visible islands. With decorations disabled each point must
        // be one island; this also catches accidental rendering of other plots' cells.
        func islands(_ bitmap: NSBitmapImageRep) -> Int {
            let width = bitmap.pixelsWide, height = bitmap.pixelsHigh
            var ink = [Bool](repeating:false,count:width*height)
            for y in 0..<height { for x in 0..<width {
                ink[y*width+x] = (bitmap.colorAt(x:x,y:y)?.alphaComponent ?? 0) > 0.2
            } }
            var count = 0
            for start in ink.indices where ink[start] {
                count += 1
                ink[start] = false
                var pending = [start]
                while let current = pending.popLast() {
                    let x = current % width, y = current / width
                    for yy in max(0,y-1)...min(height-1,y+1) {
                        for xx in max(0,x-1)...min(width-1,x+1) {
                            let next = yy*width+xx
                            if ink[next] { ink[next] = false; pending.append(next) }
                        }
                    }
                }
            }
            return count
        }
        func rejects(_ label: String, _ operation: () throws -> NSImage) {
            do { _ = try operation(); fatalError(label) } catch { }
        }

        let cells = [item("a",x:-2,y:-1,group:0,value:0.2),
                     item("b",x:0,y:2,group:1,value:0.5),
                     item("c",x:3,y:0,group:2,value:0.9)]
        let encoder = JSONEncoder()
        encoder.outputFormatting = .sortedKeys
        let original = try encoder.encode(cells)
        var options = PlotRenderOptions()
        options.title = "A plot title"
        options.xLabel = "First component"
        options.yLabel = "Second component"
        options.showAxes = false
        options.showGrid = false
        options.showTitle = false
        options.showLegend = false
        options.highlightSelection = false
        options.background = .transparent
        let size = CGSize(width:640,height:480)

        let transparent = try PlotRender2D.image(items:cells,methodID:"shown",options:options,size:size)
        let transparentBitmap = try bitmap(transparent)
        require(transparentBitmap.pixelsWide == 640 && transparentBitmap.pixelsHigh == 480,
                "PNG dimensions must equal requested pixels regardless of Retina display scaling")
        for x in [0,639] { for y in [0,479] {
            require((transparentBitmap.colorAt(x:x,y:y)?.alphaComponent ?? 1) == 0,
                    "Transparent export retains clear corner pixels")
        } }
        require(islands(transparentBitmap) == 3,
                "Hiding axes, grid, title and legend leaves exactly the three visible points")
        let subset = try bitmap(PlotRender2D.image(items:Array(cells.prefix(2)),methodID:"shown",options:options,size:size))
        require(islands(subset) == 2, "Export renders only the cohort supplied for this plot")
        let singleton = try bitmap(PlotRender2D.image(items:[cells[0]],methodID:"shown",options:options,size:size))
        require(islands(singleton) == 1, "A single object has a defined center even when both axis ranges are zero")
        for group in 0..<3 {
            require(countColor(transparentBitmap,PlotPalette.atlas.groupColor(group)) > 3,
                    "Default group colors are preserved in rendered point interiors")
        }

        var featureOptions = options
        featureOptions.showLegend = false
        let feature = PlotColorSettings(feature:"sphericity",palette:.viridis)
        let featureBitmap = try bitmap(PlotRender2D.image(items:cells,methodID:"shown",coloring:feature,options:featureOptions,size:size))
        require(countColor(featureBitmap,PlotPalette.viridis.color(0)) > 3 &&
                countColor(featureBitmap,PlotPalette.viridis.color(1)) > 3,
                "Feature coloring preserves the selected palette and visible value range")
        let subsetDomain = try bitmap(PlotRender2D.image(items:Array(cells.prefix(2)), methodID:"shown", coloring:feature, options:options, size:size, colorDomain:cells))
        require(countColor(subsetDomain, PlotPalette.viridis.color((0.5-0.2)/(0.9-0.2))) > 3,
                "A shared color domain keeps the same feature color when a plot contains only a subset of cells")
        let groupBitmap = try bitmap(PlotRender2D.image(items:cells,methodID:"shown",groups:["a":5,"b":5,"c":5],options:options,size:size))
        require(countColor(groupBitmap,PlotPalette.atlas.groupColor(5)) > 12,
                "Explicit comparison groups override a representation's original clusters")
        let otherMethodBitmap = try bitmap(PlotRender2D.image(items:cells,methodID:"shown",colorMethodID:"color",options:options,size:size))
        require(countColor(otherMethodBitmap,PlotPalette.atlas.groupColor(5)) > 3,
                "Coloring by another method retains that method's group assignment")

        for background in [PlotRenderBackground.dark,.light,.custom] {
            var configured = options
            configured.background = background
            configured.customBackground = Color(red:0.2,green:0.4,blue:0.7)
            let rendered = try bitmap(PlotRender2D.image(items:cells,methodID:"shown",options:configured,size:size))
                require(close(pixel(rendered,x:0,y:0),configured.backgroundColor),
                    "Opaque export uses the selected \(background.rawValue.lowercased()) background")
        }

        var decorated = options
        decorated.showAxes = true
        decorated.showGrid = true
        decorated.showTitle = true
        decorated.showLegend = true
        decorated.axisColor = .red
        let decoratedData = try PlotImageExport.pngData(PlotRender2D.image(items:cells,methodID:"shown",options:decorated,size:size))
        let plainData = try PlotImageExport.pngData(transparent)
        require(decoratedData != plainData, "Axis, grid, title and legend settings affect the rendered output")
        for (name,key) in [("axes",\PlotRenderOptions.showAxes),("grid",\PlotRenderOptions.showGrid),
                           ("title",\PlotRenderOptions.showTitle),("legend",\PlotRenderOptions.showLegend)] {
            var oneDecoration = options
            oneDecoration[keyPath:key] = true
            let data = try PlotImageExport.pngData(PlotRender2D.image(items:cells,methodID:"shown",options:oneDecoration,size:size))
            require(data != plainData, "The \(name) visibility control independently changes the exported image")
        }
        var originalLabels = options
        originalLabels.showAxes = true
        var changedLabels = originalLabels
        changedLabels.xLabel = "Custom horizontal label"
        changedLabels.yLabel = "Custom vertical label"
        changedLabels.axisColor = .red
        let originalLabelsData = try PlotImageExport.pngData(PlotRender2D.image(items:cells,methodID:"shown",options:originalLabels,size:size))
        let changedLabelsData = try PlotImageExport.pngData(PlotRender2D.image(items:cells,methodID:"shown",options:changedLabels,size:size))
        require(originalLabelsData != changedLabelsData, "Custom axis labels and color are included in the rendered image")
        let unavailable = Specimen(id:"not-in-this-plot",sample:"sample",split:"test",flags:"",label:20,qc:true,
                                   metrics:[:],representations:[:],cellCount:nil)
        let invalid = Specimen(id:"nonfinite",sample:"sample",split:"test",flags:"",label:21,qc:true,
                               metrics:[:],representations:["shown":Representation(vector:[0,0],point:[.nan,1],cluster:0)],cellCount:nil)
        let filteredData = try PlotImageExport.pngData(PlotRender2D.image(items:cells+[unavailable,invalid],methodID:"shown",options:options,size:size))
        require(filteredData == plainData, "Unavailable and nonfinite projections are skipped safely")

        let noShapesData = try PlotImageExport.pngData(PlotRender2D.image(items:cells,methodID:"shown",cellShapes:true,options:options,size:size))
        require(noShapesData == plainData, "Missing reconstruction thumbnails fall back to regular plot marks")
        let thumbnail = NSImage(size:CGSize(width:80,height:80),flipped:false) { _ in
            NSColor.white.setFill()
            NSBezierPath(roundedRect:CGRect(x:20,y:8,width:40,height:64),xRadius:12,yRadius:12).fill()
            return true
        }
        let shapes = try PlotRender2D.image(items:cells,methodID:"shown",thumbnails:["a":thumbnail],cellShapes:true,options:options,size:size)
        let shapesData = try PlotImageExport.pngData(shapes)
        let shapesBitmap = try bitmap(shapes)
        require(shapesData != plainData && islands(shapesBitmap) == 3,
                "Available cell reconstructions replace only their own marks and retain transparent surroundings")
        var highlighted = options
        highlighted.highlightSelection = true
        let selectedData = try PlotImageExport.pngData(PlotRender2D.image(items:cells,methodID:"shown",selected:"b",marked:["a"],options:highlighted,size:size))
        let suppressedData = try PlotImageExport.pngData(PlotRender2D.image(items:cells,methodID:"shown",selected:"b",marked:["a"],options:options,size:size))
        require(selectedData != suppressedData && suppressedData == plainData,
                "Selection highlighting can be removed without changing the underlying plot")
        let afterRendering = try encoder.encode(cells)
        require(afterRendering == original, "Export customization leaves input metrics and representations unchanged")

        rejects("An empty cohort must produce a recoverable rendering error") {
            try PlotRender2D.image(items:[],methodID:"shown",options:options,size:size)
        }
        rejects("Zero-sized output must produce a recoverable rendering error") {
            try PlotRender2D.image(items:cells,methodID:"shown",options:options,size:CGSize(width:0,height:480))
        }
        rejects("Nonfinite output size must produce a recoverable rendering error") {
            try PlotRender2D.image(items:cells,methodID:"shown",options:options,size:CGSize(width:CGFloat.nan,height:480))
        }
        let destination = FileManager.default.temporaryDirectory.appendingPathComponent("atlas-plot-render-" + UUID().uuidString + ".png")
        defer { try? FileManager.default.removeItem(at:destination) }
        try PlotImageExport.writePNG(transparent,to:destination)
        let written = try Data(contentsOf:destination)
        require(written == plainData,
                "Saving a PNG retains the renderer's original pixels and transparency")
        print("PASS: exact PNG resolution; transparent/custom backgrounds; decoration controls; plot-local cohorts; feature and group palettes; selection toggle; input immutability; invalid rendering inputs; atomic PNG roundtrip.")
    }
}
