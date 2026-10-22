import SwiftUI
import AppKit

/// Draws only the requested plot. Existing cell thumbnails are reused without mesh I/O.
enum PlotRender2D {
    @MainActor static func image(items: [Specimen], methodID: String, colorMethodID: String? = nil,
                                groups: [String: Int]? = nil, coloring: PlotColorSettings = PlotColorSettings(),
                                selected: String? = nil, marked: Set<String> = [], thumbnails: [String: NSImage] = [:],
                                cellShapes: Bool = false, shapeSize: Double = 28, referenceWidth: CGFloat = 400,
                                options: PlotRenderOptions, size: CGSize, colorDomain: [Specimen]? = nil) throws -> NSImage {
        guard size.width.isFinite, size.height.isFinite, size.width >= 64, size.height >= 64,
              size.width <= 8192, size.height <= 8192 else { throw DataError.invalid("Choose an image size between 64 and 8192 pixels per side.") }
        let plotted = items.filter { item in
            guard let point = item.representations[methodID]?.point else { return false }
            return point.count >= 2 && point[0].isFinite && point[1].isFinite
        }
        guard !plotted.isEmpty else { throw DataError.invalid("This plot has no visible objects to render.") }
        let width = Int(size.width.rounded()), height = Int(size.height.rounded())
        guard let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height,
                                             bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                                             colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0),
              let context = NSGraphicsContext(bitmapImageRep: bitmap) else { throw DataError.invalid("Could not create the plot image.") }
        NSGraphicsContext.saveGraphicsState(); NSGraphicsContext.current = context
        defer { NSGraphicsContext.restoreGraphicsState() }
        let size = CGSize(width: width, height: height), cg = context.cgContext
        cg.clear(CGRect(origin: .zero, size: size)); options.backgroundColor.setFill(); CGRect(origin: .zero, size: size).fill()
        let unit = max(0.25, size.width / 800), glyphScale = size.width / max(100, referenceWidth)
        let ink = options.foregroundColor, axisInk = NSColor(options.axisColor)
        func text(_ value: String, _ rect: CGRect, size: CGFloat = 13, color: NSColor? = nil, alignment: NSTextAlignment = .left, bold: Bool = false) {
            let paragraph = NSMutableParagraphStyle(); paragraph.alignment = alignment
            (value as NSString).draw(in: rect, withAttributes: [.font: bold ? NSFont.boldSystemFont(ofSize: size*unit) : NSFont.systemFont(ofSize: size*unit),
                .foregroundColor: color ?? ink, .paragraphStyle: paragraph])
        }
        let margin: CGFloat = 22*unit
        let titleHeight: CGFloat = options.showTitle ? 44*unit : 0
        let legendHeight: CGFloat = options.showLegend ? 42*unit : 0
        let left: CGFloat = options.showAxes ? 72*unit : margin
        let bottom: CGFloat = margin + legendHeight + (options.showAxes ? 48*unit : 0)
        let frame = CGRect(x: left, y: bottom, width: max(1, size.width-left-margin), height: max(1, size.height-bottom-margin-titleHeight))
        // A little breathing room keeps endpoint glyphs clear of tick labels.
        let plot = frame.insetBy(dx: 15*unit, dy: 15*unit)
        let points = plotted.map { $0.representations[methodID]!.point }
        let xmin = points.map { $0[0] }.min()!, xmax = points.map { $0[0] }.max()!
        let ymin = points.map { $0[1] }.min()!, ymax = points.map { $0[1] }.max()!
        let map = PlotColorMap(settings: coloring, items: colorDomain ?? items)
        if options.showTitle { text(options.title, CGRect(x: margin, y: size.height-margin-28*unit, width: size.width-2*margin, height: 30*unit), size: 19, bold: true) }
        for tick in 0...4 {
            let fraction = CGFloat(tick)/4, x = plot.minX + fraction*plot.width, y = plot.minY + fraction*plot.height
            if options.showGrid {
                cg.setStrokeColor(axisInk.withAlphaComponent(0.2).cgColor); cg.setLineWidth(unit)
                cg.move(to: CGPoint(x: x, y: frame.minY)); cg.addLine(to: CGPoint(x: x, y: frame.maxY)); cg.strokePath()
                cg.move(to: CGPoint(x: frame.minX, y: y)); cg.addLine(to: CGPoint(x: frame.maxX, y: y)); cg.strokePath()
            }
            if options.showAxes {
                text(String(format: "%.3g", xmin+Double(fraction)*(xmax-xmin)), CGRect(x: x-34*unit, y: frame.minY-21*unit, width: 68*unit, height: 17*unit), color: axisInk, alignment: .center)
                text(String(format: "%.3g", ymin+Double(fraction)*(ymax-ymin)), CGRect(x: 3*unit, y: y-8*unit, width: left-12*unit, height: 18*unit), color: axisInk, alignment: .right)
            }
        }
        if options.showAxes {
            cg.setStrokeColor(axisInk.cgColor); cg.setLineWidth(1.2*unit)
            cg.move(to: CGPoint(x: frame.minX, y: frame.maxY)); cg.addLine(to: CGPoint(x: frame.minX, y: frame.minY)); cg.addLine(to: CGPoint(x: frame.maxX, y: frame.minY)); cg.strokePath()
            text(options.xLabel, CGRect(x: frame.minX, y: frame.minY-44*unit, width: frame.width, height: 22*unit), size: 14, color: axisInk, alignment: .center)
            cg.saveGState(); cg.translateBy(x: 18*unit, y: frame.midY); cg.rotate(by: .pi/2)
            text(options.yLabel, CGRect(x: -frame.height/2, y: -10*unit, width: frame.height, height: 22*unit), size: 14, color: axisInk, alignment: .center)
            cg.restoreGState()
        }
        for (item, point) in zip(plotted, points) {
            let isSelected = options.highlightSelection && item.id == selected
            let isMarked = options.highlightSelection && marked.contains(item.id)
            let opacity: CGFloat = !options.highlightSelection || marked.isEmpty || isMarked ? 0.85 : 0.18
            let group = groups.map { $0[item.id] ?? -1 } ?? item.representations[colorMethodID ?? methodID]?.cluster ?? -1
            let baseColor = map.color(item: item, group: group)
            let color = isSelected && coloring.feature.isEmpty ? NSColor.systemOrange : baseColor
            let x = plot.minX + (xmax > xmin ? CGFloat((point[0]-xmin)/(xmax-xmin)) : 0.5)*plot.width
            let y = plot.minY + (ymax > ymin ? CGFloat((point[1]-ymin)/(ymax-ymin)) : 0.5)*plot.height
            cg.saveGState()
            if cellShapes, let image = thumbnails[item.id] {
                let edge = shapeSize*glyphScale*(isSelected ? 1.25 : isMarked ? 1.1 : 1)
                let rect = CGRect(x: x-edge/2, y: y-edge/2, width: edge, height: edge)
                let tinted = tintedThumbnail(image, color: color)
                tinted.draw(in: rect, from: .zero, operation: .sourceOver, fraction: isSelected ? 1 : opacity)
                if isSelected || isMarked {
                    cg.setStrokeColor((isSelected ? NSColor.systemOrange : .systemTeal).cgColor); cg.setLineWidth(1.5*glyphScale); cg.stroke(rect)
                }
            } else {
                let radius = (isSelected ? 6.5 : isMarked ? 4.5 : 3.2)*glyphScale
                let rect = CGRect(x: x-radius, y: y-radius, width: 2*radius, height: 2*radius)
                cg.setFillColor(color.withAlphaComponent(isSelected ? 1 : opacity).cgColor); cg.fillEllipse(in: rect)
                if isSelected || isMarked {
                    cg.setStrokeColor((isSelected ? ink : .systemTeal).cgColor); cg.setLineWidth((isSelected ? 2 : 1)*glyphScale); cg.strokeEllipse(in: rect.insetBy(dx: -2*glyphScale, dy: -2*glyphScale))
                }
            }
            cg.restoreGState()
        }
        if options.showLegend {
            let area = CGRect(x: left, y: margin, width: frame.width, height: 32*unit)
            if coloring.feature.isEmpty {
                let visibleGroups = Set(plotted.map { item in groups.map { $0[item.id] ?? -1 } ?? item.representations[colorMethodID ?? methodID]?.cluster ?? -1 }).sorted()
                let groupsWidth = min(area.width, CGFloat(visibleGroups.count)*90*unit)
                let step = groupsWidth/CGFloat(max(1, visibleGroups.count))
                for (index, group) in visibleGroups.enumerated() {
                    let x = area.minX+CGFloat(index)*step
                    cg.setFillColor(map.color(value: nil, group: group).cgColor); cg.fillEllipse(in: CGRect(x: x, y: area.minY+10*unit, width: 9*unit, height: 9*unit))
                    text(group < 0 ? "Unassigned" : "Group \(group+1)", CGRect(x: x+14*unit, y: area.minY+4*unit, width: step-14*unit, height: 23*unit), size: 12)
                }
            } else {
                let labelWidth = min(area.width*0.45, 230*unit), bar = CGRect(x: area.minX+labelWidth, y: area.minY+14*unit, width: max(1, area.width-labelWidth), height: 10*unit)
                text(plotFeatureName(coloring.feature), CGRect(x: area.minX, y: area.minY+5*unit, width: labelWidth-8*unit, height: 28*unit), size: 12)
                for i in 0..<128 {
                    let color = map.minimum == nil ? NSColor.gray : coloring.palette.color(map.minimum == map.maximum ? 0.5 : Double(i)/127)
                    cg.setFillColor(color.cgColor)
                    cg.fill(CGRect(x: bar.minX+CGFloat(i)/128*bar.width, y: bar.minY, width: bar.width/128+0.5, height: bar.height))
                }
                text(map.minimum.map { String(format: "%.3g", $0) } ?? "No values", CGRect(x: bar.minX, y: area.minY-4*unit, width: bar.width/2, height: 18*unit), size: 11)
                text(map.maximum.map { String(format: "%.3g", $0) } ?? "", CGRect(x: bar.midX, y: area.minY-4*unit, width: bar.width/2, height: 18*unit), size: 11, alignment: .right)
            }
        }
        context.flushGraphics()
        let result = NSImage(size: size); result.addRepresentation(bitmap); return result
    }

    private static func tintedThumbnail(_ image: NSImage, color: NSColor) -> NSImage {
        let size = CGSize(width: 80, height: 80)
        guard let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 80, pixelsHigh: 80, bitsPerSample: 8, samplesPerPixel: 4,
                                             hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0),
              let context = NSGraphicsContext(bitmapImageRep: bitmap) else { return image }
        NSGraphicsContext.saveGraphicsState(); NSGraphicsContext.current = context
        context.cgContext.clear(CGRect(origin: .zero, size: size))
        image.draw(in: CGRect(origin: .zero, size: size))
        color.setFill(); CGRect(origin: .zero, size: size).fill(using: .multiply)
        image.draw(in: CGRect(origin: .zero, size: size), from: .zero, operation: .destinationIn, fraction: 1)
        context.flushGraphics(); NSGraphicsContext.restoreGraphicsState()
        let tinted = NSImage(size: size); tinted.addRepresentation(bitmap); return tinted
    }
}
