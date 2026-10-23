import SwiftUI
import AppKit

/// Display settings; never modifies representations or analysis groups.
struct PlotColorSettings: Hashable {
    var feature = ""
    var palette: PlotPalette = .atlas
}

enum PlotPalette: String, CaseIterable, Identifiable {
    case atlas = "Atlas groups", viridis = "Viridis", magma = "Magma", inferno = "Inferno"
    case cividis = "Cividis", turbo = "Turbo", blueRed = "Blue–white–red", grayscale = "Grayscale"
    var id: String { rawValue }
    var stops: [NSColor] {
        if self == .atlas { return clusterColors.map { NSColor($0) } }
        let hex: [UInt32]
        switch self {
        case .viridis: hex = [0x440154,0x3b528b,0x21918c,0x5ec962,0xfde725]
        case .magma: hex = [0x000004,0x51127c,0xb73779,0xfc8961,0xfcfdbf]
        case .inferno: hex = [0x000004,0x57106e,0xbc3754,0xf98e09,0xfcffa4]
        case .cividis: hex = [0x00224e,0x434e6c,0x7d7c78,0xbcae6c,0xfee838]
        case .turbo: hex = [0x30123b,0x466be3,0x28bceb,0x32f298,0xa4fc3c,0xf9ba38,0xe34a0c,0x7a0403]
        case .blueRed: hex = [0x2166ac,0x67a9cf,0xf7f7f7,0xef8a62,0xb2182b]
        case .grayscale: hex = [0x333333,0xffffff]
        case .atlas: hex = []
        }
        return hex.map { NSColor(srgbRed: Double(($0 >> 16) & 255)/255, green: Double(($0 >> 8) & 255)/255, blue: Double($0 & 255)/255, alpha: 1) }
    }
    func color(_ fraction: Double) -> NSColor {
        let colors = stops
        let position = min(1, max(0, fraction.isFinite ? fraction : 0.5)) * Double(colors.count - 1)
        let index = min(colors.count - 2, Int(position))
        let a = colors[index].usingColorSpace(.sRGB)!, b = colors[index+1].usingColorSpace(.sRGB)!
        let t = position - Double(index)
        return NSColor(srgbRed: a.redComponent + (b.redComponent-a.redComponent)*t,
                       green: a.greenComponent + (b.greenComponent-a.greenComponent)*t,
                       blue: a.blueComponent + (b.blueComponent-a.blueComponent)*t, alpha: 1)
    }
    func groupColor(_ group: Int) -> NSColor {
        guard group >= 0 else { return .gray }
        if self == .atlas { let colors = stops; return colors[group % colors.count] }
        return color(Double(group % 6)/5)
    }
}

func plotFeatureValue(_ item: Specimen, key: String) -> Double? {
    let value = key == "@cellCount" ? item.cellCount.map(Double.init) : item.metrics[key]
    return value.flatMap { $0.isFinite ? $0 : nil }
}
func plotFeatureKeys(_ items: [Specimen]) -> [String] {
    var keys = Set(items.flatMap { $0.metrics.keys })
    if items.contains(where: { $0.cellCount != nil }) { keys.insert("@cellCount") }
    return keys.sorted { plotFeatureName($0).localizedStandardCompare(plotFeatureName($1)) == .orderedAscending }
}
func plotFeatureName(_ key: String) -> String {
    let labels = ["volume_um3":"Volume (µm³)", "surface_area_um2":"Surface area (µm²)", "sphericity":"Sphericity",
                  "elongation":"Elongation", "flatness":"Flatness", "mesh_solidity":"Mesh solidity",
                  "major_axis_um":"Major axis (µm)", "middle_axis_um":"Middle axis (µm)", "minor_axis_um":"Minor axis (µm)",
                  "centroid_x_um":"Centroid X (µm)", "centroid_y_um":"Centroid Y (µm)", "centroid_z_um":"Centroid Z (µm)", "@cellCount":"Cell count",
                  "demo_shape_class":"Demo shape family", "demo_variant":"Demo variation", "demo_euler_characteristic":"Euler characteristic"]
    return labels[key] ?? key.replacingOccurrences(of: "_", with: " ").capitalized
}

struct PlotColorMap {
    let settings: PlotColorSettings
    let minimum: Double?
    let maximum: Double?
    let missing: Int
    init(settings: PlotColorSettings, items: [Specimen]) {
        self.settings = settings
        let values = items.compactMap { plotFeatureValue($0, key: settings.feature) }
        minimum = values.min(); maximum = values.max(); missing = items.count - values.count
    }
    func color(value: Double?, group: Int) -> NSColor {
        guard !settings.feature.isEmpty else { return settings.palette.groupColor(group) }
        guard let value, value.isFinite, let lo = minimum, let hi = maximum else { return .gray }
        if settings.feature == "demo_shape_class" { return settings.palette.groupColor(Int(value)) }
        return settings.palette.color(hi > lo ? (value-lo)/(hi-lo) : 0.5)
    }
    func color(item: Specimen, group: Int) -> NSColor { color(value: plotFeatureValue(item, key: settings.feature), group: group) }
}

extension ExplorerStore {
    func plotColorKey(_ method: String) -> String { level + "|" + method }
    var unifiedPlotColors: Bool { unifiedPlotColorLevels.contains(level) }
    func setUnifiedPlotColors(_ enabled: Bool, seedMethod: String) {
        if enabled {
            if unifiedPlotColorSettings[level] == nil {
                unifiedPlotColorSettings[level] = plotColorSettings[plotColorKey(seedMethod)] ?? PlotColorSettings()
            }
            unifiedPlotColorLevels.insert(level)
        } else {
            unifiedPlotColorLevels.remove(level)
        }
    }
    func plotColors(_ method: String) -> PlotColorSettings {
        if unifiedPlotColors { return unifiedPlotColorSettings[level] ?? PlotColorSettings() }
        return plotColorSettings[plotColorKey(method)] ?? PlotColorSettings()
    }
    func plotColorDomain(_ items: [Specimen]) -> [Specimen] { unifiedPlotColors ? filtered : items }
    func plotColorBinding(_ method: String) -> Binding<PlotColorSettings> {
        Binding(get: { self.plotColors(method) }, set: {
            if self.unifiedPlotColors { self.unifiedPlotColorSettings[self.level] = $0 }
            else { self.plotColorSettings[self.plotColorKey(method)] = $0 }
        })
    }
    func unifiedPlotColorBinding() -> Binding<PlotColorSettings> {
        Binding(get: { self.unifiedPlotColorSettings[self.level] ?? PlotColorSettings() },
                set: { self.unifiedPlotColorSettings[self.level] = $0 })
    }
}

enum PlotColorScope {
    case thisPlot, allMethods
    var help: String {
        switch self {
        case .thisPlot: return "Only this method's plot changes. Feature ranges use the same visible cohort across plots; coloring does not change grouping or similarity."
        case .allMethods: return "Every method plot, including expanded windows and rendered images, uses this feature, palette and one range across the filtered cells or organoids. Objects without a method’s representation still contribute to that range. Coloring does not change grouping or similarity."
        }
    }
}

struct PlotColorControls: View {
    @Binding var settings: PlotColorSettings
    let features: [String]
    let map: PlotColorMap
    var scope: PlotColorScope = .thisPlot
    var emptyFeatureLabel = "Groups (comparison setting)"
    var showGroupPalette = true
    var helpOverride: String? = nil
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Picker("Color by", selection: Binding(get: { settings.feature }, set: { feature in
                settings.feature = feature
                if !feature.isEmpty && settings.palette == .atlas { settings.palette = .viridis }
            })) {
                Text(emptyFeatureLabel).tag("")
                ForEach(features, id: \.self) { Text(plotFeatureName($0)).tag($0) }
                if !settings.feature.isEmpty && !features.contains(settings.feature) { Text(plotFeatureName(settings.feature) + " · unavailable").tag(settings.feature) }
            }.accessibilityLabel(scope == .allMethods ? "Color all method plots by a feature" : "Color this plot by a feature")
            if showGroupPalette || !settings.feature.isEmpty {
            Picker("Palette", selection: $settings.palette) {
                ForEach(PlotPalette.allCases.filter { settings.feature.isEmpty || $0 != .atlas }) { Text($0.rawValue).tag($0) }
            }.accessibilityLabel(scope == .allMethods ? "Color palette for all method plots" : "Color palette for this plot")
            }
            if settings.feature.isEmpty && showGroupPalette {
                HStack(spacing: 5) {
                    ForEach(0..<6) { group in
                        Circle().fill(Color(settings.palette.groupColor(group))).frame(width: 6, height: 6)
                        Text("\(group+1)")
                    }
                }.font(.caption2).foregroundStyle(.secondary)
            } else if settings.feature.isEmpty {
                Text(emptyFeatureLabel).font(.caption2).foregroundStyle(.secondary)
            } else if let lo = map.minimum, let hi = map.maximum {
                VStack(spacing: 2) {
                    if hi == lo {
                        Capsule().fill(Color(settings.palette.color(0.5))).frame(height: 6)
                    } else {
                        LinearGradient(colors: settings.palette.stops.map { Color($0) }, startPoint: .leading, endPoint: .trailing).frame(height: 6).clipShape(Capsule())
                    }
                    HStack {
                        Text(String(format: "%.4g", lo)); Spacer()
                        if hi == lo { Text("Constant"); Spacer() }
                        Text(String(format: "%.4g", hi))
                    }.monospacedDigit()
                }.font(.caption2).foregroundStyle(.secondary)
                if map.missing > 0 { Text("Gray: missing value").font(.caption2).foregroundStyle(.secondary) }
            } else { Text("No values available · gray points").font(.caption2).foregroundStyle(.secondary) }
        }.controlSize(.small).font(.caption)
        .help(helpOverride ?? scope.help)
    }
}
