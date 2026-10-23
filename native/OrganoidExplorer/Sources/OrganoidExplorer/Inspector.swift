import SwiftUI

struct Inspector: View {
    @EnvironmentObject var store: ExplorerStore
    var body: some View { InspectorContent(navigation: store.sliceNavigation) }
}

// Only this inspector subscribes to rapid slice changes; analysis views stay untouched.
private struct InspectorContent: View {
    @ObservedObject var navigation: SliceNavigation
    @EnvironmentObject var store: ExplorerStore
    @State private var meshError: String?
    @State private var showMeasurements = false
    @State private var showLearnedShape = false
    var body: some View {
        ScrollView {
        VStack(alignment: .leading, spacing: 10) {
            HStack { Text("Reconstruction").font(.headline); Spacer(); Text(store.sample?.id ?? "—").monospaced().foregroundStyle(.secondary) }
            if let folder = store.folder, let sample = store.sample {
                ZStack(alignment: .bottomLeading) {
                    ReconstructionView(folder: folder, sample: sample, cells: store.dataset!.cells.filter { $0.sample == sample.id }, selectedLabel: store.selected?.label ?? 0,
                                       isolate: store.isolate, hideExcluded: store.hideExcluded, clipping: store.clip, axis: store.axis, slice: store.slice, reset: store.cameraReset,
                                       pick: { label in if store.level == "Cells" { store.select("\(sample.id):L\(label)") } },
                                       onError: { meshError = $0 })
                    VStack(alignment: .leading, spacing: 4) {
                        Text(meshError ?? (store.selected?.label ?? 0 > 0 ? (store.hideExcluded ? "Orange · selected cell · excluded cells hidden" : "Orange · selected cell · red = excluded") : (store.hideExcluded ? "Excluded cells hidden" : "Red · excluded from analyses")))
                        Text("Drag to orbit · scroll to zoom · click to select")
                    }.font(.caption2).foregroundStyle(.white.opacity(0.8)).padding(12).allowsHitTesting(false)
                }.frame(height: 250).clipShape(RoundedRectangle(cornerRadius: 10))
                .onChange(of: sample.id) { meshError = nil }
                HStack {
                    Toggle("Isolate cell", isOn: $store.isolate).disabled(store.level != "Cells")
                    Toggle("Slice 3D", isOn: $store.clip)
                    Spacer()
                    Button { store.cameraReset += 1 } label: { Image(systemName: "arrow.counterclockwise") }.help("Reset camera")
                }.toggleStyle(.checkbox).font(.caption)
                Toggle("Hide excluded cells", isOn: $store.hideExcluded)
                    .toggleStyle(.checkbox).font(.caption)
                    .help("Hide excluded 3D surfaces and their label overlays. Raw microscopy stays visible.")
                if store.hideExcluded && store.selected?.includedInAnalysis == false {
                    Text("This cell is excluded and hidden. Turn off Hide excluded cells to review its reconstruction.")
                        .font(.caption2).foregroundStyle(.secondary)
                }
                Divider()
                HStack { Text(store.dataset?.syntheticDemo == true ? "Synthetic image + labels" : "Raw microscopy + labels").font(.headline); Spacer(); Toggle("Overlay", isOn: $store.overlay).toggleStyle(.checkbox).font(.caption) }
                if let volume = store.volume {
                    RawSliceView(volume: volume, navigation: navigation)
                    HStack {
                        Picker("Plane", selection: $store.axis) { Text("XY").tag(0); Text("XZ").tag(1); Text("YZ").tag(2) }.pickerStyle(.segmented).labelsHidden().frame(width: 125)
                        Slider(value: $navigation.fraction, in: 0...1)
                        Text("\(Int(store.slice * Double(volume.dimensions(axis: store.axis).2-1))+1)/\(volume.dimensions(axis: store.axis).2)").font(.caption.monospacedDigit()).frame(width: 55)
                    }
                    HStack {
                        Text("Brightness").font(.caption).foregroundStyle(.secondary)
                        Slider(value: $store.contrast, in: 0.3...3)
                        Text(String(format: "%.1f µm", sample.extent[store.axis]*store.slice)).font(.caption.monospacedDigit())
                    }
                    if !sample.rawAvailable { Text("Raw source unavailable · showing segmentation only").font(.caption).foregroundStyle(.orange) }
                } else if let error = store.volumeError { Text(error).font(.caption).foregroundStyle(.orange).frame(height: 180) }
                else { ProgressView("Loading raw stack…").frame(maxWidth: .infinity).frame(height: 180) }
                if let selected = store.selected {
                    HStack(spacing: 18) {
                        metric("VOLUME", String(format: "%.0f µm³", selected.volume))
                        metric("SPHERICITY", String(format: "%.3f", selected.sphericity))
                        metric("ELONGATION", String(format: "%.2f", selected.elongation))
                    }.padding(.top, 5)
                    Text(store.level == "Cells" ? "\(selected.qc ? "QC pass" : selected.flags) · \(selected.split) split" : "Volume: main-organoid cells · shape metrics: included QC-passing cell mean").font(.caption2).foregroundStyle(.secondary)
                }
                if let selected = store.selected, store.level == "Cells" {
                    Label(selected.includedInAnalysis ? "Included in analyses" : "Outside main organoid · excluded from analyses",
                          systemImage: selected.includedInAnalysis ? "checkmark.circle.fill" : "minus.circle.fill")
                        .font(.caption).foregroundStyle(selected.includedInAnalysis ? Color.teal : Color.red)
                    HStack {
                        Button(selected.includedInAnalysis ? "Exclude cell" : "Include cell") {
                            store.updateMembership(cellID: selected.id, included: !selected.includedInAnalysis)
                        }
                        Button("Use automatic decision") { store.updateMembership(cellID: selected.id) }
                    }.font(.caption).disabled(store.membershipBusy)
                }
                if store.level == "Cells", store.selected?.pointReconstructions?["pointnet_shape128"] != nil {
                    Button("Compare PointNet++ reconstruction") { showLearnedShape = true }.font(.caption)
                }
                Button("All measurements & source files") { showMeasurements = true }
                    .font(.caption)
                    .popover(isPresented: $showMeasurements) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(store.selected?.id ?? "Measurements").font(.headline)
                            ForEach((store.selected?.metrics.keys.sorted() ?? []), id: \.self) { key in
                                HStack {
                                    Text(key.replacingOccurrences(of: "_", with: " "))
                                    Spacer()
                                    Text(store.selected?.metrics[key] ?? 0, format: .number.precision(.fractionLength(3))).monospacedDigit()
                                }
                            }
                            Divider()
                            Text("Source microscopy").font(.caption.bold())
                            Text(sample.sourceImage).font(.caption2).textSelection(.enabled)
                            Text("Predicted labels").font(.caption.bold())
                            Text(sample.sourceLabels).font(.caption2).textSelection(.enabled)
                        }.padding(20).frame(width: 420)
                    }
                Text(store.dataset?.syntheticDemo == true ? "Generated shapes · nominal µm units\n3D slicing clips surfaces; cut faces are not capped." : "Original surfaces · inferred µm calibration\n3D slicing clips surfaces; cut faces are not capped.").font(.caption2).foregroundStyle(.secondary)
            } else { ContentUnavailableView("Select an object", systemImage: "cursorarrow.click", description: Text("Choose a row or plot point to open its parent reconstruction.")) }
        }
    }
    .sheet(isPresented:$showLearnedShape) {
        if let selected = store.selected { LearnedShapeComparison(item:selected).environmentObject(store) }
    }
    }
    func metric(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 3) { Text(title).font(.system(size: 9, weight: .semibold)).foregroundStyle(.secondary); Text(value).font(.callout.monospacedDigit().weight(.medium)) }
    }
}
struct RawSliceView: View {
    @EnvironmentObject var store: ExplorerStore
    let volume: VolumeData
    @ObservedObject var navigation: SliceNavigation
    var style: VolumeSliceStyle {
        VolumeSliceStyle(axis: store.axis, selected: store.selected?.label ?? 0, overlay: store.overlay,
                         contrast: store.contrast,
                         excluded: Set(store.excludedCells.filter { $0.sample == volume.sample.id }.map(\.label)),
                         hideExcluded: store.hideExcluded)
    }
    var body: some View {
        GeometryReader { geo in
            let extent = volume.sample.extent
            let width = store.axis == 2 ? extent[1] : extent[2]
            let height = store.axis == 0 ? extent[1] : extent[0]
            let scale = min(geo.size.width / width, geo.size.height / height)
            let size = CGSize(width: width*scale, height: height*scale)
            ZStack {
                Color.black
                if let image = volume.image(style: style, fraction: navigation.fraction) {
                    Image(decorative: image, scale: 1).resizable().interpolation(.none).frame(width: size.width, height: size.height)
                        .contentShape(Rectangle())
                        .gesture(SpatialTapGesture().onEnded { value in
                            let (w,h,d) = volume.dimensions(axis: store.axis)
                            let x = min(w-1,max(0,Int(value.location.x/size.width*Double(w))))
                            let y = min(h-1,max(0,Int(value.location.y/size.height*Double(h))))
                            let z = min(d-1,max(0,Int(store.slice*Double(d-1))))
                            let label = volume.labels[volume.index(x:x,y:y,slice:z,axis:store.axis)]
                            if label > 0 && store.level == "Cells" && !(store.hideExcluded && store.excludedCells.contains { $0.sample == volume.sample.id && $0.label == Int(label) }) { store.select("\(volume.sample.id):L\(label)") }
                        })
                }
            }.frame(maxWidth: .infinity, maxHeight: .infinity)
        }.frame(height: 180).clipShape(RoundedRectangle(cornerRadius: 8))
        .task(id: SlicePreparation(volumeID: volume.cacheID, style: style)) {
            let snapshot = volume, settings = style, fraction = navigation.fraction
            let worker = Task.detached(priority: .utility) {
                snapshot.prepareSlices(style: settings, near: fraction)
            }
            await withTaskCancellationHandler(operation: { await worker.value }, onCancel: { worker.cancel() })
        }
    }
}
struct ComparisonSheet: View {
    @EnvironmentObject var store: ExplorerStore
    @Environment(\.dismiss) private var dismiss
    @StateObject private var link = CameraLink()
    @State private var isolated = true
    @State private var reset = 0
    @State private var error: String?
    var comparisonMethod: String { store.shapeComparisonMethodID ?? store.method }
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Compare shapes").font(.title2.bold())
                    Text("Original geometry at a shared camera scale · context available for each cell").foregroundStyle(.secondary)
                }
                Spacer(); Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            }
            HStack {
                Toggle("Isolate cells", isOn: $isolated).disabled(store.level != "Cells")
                Toggle("Hide excluded cells", isOn: $store.hideExcluded)
                Toggle("Link rotation & zoom", isOn: $store.syncCameras).onChange(of: store.syncCameras) { link.enabled = store.syncCameras }
                Button("Reset views") { reset += 1 }
                Spacer()
                Menu("Choose comparison") {
                    Section("Traditional neighbors") { ForEach(store.neighbors(method: "traditional")) { n in Button(n.id) { store.compared = n.item } } }
                    Section("Selected method neighbors") { ForEach(store.neighbors(method: comparisonMethod)) { n in Button(n.id) { store.compared = n.item } } }
                }
            }.toggleStyle(.checkbox)
            HStack(spacing: 16) {
                if let query = store.selected { comparison(query, title: "QUERY") }
                if let candidate = store.compared { comparison(candidate, title: "COMPARISON") }
            }
            if let error { Text(error).foregroundStyle(.orange) }
            HStack {
                ForEach(Array(Set(["traditional", comparisonMethod])).sorted(), id: \.self) { key in
                    if let a = store.selected?.representations[key], let b = store.compared?.representations[key], let method = store.dataset?.methods.first(where: { $0.id == key }) {
                        Text("\(method.name): \(representationDistance(a.vector,b.vector,metric:method.metric), specifier: "%.5g") \(method.metric)").font(.caption).padding(10).background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
                    }
                }
            }
        }.padding(24).frame(width: 1100, height: 690).onAppear { link.enabled = store.syncCameras }
    }
    var comparisonSpan: Double {
        if isolated && store.level == "Cells" {
            return max(store.selected?.metrics["major_axis_um"] ?? 20, store.compared?.metrics["major_axis_um"] ?? 20) * 1.3
        }
        let ids = [store.selected?.sample, store.compared?.sample].compactMap { $0 }
        return store.dataset?.samples.filter { ids.contains($0.id) }.flatMap(\.extent).max() ?? 80
    }
    func comparison(_ item: Specimen, title: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.caption2).foregroundStyle(.secondary)
            HStack { Text(item.id).font(.title3.monospaced().bold()); Spacer(); Text("\(item.volume, specifier: "%.0f") µm³").monospacedDigit() }
            if let folder = store.folder, let sample = store.dataset?.samples.first(where: { $0.id == item.sample }) {
                // Center an isolated cell at the origin; retain a common framing extent across panels.
                let cells = store.dataset!.cells.filter { $0.sample == item.sample }
                ReconstructionView(folder: folder, sample: sample, cells: cells, selectedLabel: item.label,
                                   isolate: isolated && store.level == "Cells", hideExcluded: store.hideExcluded, reset: reset, centered: true, cameraSpan: comparisonSpan, link: link, onError: { error = $0 })
                    .clipShape(RoundedRectangle(cornerRadius: 10))
            }
            HStack { Text("Sphericity \(item.sphericity, specifier: "%.3f")"); Spacer(); Text("Elongation \(item.elongation, specifier: "%.2f")") }.font(.caption)
        }.frame(maxWidth: .infinity)
    }
}
