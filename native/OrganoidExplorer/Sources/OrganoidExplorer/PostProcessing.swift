import SwiftUI
import AppKit

struct PostProcessingView: View {
    @EnvironmentObject var atlas: ExplorerStore
    @EnvironmentObject var segmentation: SegmentationStore
    @ObservedObject var post: SegmentationStore
    @State private var stackID = ""
    @State private var showLog = false
    var originals: [SegResult] { post.results.filter { $0.method != "refined" } }
    var trials: [SegResult] { post.results.filter { $0.refinement?.parentID == post.resultA?.id && $0.method == "refined" } }
    var body: some View {
        VStack(alignment:.leading,spacing:12) {
            HStack {
                VStack(alignment:.leading,spacing:4) {
                    Text("Post processing").font(.system(size:26,weight:.semibold))
                    Text("Membrane-guided refinement · linked before / after inspection").foregroundStyle(.secondary)
                }
                Spacer()
                if post.busy { Button("Cancel") { post.cancel() } }
            }
            inputControls
            HStack {
                Picker("Before",selection:Binding(get:{post.resultA?.id ?? ""},set:{selectOriginal($0)})) {
                    Text("Choose original segmentation…").tag("")
                    ForEach(originals) { Text("\(post.methodName($0.method)) · \($0.title)").tag($0.id) }
                }
                Picker("After",selection:Binding(get:{post.resultB?.id ?? ""},set:{id in if let result = trials.first(where:{$0.id == id}) {post.showRefinement(result);post.compare()} else {clearAfter()}})) {
                    Text("No refined result").tag("")
                    ForEach(trials) { Text($0.title).tag($0.id) }
                }
            }.disabled(post.busy)
            HStack {
                Button("Run post processing") { post.refineDisplayed(side:1) }.buttonStyle(.borderedProminent).disabled(!post.canRefine(side:1))
                Button("Restore trial settings") { restoreSettings() }.disabled(post.busy || post.resultB == nil)
                Button("Conservative defaults") {post.refinementSettings=RefinementSettings()}.disabled(post.busy)
                Button("Import original labels…") { clearAfter();post.importLabels(method:"imported",side:1) }.disabled(!post.canRun)
                Button("Refresh history") { refreshHistory() }.disabled(post.busy || post.source == nil)
                Spacer()
                Text(post.status).font(.caption).foregroundStyle(.secondary).lineLimit(2)
            }
            if post.busy {
                HStack { Text(post.progressStage).font(.caption);ProgressView(value:post.progress).frame(maxWidth:220);Text("\(Int(post.progress*100))%").font(.caption.monospacedDigit()) }
            }
            HSplitView {
                ScrollView { RefinementControls().environmentObject(post).padding(.trailing,8) }.frame(minWidth:280,idealWidth:310,maxWidth:350)
                ScrollView {
                    VStack(alignment:.leading,spacing:14) {
                        if post.source != nil {
                            sliceControls
                            HStack(alignment:.top,spacing:12) {
                                PostVolumePanel(title:"Before",side:1,height:310).environmentObject(post)
                                PostVolumePanel(title:"After",side:2,height:310).environmentObject(post)
                            }
                            summary
                            HStack(alignment:.top,spacing:12) {
                                PostVolumePanel(title:"Raw membrane image",side:0,height:200).environmentObject(post)
                                PostVolumePanel(title:"Changed voxels",side:7,height:200).environmentObject(post)
                            }
                            Text("Green: added foreground · magenta: removed · amber: reassigned cell ID. Colors in before/after refer to the same cell IDs.").font(.caption2).foregroundStyle(.secondary)
                            cellChanges
                            if let result = post.resultB {
                                DisclosureGroup("Saved refinement settings & provenance") {
                                    Text(post.settingsText(result)).font(.system(size:10,design:.monospaced)).textSelection(.enabled)
                                    Button("Reveal saved outputs") { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath:result.folder)]) }
                                }
                            }
                        } else {
                            ContentUnavailableView("Choose an image stack",systemImage:"square.stack.3d.up",description:Text("Use the current segmentation stack, choose an experiment stack, or open a raw TIFF."))
                        }
                        DisclosureGroup("Run log",isExpanded:$showLog) {
                            Text(post.log.isEmpty ? "No run log." : post.log).font(.system(size:10,design:.monospaced)).textSelection(.enabled)
                        }
                    }.padding(.leading,8)
                }.frame(minWidth:560)
            }
        }.padding(20)
        .onAppear {
            post.comparisonMethod = "refined"
            if post.source == nil, let source = segmentation.source, !post.busy { post.adoptPreparedSource(source,raw:segmentation.raw) }
            if let source = post.source {
                stackID = atlas.dataset?.samples.first(where:{$0.sourceImage == source.path})?.id ?? ""
                if !post.busy { refreshHistory(preferred:segmentation.resultA?.id) }
            }
        }
        .onChange(of:stackID) {
            guard !post.busy, let sample = atlas.dataset?.samples.first(where:{$0.id == stackID}), sample.sourceImage != post.source?.path else { return }
            let path = URL(fileURLWithPath:sample.sourceLabels).deletingLastPathComponent().appendingPathComponent("settings.json")
            let json = (try? JSONSerialization.jsonObject(with:Data(contentsOf:path))) as? [String:Any]
            let spacing = ((json?["config"] as? [String:Any]) ?? json)?["spacing_zyx_um"] as? [Double]
            post.prepare(sample.sourceImage,baseline:sample.sourceLabels,spacing:spacing)
        }
        .alert("Post processing failed",isPresented:Binding(get:{post.error != nil && atlas.activeTab == "postprocessing"},set:{if !$0 {post.error=nil}})) {Button("OK") {post.error=nil}} message:{Text(post.error ?? "")}
    }
    var inputControls: some View {
        VStack(alignment:.leading,spacing:4) {
            HStack {
                Picker("Stack",selection:$stackID) { Text("Choose stack…").tag("");ForEach(atlas.dataset?.samples ?? []) {Text($0.id).tag($0.id)} }.frame(maxWidth:230)
                Button("Use segmentation stack") {
                    if let source = segmentation.source {post.adoptPreparedSource(source,raw:segmentation.raw);refreshHistory(preferred:segmentation.resultA?.id);stackID=atlas.dataset?.samples.first(where:{$0.sourceImage == source.path})?.id ?? ""}
                }.disabled(segmentation.source == nil)
                Button("Open raw TIFF…") { post.open();stackID="" }
                Spacer()
            }.disabled(post.busy)
            if let source = post.source { Text(source.path).font(.caption.monospaced()).foregroundStyle(.secondary).lineLimit(1).help(source.path) }
            DisclosureGroup("Voxel calibration (µm)") {
                HStack {
                    spacingField("Z",value:$post.spacingZ);spacingField("Y",value:$post.spacingY);spacingField("X",value:$post.spacingX)
                    if post.calibrationChanged { Button("Apply calibration") {post.reloadCalibration()} }
                    Text("Changing calibration requires matching runs or re-importing labels.").font(.caption2).foregroundStyle(.secondary)
                }.disabled(post.busy)
            }.font(.caption)
        }
    }
    var sliceControls: some View {
        VStack(spacing:8) {
            HStack {
                Picker("Plane",selection:$post.axis) {Text("XY").tag(0);Text("XZ").tag(1);Text("YZ").tag(2)}.pickerStyle(.segmented).frame(width:150)
                Slider(value:$post.slice,in:0...1)
                if let source = post.source { Text("\(Int(post.slice*Double(source.nativeShape[post.axis]-1))+1)/\(source.nativeShape[post.axis])").font(.caption.monospacedDigit()) }
            }
            HStack {Text("Brightness").font(.caption);Slider(value:$post.brightness,in:0.2...4).frame(maxWidth:160);Toggle("Overlay",isOn:$post.overlay);Toggle("Contours",isOn:$post.boundaries)}.font(.caption)
        }
    }
    @ViewBuilder var summary: some View {
        if let info = post.resultB?.refinement {
            if (info.algorithmVersion ?? 1) < 2 {Text("Legacy trial: this used the earlier free-form fitter. Rerun with Conservative defaults to use shape-preserving refinement.").font(.caption).foregroundStyle(.orange)}
            HStack(spacing:20) {
                metric("Changed voxels",info.changedVoxels.formatted())
                metric("Protected cells",String(info.rejectedCells.count))
                if let agreement = post.agreement { metric("Foreground Dice",String(format:"%.3f",agreement.dice)) }
            }
            Text("Optimization loss: \(info.initialLoss,specifier:"%.5f") → \(info.finalLoss,specifier:"%.5f") before safety guards. Fit loss and before/after agreement do not measure accuracy.").font(.caption2).foregroundStyle(.secondary)
        } else { Text("Choose an original run, then run post processing or select an existing refined trial.").font(.caption).foregroundStyle(.secondary) }
    }
    var cellChanges: some View {
        let before = post.analysisCells(side: 1).filter { cell in !post.excludedLabels(side: 2).contains(cell.label) }
        let after = Dictionary(uniqueKeysWithValues:post.analysisCells(side: 2).map {($0.label,$0)})
        return VStack(alignment:.leading,spacing:6) {
            Text("Cell volume changes · click to inspect").font(.headline)
            HStack {Text("Cell").frame(width:70,alignment:.leading);Spacer();Text("Before µm³").frame(width:90,alignment:.trailing);Text("After µm³").frame(width:90,alignment:.trailing);Text("Change").frame(width:75,alignment:.trailing);Text("Guard").frame(width:40)}.font(.caption.bold())
            ForEach(before) { cell in
                Button {post.select(label:cell.label,side:1,center:true)} label: {
                    HStack {
                        Text(String(cell.label)).frame(width:70,alignment:.leading);Spacer()
                        Text(String(format:"%.0f",cell.volume)).frame(width:90,alignment:.trailing)
                        Text(after[cell.label].map {String(format:"%.0f",$0.volume)} ?? "—").frame(width:90,alignment:.trailing)
                        Text(after[cell.label].map {String(format:"%+.1f%%",100*($0.volume-cell.volume)/max(cell.volume,1e-9))} ?? "—").frame(width:75,alignment:.trailing)
                        Image(systemName:post.resultB?.refinement?.rejectedCells.contains(cell.label) == true ? "shield.fill" : "minus").frame(width:40)
                    }.font(.caption.monospacedDigit()).padding(6).background(post.selectedA == cell.label ? Color.orange.opacity(0.2) : Color.clear,in:RoundedRectangle(cornerRadius:5))
                }.buttonStyle(.plain)
            }
        }
    }
    func metric(_ label:String,_ value:String) -> some View {VStack(alignment:.leading) {Text(value).font(.title3.bold().monospacedDigit());Text(label).font(.caption2).foregroundStyle(.secondary)}}
    func clearAfter() {post.resultB=nil;post.labelsB=[];post.agreement=nil;post.selectedB=0}
    func spacingField(_ label:String,value:Binding<Double>) -> some View {HStack {Text(label);TextField(label,value:value,format:.number).textFieldStyle(.roundedBorder).frame(width:70)}}
    func selectOriginal(_ id:String, preferredAfter:String? = nil) {
        guard !post.busy else {return}
        clearAfter()
        guard let result = originals.first(where:{$0.id == id}) else {post.resultA=nil;post.labelsA=[];post.selectedA=0;return}
        post.choose(result,side:1)
        if let child = trials.first(where:{$0.id == preferredAfter}) ?? trials.first {post.showRefinement(child);post.compare()}
    }
    func refreshHistory(preferred:String? = nil) {
        guard !post.busy else {return}
        let current = post.resultA?.id
        let child = post.resultB?.id
        post.loadRunHistory()
        let id = originals.first(where:{$0.id == current})?.id ?? originals.first(where:{$0.id == preferred})?.id ?? originals.first?.id ?? ""
        selectOriginal(id,preferredAfter:child)
    }
    func restoreSettings() {
        guard let result = post.resultB else {return}
        do {
            guard let json = try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:result.folder).appendingPathComponent("result.json"))) as? [String:Any] else {throw DataError.invalid("Invalid saved refinement settings")}
            post.refinementSettings = try RefinementSettings.fromJSON(json["settings"] as? [String:Any] ?? [:])
        } catch {post.error=error.localizedDescription}
    }
}

struct PostVolumePanel: View {
    @EnvironmentObject var store: SegmentationStore
    let title:String
    let side:Int
    let height:CGFloat
    var body: some View {
        VStack(alignment:.leading,spacing:6) {
            Text(title).font(.headline)
            GeometryReader {geo in
                if let source = store.source, let image = store.image(side:side) {
                    let axes = store.axis == 0 ? (2,1) : store.axis == 1 ? (2,0) : (1,0)
                    let aspect = Double(source.nativeShape[axes.0])*source.spacing[axes.0]/(Double(source.nativeShape[axes.1])*source.spacing[axes.1])
                    let width=min(geo.size.width,geo.size.height*aspect),height=width/aspect
                    Image(decorative:image,scale:1).resizable().interpolation(.none).frame(width:width,height:height).contentShape(Rectangle())
                        .gesture(SpatialTapGesture().onEnded {tap in
                            guard side == 1 || side == 2 else {return}
                            let labels = side == 1 ? store.labelsA : store.labelsB
                            guard !labels.isEmpty else {return}
                            let (w,h,_) = store.dimensions()
                            let x=min(w-1,max(0,Int(tap.location.x/width*Double(w)))),y=min(h-1,max(0,Int(tap.location.y/height*Double(h))))
                            store.select(label:Int(labels[store.index(x,y)]),side:side)
                        }).position(x:geo.size.width/2,y:geo.size.height/2)
                    if (side == 1 && store.resultA == nil) || ((side == 2 || side == 7) && store.resultB == nil) {
                        Text(side == 1 ? "Select an original segmentation" : "Awaiting refined result").font(.caption).foregroundStyle(.white).padding(8).background(.black.opacity(0.7)).frame(maxWidth:.infinity,maxHeight:.infinity)
                    }
                }
            }.frame(height:height).background(.black,in:RoundedRectangle(cornerRadius:8)).clipped()
            if side == 1 || side == 2 {
                let result = side == 1 ? store.resultA : store.resultB
                Text(result.map {store.methodName($0.method)+" · "+$0.title} ?? "No selected run").font(.caption2).foregroundStyle(.secondary).lineLimit(2)
            }
        }.frame(maxWidth:.infinity)
    }
}
