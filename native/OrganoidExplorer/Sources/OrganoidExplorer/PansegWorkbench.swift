import SwiftUI
import AppKit

struct PansegWorkbenchView: View {
    @EnvironmentObject var atlas: ExplorerStore
    @EnvironmentObject var store: SegmentationStore
    @State private var stackID = ""
    @State private var showLog = false
    var models: [PansegModel] {
        let path = store.root.appendingPathComponent("config/panseg_models.json")
        return (try? JSONDecoder().decode([PansegModel].self,from:Data(contentsOf:path))) ?? []
    }
    var body: some View {
        VStack(alignment:.leading,spacing:12) {
            header
            inputControls
            presets
            if store.busy { progressPanel }
            actions
            HSplitView {
                ScrollView {
                    VStack(spacing:12) {predictionControls;preprocessingControls;watershedControls;clusteringControls}
                        .disabled(store.busy).padding(.trailing,8)
                }.frame(minWidth:360,idealWidth:400,maxWidth:480)
                ScrollView {
                    VStack(alignment:.leading,spacing:14) {
                        if store.source != nil {
                            sliceControls
                            HStack(alignment:.top,spacing:10) {
                                PansegVolumePanel(title:"Raw microscopy",side:0)
                                PansegVolumePanel(title:"Boundary probability",side:6)
                                PansegVolumePanel(title:"Cell instances",side:5)
                            }
                            trialInspector
                        } else {
                            ContentUnavailableView("Load a raw microscopy stack",systemImage:"square.stack.3d.up",description:Text("Choose an experiment stack or open a single-channel 3D TIFF."))
                        }
                        DisclosureGroup("Run log",isExpanded:$showLog) {
                            Text(store.log.isEmpty ? "No active run log." : store.log).font(.system(size:10,design:.monospaced)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading)
                        }
                    }.padding(.leading,8)
                }.frame(minWidth:380)
            }
        }.padding(20)
        .onAppear {
            store.refreshPresets()
            if let source = store.source { stackID = atlas.dataset?.samples.first(where:{$0.sourceImage == source.path})?.id ?? "" }
            else if !store.busy { stackID = atlas.selected?.sample ?? atlas.dataset?.samples.first?.id ?? "" }
        }
        .onChange(of:stackID) {
            guard let sample = atlas.dataset?.samples.first(where:{$0.id == stackID}), !store.busy, sample.sourceImage != store.source?.path else { return }
            let path = URL(fileURLWithPath:sample.sourceLabels).deletingLastPathComponent().appendingPathComponent("settings.json")
            let json = (try? JSONSerialization.jsonObject(with:Data(contentsOf:path))) as? [String:Any]
            let spacing = ((json?["config"] as? [String:Any]) ?? json)?["spacing_zyx_um"] as? [Double]
            store.prepare(sample.sourceImage,baseline:sample.sourceLabels,spacing:spacing)
        }
        .alert("PanSeg action failed",isPresented:Binding(get:{store.error != nil && atlas.activeTab == "panseg"},set:{if !$0 {store.error=nil}})) { Button("OK") {store.error=nil} } message:{Text(store.error ?? "")}
    }
    var header: some View {
        HStack {
            VStack(alignment:.leading,spacing:4) {
                Text("PanSeg try out").font(.system(size:26,weight:.semibold))
                Text("Tune prediction and instance segmentation, inspect the result, then transfer a working run.").foregroundStyle(.secondary)
            }
            Spacer()
            if store.busy { Button("Cancel") {store.cancel()} }
        }
    }
    var inputControls: some View {
        VStack(alignment:.leading,spacing:8) {
            HStack {
                Picker("Raw stack",selection:$stackID) {
                    Text("Choose stack…").tag("")
                    ForEach(atlas.dataset?.samples ?? []) {Text($0.id).tag($0.id)}
                }.frame(width:230)
                Button("Open raw TIFF…") {store.open();stackID=""}
                Spacer()
                field("Z µm",$store.spacingZ);field("Y µm",$store.spacingY);field("X µm",$store.spacingX)
                if store.calibrationChanged {Button("Apply calibration") {store.reloadCalibration()}}
            }.disabled(store.busy)
            if let source = store.source {Text(source.path).font(.caption.monospaced()).foregroundStyle(.secondary).textSelection(.enabled)}
            Text(store.status).font(.caption).foregroundStyle(.secondary)
        }
    }
    var presets: some View {
        HStack {
            Menu("Load preset") {ForEach(store.presets) { p in Button(p.name) {store.trialSettings=p.settings} }}.disabled(store.presets.isEmpty)
            TextField("Preset name",text:$store.presetName).textFieldStyle(.roundedBorder).frame(maxWidth:210)
            Button("Save settings") {store.savePreset()}
            Button("Import JSON…") {store.importPreset()}
            Button("Export JSON…") {store.exportPreset()}
            Spacer()
            Button("Reset defaults") {store.trialSettings=PansegSettings()}
        }.disabled(store.busy)
    }
    var predictionControls: some View {
        GroupBox("1 · Boundary prediction") {
            VStack(alignment:.leading,spacing:9) {
                Picker("Model source",selection:$store.trialSettings.modelSource) {Text("PanSeg zoo").tag("zoo");Text("BioImage.IO U-Net").tag("bioimageio");Text("Custom U-Net").tag("custom")}
                if store.trialSettings.modelSource == "zoo" {
                    Picker("3D boundary model",selection:$store.trialSettings.model) {ForEach(models) {Text($0.id).tag($0.id)}}
                    if let model = models.first(where:{$0.id == store.trialSettings.model}) {
                        Text(model.description).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal:false,vertical:true)
                        Text("Training voxel size ZYX: \(model.resolution.map {String(format:"%.4g",$0)}.joined(separator:" / ")) µm").font(.caption)
                        Button("Match training voxel size") {
                            if let spacing = store.source?.spacing, model.resolution.count == 3 {
                                store.trialSettings.rescaleZ=spacing[0]/model.resolution[0];store.trialSettings.rescaleY=spacing[1]/model.resolution[1];store.trialSettings.rescaleX=spacing[2]/model.resolution[2]
                            }
                        }.disabled(store.source == nil)
                    }
                } else if store.trialSettings.modelSource == "bioimageio" {
                    TextField("Compatible U-Net model ID",text:$store.trialSettings.modelID).textFieldStyle(.roundedBorder)
                    Text("Uses PanSeg’s BioImage.IO-to-U-Net loader. Choose a compatible boundary model with a single raw input channel.").font(.caption2).foregroundStyle(.secondary)
                } else {
                    HStack {TextField("U-Net config YAML",text:$store.trialSettings.configPath);Button("Choose…") {store.openModelFile(weights:false)}}
                    HStack {TextField("Weights (optional override)",text:$store.trialSettings.weightsPath);Button("Choose…") {store.openModelFile(weights:true)}}
                    Text("Single raw input channel. Choose the boundary output channel below for a model with multiple outputs.").font(.caption2).foregroundStyle(.secondary)
                }
                HStack {Picker("Device",selection:$store.trialSettings.device) {Text("Auto GPU / CPU").tag("auto");Text("CPU").tag("cpu");Text("Apple MPS").tag("mps");Text("CUDA").tag("cuda")};field("Output channel",$store.trialSettings.boundaryChannel)}
                Toggle("Use GPU in Auto mode",isOn:$store.trialSettings.gpu).disabled(store.trialSettings.device != "auto")
                Toggle("Automatic bounded patches (up to 80 × 128 × 128)",isOn:$store.trialSettings.automaticPatch)
                HStack {field("Patch Z",$store.trialSettings.patchZ);field("Y",$store.trialSettings.patchY);field("X",$store.trialSettings.patchX)}.disabled(store.trialSettings.automaticPatch)
                Picker("Halo",selection:$store.trialSettings.haloMode) {Text("44 voxels on tiled axes").tag("tiled");Text("Derive from model").tag("model");Text("Manual").tag("manual")}
                if store.trialSettings.haloMode == "manual" {HStack {field("Halo Z",$store.trialSettings.haloZ);field("Y",$store.trialSettings.haloY);field("X",$store.trialSettings.haloX)}}
                Toggle("Single-patch batches (lower memory)",isOn:$store.trialSettings.singleBatch)
                Text("Zoo models have different training domains, including mouse embryo cells. Inspect transfer quality on your own stack.").font(.caption2).foregroundStyle(.secondary)
            }.padding(7).frame(maxWidth:.infinity,alignment:.leading)
        }
    }
    var workingGridDescription: String {
        guard let source = store.source else { return "Load a stack to see the working grid." }
        let factors = [store.trialSettings.rescaleZ,store.trialSettings.rescaleY,store.trialSettings.rescaleX]
        var counts: [String] = []
        for i in 0..<3 {
            let factor = factors[i]
            if factor.isFinite && factor > 0 && factor < 100 {
                let count = max(1,Int((Double(source.nativeShape[i])*factor).rounded()))
                counts.append(String(count))
            } else { counts.append("invalid") }
        }
        return "Working grid: " + counts.joined(separator:" × ") + " voxels"
    }
    var preprocessingControls: some View {
        GroupBox("Raw preprocessing") {
            VStack(alignment:.leading,spacing:9) {
                HStack {field("Resample Z",$store.trialSettings.rescaleZ);field("Y",$store.trialSettings.rescaleY);field("X",$store.trialSettings.rescaleX)}
                HStack {field("Gaussian σ",$store.trialSettings.gaussianSigma);field("Median radius",$store.trialSettings.medianRadius)}
                Text(workingGridDescription).font(.caption.monospacedDigit())
                Text("Factors change voxel counts; 1 keeps native resolution. Smoothing uses working voxels. Final labels return to the original raw grid for comparison. Upstream normalization runs inside the model.").font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal:false,vertical:true)
            }.padding(7).frame(maxWidth:.infinity,alignment:.leading)
        }
    }
    var watershedControls: some View {
        GroupBox("2 · Distance-transform watershed") {
            VStack(alignment:.leading,spacing:9) {
                HStack {field("Threshold",$store.trialSettings.threshold);field("Min. voxels",$store.trialSettings.minSize)}
                HStack {field("Seed σ",$store.trialSettings.sigmaSeeds);field("Weight σ",$store.trialSettings.sigmaWeights);field("Alpha",$store.trialSettings.alpha)}
                Toggle("Slice-by-slice watershed (2D over-segmentation)",isOn:$store.trialSettings.stacked)
                Toggle("Use calibrated voxel anisotropy",isOn:$store.trialSettings.anisotropy)
                Toggle("Suppress non-maximum seeds",isOn:$store.trialSettings.nonmax)
                HStack {Toggle("Blockwise 3D watershed",isOn:$store.trialSettings.blockwise).disabled(store.trialSettings.stacked);field("Threads",$store.trialSettings.threads)}
                Text("Threshold separates boundary probabilities. Seed smoothing affects cell splitting; alpha blends boundary and distance weights. Minimum size is measured on the working grid.").font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal:false,vertical:true)
            }.padding(7).frame(maxWidth:.infinity,alignment:.leading)
        }
    }
    var clusteringControls: some View {
        GroupBox("3 · Clustering & background") {
            VStack(alignment:.leading,spacing:9) {
                Picker("Algorithm",selection:$store.trialSettings.clustering) {Text("Watershed only").tag("watershed");Text("Multicut").tag("multicut");Text("GASP").tag("gasp");Text("Mutex watershed").tag("mutex_ws")}
                HStack {field("Beta",$store.trialSettings.beta);field("Post min. voxels",$store.trialSettings.postMinSize)}.disabled(store.trialSettings.clustering == "watershed")
                if store.trialSettings.clustering == "gasp" {Picker("Linkage",selection:$store.trialSettings.linkage) {ForEach(["average","sum","max","min","abs_max"],id:\.self) {Text($0).tag($0)}}}
                field("Raw foreground cutoff",$store.trialSettings.foreground)
                Text("Smaller beta favors merging; larger beta favors splitting. Cutoff 0 disables background masking; 0–1 uses displayed raw intensity after segmentation. No annotation masks enter prediction.").font(.caption2).foregroundStyle(.secondary).fixedSize(horizontal:false,vertical:true)
            }.padding(7).frame(maxWidth:.infinity,alignment:.leading)
        }
    }
    var actions: some View {
        VStack(alignment:.leading,spacing:8) {
            HStack {
                Button("Predict + segment") {store.runTrial()}.buttonStyle(.borderedProminent).disabled(!store.canRun || !store.pansegInstalled)
                Button("Re-segment saved boundaries") {store.runTrial(reuse:true)}.disabled(!store.canRun || !store.canReusePrediction)
                Spacer()
                Button("Transfer displayed run + settings to comparison") {if store.transferTrial() {atlas.activeTab="segmentation"}}.disabled(store.busy || store.trialResult?.prediction == nil || store.calibrationChanged)
            }
            Text(store.canReusePrediction ? "Saved boundary prediction matches these inference settings. Change watershed or clustering settings and re-segment quickly." : "Predict once to create reusable boundaries. Changing model, preprocessing or inference settings requires a new prediction.").font(.caption).foregroundStyle(.secondary)
            if !store.pansegInstalled {Text("Install the backend with scripts/setup_panseg.sh").font(.caption).foregroundStyle(.orange)}
        }
    }
    var sliceControls: some View {
        VStack(spacing:8) {
            HStack {
                Picker("Plane",selection:$store.axis) {Text("XY").tag(0);Text("XZ").tag(1);Text("YZ").tag(2)}.pickerStyle(.segmented).frame(width:150)
                Slider(value:$store.slice,in:0...1)
                if let source = store.source {
                    let axis = store.axis == 0 ? 0 : store.axis == 1 ? 1 : 2
                    Text("\(Int(store.slice*Double(source.nativeShape[axis]-1))+1)/\(source.nativeShape[axis])").font(.caption.monospacedDigit())
                }
            }
            HStack {
                Text("Brightness").font(.caption);Slider(value:$store.brightness,in:0.2...4).frame(maxWidth:140)
                Toggle("Overlay",isOn:$store.overlay);Toggle("Contours",isOn:$store.boundaries)
            }
        }
    }
    var progressPanel: some View {
        VStack(alignment:.leading,spacing:5) {HStack {Text(store.progressStage);Spacer();Text("\(Int(store.progress*100))%").monospacedDigit()};ProgressView(value:store.progress);Text("Stage-based progress; inference counts completed patches.").font(.caption2).foregroundStyle(.secondary)}.padding(10).background(Color.teal.opacity(0.08),in:RoundedRectangle(cornerRadius:8))
    }
    var trialInspector: some View {
        VStack(alignment:.leading,spacing:10) {
            HStack {
                Picker("Displayed run",selection:Binding(get:{store.trialResult?.id ?? ""},set:{ id in if let result = store.results.first(where:{$0.id == id}) {store.chooseTrial(result)}})) {
                    Text("No completed run").tag("")
                    ForEach(store.results.filter {$0.method == "panseg"}) {Text($0.title).tag($0.id)}
                }.disabled(store.busy)
                Button("Restore this run’s settings") {store.loadTrialSettings()}.disabled(store.busy || store.trialResult == nil)
            }
            if let result = store.trialResult {
                if result.prediction == nil {Text("This older/imported run has no reusable prediction record. Run a new trial to create transferable settings.").font(.caption).foregroundStyle(.secondary)}
                HStack {Text("\(result.analysisCells.count) main-organoid cells · \(Int(result.seconds)) s").font(.headline);if let p = result.prediction {Text("\(p.device) · working grid \(p.shape.map(String.init).joined(separator:" × "))").font(.caption)}}
                if store.trialSettingsChanged {Text("Controls have edits. The images and transfer button refer to the displayed run’s saved settings.").font(.caption).foregroundStyle(.orange)}
                ScrollView(.horizontal) {
                    HStack {ForEach(store.analysisCells(side: 5)) {cell in Button {
                        store.selectedTrial=cell.label
                        if let source = store.source {let axis = store.axis == 0 ? 0 : store.axis == 1 ? 1 : 2;store.slice=min(1,max(0,cell.centroid[axis]/Double(max(1,source.nativeShape[axis]-1))))}
                    } label:{VStack {Text("Cell \(cell.label)");Text("\(cell.volume,specifier:"%.0f") µm³").font(.caption);if cell.border {Text("Touches border").font(.caption2)}}.padding(8).background(store.selectedTrial == cell.label ? Color.orange.opacity(0.25) : Color.gray.opacity(0.1),in:RoundedRectangle(cornerRadius:6))}.buttonStyle(.plain)}}
                }
                DisclosureGroup("Displayed run’s exact settings") {Text(store.settingsText(result)).font(.system(size:10,design:.monospaced)).textSelection(.enabled)}
            }
        }
    }
    func field(_ title:String,_ value:Binding<Double>) -> some View {HStack(spacing:4) {Text(title).font(.caption);TextField(title,value:value,format:.number).textFieldStyle(.roundedBorder).frame(width:64)}}
    func field(_ title:String,_ value:Binding<Int>) -> some View {HStack(spacing:4) {Text(title).font(.caption);TextField(title,value:value,format:.number).textFieldStyle(.roundedBorder).frame(width:60)}}
}

struct PansegVolumePanel: View {
    @EnvironmentObject var store: SegmentationStore
    let title: String
    let side: Int
    var body: some View {
        VStack(alignment:.leading,spacing:8) {
            Text(title).font(.headline)
            GeometryReader {geo in
                if let source = store.source, let image = store.image(side:side) {
                    let axes = store.axis == 0 ? (2,1) : store.axis == 1 ? (2,0) : (1,0)
                    let ratio=Double(source.nativeShape[axes.0])*source.spacing[axes.0]/(Double(source.nativeShape[axes.1])*source.spacing[axes.1])
                    let width=min(geo.size.width,geo.size.height*ratio),height=width/ratio
                    Image(decorative:image,scale:1).resizable().interpolation(.none).frame(width:width,height:height).contentShape(Rectangle())
                        .gesture(SpatialTapGesture().onEnded {tap in
                            guard side == 5, !store.trialLabels.isEmpty else {return}
                            let (w,h,_) = store.dimensions();let x=min(w-1,max(0,Int(tap.location.x/width*Double(w)))),y=min(h-1,max(0,Int(tap.location.y/height*Double(h))))
                            store.selectedTrial=Int(store.trialLabels[store.index(x,y)])
                        }).position(x:geo.size.width/2,y:geo.size.height/2)
                    if side != 0 && (side == 6 ? store.trialBoundaries.isEmpty : store.trialResult == nil) {Text("Awaiting result").foregroundStyle(.white).padding(8).background(.black.opacity(0.7)).frame(maxWidth:.infinity,maxHeight:.infinity)}
                }
            }.frame(height:245).background(.black,in:RoundedRectangle(cornerRadius:8)).clipped()
            Text(side == 6 ? "Black 0 → white 1 · brightness is display only" : side == 5 ? "Click an instance to highlight it" : "Same calibrated grid as both result views").font(.caption2).foregroundStyle(.secondary)
        }.frame(maxWidth:.infinity)
    }
}
