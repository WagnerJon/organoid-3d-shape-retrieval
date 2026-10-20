import SwiftUI
import AppKit

@main struct ExplorerApp: App {
    @StateObject private var store = ExplorerStore()
    var body: some Scene {
        WindowGroup("Organoid Atlas") {
            GeometryReader { geometry in
                ContentView().environmentObject(store)
                    .frame(width: geometry.size.width, height: geometry.size.height)

            }
                .frame(minWidth: 1240, minHeight: 740)
                .onAppear {
                    NSApp.setActivationPolicy(.regular)
                    NSApp.activate(ignoringOtherApps: true)
                    guard store.dataset == nil, !store.loading else { return }
                    let argument = CommandLine.arguments.dropFirst().first
                    let bundledPath = Bundle.main.object(forInfoDictionaryKey: "ExplorerDatasetPath") as? String
                    let path = argument ?? UserDefaults.standard.string(forKey: "datasetPath") ?? bundledPath
                    if let path { store.load(URL(fileURLWithPath: path)) }
                }
        }
        .defaultSize(width: 1380, height: 790)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Open Experiment…") { store.openPanel() }.keyboardShortcut("o")
                Button("Export Visible / Marked Rows…") { store.exportSelection() }.keyboardShortcut("e", modifiers: [.command, .shift])
            }
        }
    }
}

struct ContentView: View {
    @StateObject private var segmentation = SegmentationStore()
    @StateObject private var postProcessing = SegmentationStore()
    @EnvironmentObject var store: ExplorerStore
    var body: some View {
        NavigationSplitView {
            sidebar.navigationSplitViewColumnWidth(min: 190, ideal: 215, max: 270)
        } detail: {
            if store.loading || (store.membershipBusy && store.dataset == nil) {
                ProgressView("Opening experiment…").frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if store.dataset != nil {
                TabView(selection: $store.activeTab) {
                    workspace.tabItem { Label("Explore", systemImage: "cube.transparent") }.tag("explore")
                    MembershipView().tabItem { Label("Organoid Membership", systemImage: "checkmark.circle") }.tag("membership")
                    MethodsComparisonView().tabItem { Label("Compare Methods", systemImage: "chart.xyaxis.line") }.tag("methods")
                    DiscoverShapeGroupsView().tabItem { Label("Discover Shape Groups", systemImage: "circle.hexagongrid") }.tag("discover")
                    SegmentationView().environmentObject(segmentation).tabItem { Label("Segmentation", systemImage: "square.stack.3d.up") }.tag("segmentation")
                    PansegWorkbenchView().environmentObject(segmentation).tabItem { Label("PanSeg Try Out", systemImage:"slider.horizontal.3") }.tag("panseg")
                    PostProcessingView(post:postProcessing).environmentObject(segmentation).tabItem { Label("Post Processing",systemImage:"wand.and.stars") }.tag("postprocessing")
                }
                .disabled(store.membershipBusy)
                .onChange(of: store.activeTab) { store.clusterFilter = -1 }
            } else {
                ContentUnavailableView {
                    Label("Your experiments, connected", systemImage: "cube.transparent")
                } description: {
                    Text("Open a prepared experiment to explore cell morphology, learned representations, raw microscopy and 3D surfaces together.")
                } actions: { Button("Open Experiment…") { store.openPanel() }.buttonStyle(.borderedProminent) }
            }
        }
        .tint(.teal)
        .toolbar {
            ToolbarItemGroup {
                Button { store.openPanel() } label: { Label("Open", systemImage: "folder") }
                Button { store.exportSelection() } label: { Label("Export rows", systemImage: "square.and.arrow.up") }.disabled(store.dataset == nil)
            }
        }
        .alert("Unable to complete action", isPresented: Binding(get: { store.error != nil }, set: { if !$0 { store.error = nil } })) {
            Button("OK") { store.error = nil }
        } message: { Text(store.error ?? "") }
        .sheet(isPresented: $store.showCompare, onDismiss: { store.shapeComparisonMethodID = nil }) { ComparisonSheet().environmentObject(store) }
    }
    var sidebar: some View {
        ScrollView {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 10) {
                if let logoURL = Bundle.main.url(forResource: "OrganoidAtlas", withExtension: "png"),
                   let logo = NSImage(contentsOf: logoURL) {
                    Image(nsImage: logo).resizable().scaledToFit().frame(width: 40, height: 40)
                        .accessibilityHidden(true)
                } else {
                    Image(systemName: "cube.transparent.fill").font(.title).foregroundStyle(.teal)
                }
                VStack(alignment: .leading) { Text("ORGANOID").font(.caption.weight(.bold)).tracking(2); Text("Atlas").font(.title2.weight(.semibold)) }
            }.padding(.top, 12)
            Text("EXPERIMENT EXPLORER").font(.caption2.weight(.semibold)).foregroundStyle(.secondary).tracking(1)
            if !["segmentation","panseg","postprocessing","membership"].contains(store.activeTab) {
            Picker("Explore", selection: $store.level) { Text("Cells").tag("Cells"); Text("Organoids").tag("Organoids") }
                .pickerStyle(.segmented).labelsHidden().onChange(of: store.level) { store.changeLevel() }
            }
            if let data = store.dataset {
                VStack(alignment: .leading, spacing: 5) {
                    Text("\(data.cells.filter(\.includedInAnalysis).count.formatted()) included cells").font(.title3.bold())
                    Text("\(data.samples.count) reconstructed stacks").foregroundStyle(.secondary)
                }
                Text("\(store.excludedCells.count) outside main organoids · excluded").font(.caption).foregroundStyle(.red)
                if store.membershipBusy {
                    ProgressView("Updating all analyses…").font(.caption)
                }
                Divider()
                if !["segmentation","panseg","postprocessing"].contains(store.activeTab) {
                TextField("Find a cell or stack", text: $store.search).textFieldStyle(.roundedBorder)
                Picker("Stack", selection: $store.sampleFilter) {
                    Text("All stacks").tag("All stacks")
                    ForEach(data.samples) { Text($0.id).tag($0.id) }
                }
                Toggle("QC-passing only", isOn: $store.qcOnly)
                if store.level == "Cells" {
                    Picker("Split", selection: $store.splitFilter) {
                        Text("All splits").tag("All splits")
                        ForEach(Array(Set(data.cells.map(\.split))).sorted(), id: \.self) { Text($0.capitalized).tag($0) }
                    }
                }
                Picker(store.activeTab == "methods" ? "Reference group" : "Learned group", selection: $store.clusterFilter) {
                    Text("All groups").tag(-1)
                    ForEach(0..<6) { Text("Group \($0+1)").tag($0) }
                }
                Divider()
                VStack(alignment: .leading, spacing: 9) {
                    Text("LINKED MARKING").font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
                    Text("Click a plot point to mark its row. Drag to mark a group.").font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                    Text("\(store.selection.count) marked").font(.headline)
                    Toggle("Show marked only", isOn: $store.markedOnly)
                    Button("Clear marking") { store.selection = []; store.markedOnly = false }
                }
                } else {
                    Text("Choose a raw stack, tune segmentation, then inspect linked slices and matched cells.")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal:false,vertical:true)
                }
                Spacer(minLength: 10)
                Text("Exploratory groups\nImage stacks ≠ independent specimens").font(.caption2).foregroundStyle(.secondary)
                Button("Methods & provenance") {
                    let alert = NSAlert(); alert.messageText = "About this experiment"; alert.informativeText = data.provenance + "\n\nTraditional distance: Euclidean on standardized features. Learned distance: cosine. Organoid vectors: mean of QC-passing cell vectors, normalized for learned features. Filters apply to both plots, table and similarity candidates."; alert.runModal()
                }.font(.caption)
            }
            Spacer(minLength: 0)
        }.padding(18)
        }.background(Color(nsColor: .controlBackgroundColor))
    }
    var workspace: some View {
        GeometryReader { geometry in
        let panelHeight = max(580, geometry.size.height - 112)
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(store.level == "Cells" ? "Cell shape explorer" : "Organoid shape explorer").font(.system(size: 26, weight: .semibold))
                    Text("\(store.filtered.count) visible · select anywhere to connect the evidence").foregroundStyle(.secondary)
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 3) {
                    Text(store.selected?.id ?? "No selection").font(.title3.monospaced().bold()).foregroundStyle(.teal)
                    Text(store.level == "Cells" ? "SELECTED CELL" : "SELECTED STACK").font(.caption2).foregroundStyle(.secondary)
                }
            }
            HSplitView {
                VStack(spacing: 12) {
                    HStack(spacing: 12) {
                        ScatterPanel(methodID: "traditional", title: "Traditional morphology")
                        ScatterPanel(methodID: store.method, title: "Learned representation")
                    }.frame(height: 230)
                    HStack {
                        Text("Learned space").font(.caption).foregroundStyle(.secondary)
                        Picker("Learned space", selection: $store.method) {
                            ForEach(store.learnedMethods) { Text($0.name).tag($0.id) }
                        }.labelsHidden().frame(maxWidth: 270)
                        Spacer()
                    }
                    DataTable().frame(height: max(100, panelHeight - 480))
                    SimilarityPanel().frame(height: 190)
                }.frame(minWidth: 490).padding(.trailing, 10)
                Inspector().frame(minWidth: 380, idealWidth: 460).padding(.leading, 10)
            }.frame(height: panelHeight)
        }.padding(20)
        }
    }
}

struct ScatterPanel: View {
    @EnvironmentObject var store: ExplorerStore
    let methodID, title: String
    var colorMethodID: String? = nil
    var cohortIDs: Set<String>? = nil
    var groupLabels: [String: Int]? = nil
    var showActions = true
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .top) {
                Text(title).font(.headline)
                Spacer(minLength: 2)
                if showActions, let method = store.dataset?.methods.first(where: { $0.id == methodID }) {
                    Button {
                        PlotWindows.shared.open(method: method, store: store, cohort: cohortIDs, groups: groupLabels, colorMethod: colorMethodID)
                    } label: { Image(systemName: "arrow.up.left.and.arrow.down.right") }
                        .help("Open this plot full screen")
                    Button(["precomputed", "shape_hybrid95"].contains(method.metric) ? "3D" : "3 PCs") {
                        PlotWindows.shared.open(method: method, store: store, threeD: true, cohort: cohortIDs, groups: groupLabels, colorMethod: colorMethodID)
                    }.help("Open a rotatable three-axis plot")
                }
            }
            if let method = store.dataset?.methods.first(where: { $0.id == methodID }) {
                let variance = store.level == "Cells" ? method.variance : method.stackVariance
                Text(method.projection == "mds" ? "Distance map · \(Int(variance.reduce(0,+)*100))% positive inertia · up to 6 groups" : "PCA · \(Int(variance.reduce(0,+)*100))% variance · up to 6 groups").font(.caption2).foregroundStyle(.secondary)
            }
            ScatterCanvas(methodID: methodID, colorMethodID: colorMethodID, cohortIDs: cohortIDs, groupLabels: groupLabels)
            HStack { Text(store.dataset?.methods.first(where: { $0.id == methodID })?.projection == "mds" ? "Axis 2 ↑" : "PC 2 ↑"); Spacer(); Text(store.dataset?.methods.first(where: { $0.id == methodID })?.projection == "mds" ? "Axis 1 →" : "PC 1 →") }.font(.caption2).foregroundStyle(.secondary)
        }.padding(12).background(.background, in: RoundedRectangle(cornerRadius: 10)).overlay(RoundedRectangle(cornerRadius: 10).stroke(.quaternary))
    }
}
struct PlotDot {
    let id: String
    let position: CGPoint
    let cluster: Int
}
struct ScatterCanvas: View {
    @EnvironmentObject var store: ExplorerStore
    let methodID: String
    var colorMethodID: String? = nil
    var cohortIDs: Set<String>? = nil
    var groupLabels: [String: Int]? = nil
    @State private var dragStart: CGPoint?
    @State private var dragEnd: CGPoint?
    var body: some View {
        GeometryReader { geo in
            let dots = makeDots(size: geo.size)
            Canvas { context, size in draw(context: &context, size: size, dots: dots) }
                .contentShape(Rectangle())
                .gesture(DragGesture(minimumDistance: 0).onChanged { value in
                    dragStart = value.startLocation; dragEnd = value.location
                }.onEnded { value in
                    finish(start: value.startLocation, end: value.location, dots: dots)
                    dragStart = nil; dragEnd = nil
                })
        }
    }
    func makeDots(size: CGSize) -> [PlotDot] {
        let items = store.filtered.filter { $0.representations[methodID] != nil && (cohortIDs?.contains($0.id) ?? true) }
        let points = items.map { $0.representations[methodID]!.point }
        let xs = points.map { $0[0] }, ys = points.map { $0[1] }
        let xmin = xs.min() ?? 0, xmax = xs.max() ?? 1
        let ymin = ys.min() ?? 0, ymax = ys.max() ?? 1
        let width = Double(size.width) - 36, height = Double(size.height) - 36
        return items.indices.map { i in
            let x = 18 + (points[i][0]-xmin)/max(xmax-xmin, 1e-9)*width
            let y = Double(size.height)-18-(points[i][1]-ymin)/max(ymax-ymin, 1e-9)*height
            let cluster = groupLabels.map { $0[items[i].id] ?? -1 } ?? items[i].representations[colorMethodID ?? methodID]?.cluster ?? -1
            return PlotDot(id: items[i].id, position: CGPoint(x:x,y:y), cluster: cluster)
        }
    }
    func draw(context: inout GraphicsContext, size: CGSize, dots: [PlotDot]) {
        for n in 0...4 {
            let y = 18 + CGFloat(n)/4*(size.height-36)
            var line = Path(); line.move(to: CGPoint(x:18,y:y)); line.addLine(to: CGPoint(x:size.width-18,y:y))
            context.stroke(line, with: .color(.secondary.opacity(0.13)), lineWidth: 1)
        }
        for dot in dots {
            let selected = dot.id == store.selectedID, marked = store.selection.contains(dot.id)
            let radius: CGFloat = selected ? 6.5 : (marked ? 4.5 : 3.2)
            let p = dot.position
            let rect = CGRect(x:p.x-radius,y:p.y-radius,width:radius*2,height:radius*2)
            let color: Color = dot.cluster < 0 ? .gray : clusterColors[dot.cluster % 6]
            let opacity = store.selection.isEmpty || marked ? 0.85 : 0.18
            context.fill(Path(ellipseIn: rect), with: .color(selected ? .orange : color.opacity(opacity)))
            if selected || marked {
                context.stroke(Path(ellipseIn: rect.insetBy(dx:-2,dy:-2)), with: .color(selected ? .primary : .teal), lineWidth: selected ? 2 : 1)
            }
        }
        if let start = dragStart, let end = dragEnd {
            let rect = rectangle(start, end)
            context.fill(Path(rect), with: .color(.teal.opacity(0.1)))
            context.stroke(Path(rect), with: .color(.teal), lineWidth: 1)
        }
    }
    func rectangle(_ a: CGPoint, _ b: CGPoint) -> CGRect {
        CGRect(x:min(a.x,b.x),y:min(a.y,b.y),width:abs(a.x-b.x),height:abs(a.y-b.y))
    }
    func finish(start: CGPoint, end: CGPoint, dots: [PlotDot]) {
        func distance(_ dot: PlotDot) -> CGFloat { hypot(dot.position.x-end.x,dot.position.y-end.y) }
        if hypot(start.x-end.x,start.y-end.y) < 5 {
            if let dot = dots.min(by: { distance($0) < distance($1) }), distance(dot) < 15 {
                store.markPlotItems([dot.id]); store.select(dot.id)
            } else { store.markPlotItems([]) }
        } else {
            let rect = rectangle(start,end)
            store.markPlotItems(Set(dots.filter { rect.contains($0.position) }.map(\.id)))
        }
    }
}
struct DataTable: View {
    @EnvironmentObject var store: ExplorerStore
    @State private var sortOrder = [KeyPathComparator(\Specimen.id)]
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack { Text(store.level == "Cells" ? "Cell measurements" : "Stack measurements").font(.headline); Spacer(); Text("Click a row to inspect in 3D").font(.caption).foregroundStyle(.secondary) }
            Table(store.measurementItems.sorted(using: sortOrder), selection: Binding(get: { store.selectedTableRows }, set: { store.selectTableRows($0) }), sortOrder: $sortOrder) {
                TableColumn("Object", value: \.id) { item in
                    HStack(spacing: 4) { if store.selection.contains(item.id) { Circle().fill(.teal).frame(width: 5,height: 5) }; Text(item.id).monospaced() }
                }.width(min: 85, ideal: 100)
                TableColumn("Volume µm³", value: \.volume) { Text($0.volume, format: .number.precision(.fractionLength(0))).monospacedDigit() }.width(min: 90, ideal: 100)
                TableColumn("Sphericity", value: \.sphericity) { Text($0.sphericity, format: .number.precision(.fractionLength(3))) }.width(min: 65, ideal: 80)
                TableColumn("Elongation", value: \.elongation) { Text($0.elongation, format: .number.precision(.fractionLength(2))) }.width(min: 65, ideal: 80)
                TableColumn("QC") { Text($0.qc ? "Pass" : "Flagged").foregroundStyle($0.qc ? Color.secondary : Color.orange).help($0.flags) }.width(50)
            }.tableStyle(.inset(alternatesRowBackgrounds: true))
        }
    }
}
struct SimilarityPanel: View {
    @EnvironmentObject var store: ExplorerStore
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack { Text("Find similar shapes").font(.headline); Spacer(); Toggle("Other stacks only", isOn: $store.excludeSameStack).font(.caption).toggleStyle(.checkbox) }
            HStack(alignment: .top, spacing: 16) {
                neighborList("Traditional", key: "traditional")
                Divider()
                neighborList("Learned", key: store.method)
            }
            Text("Full-vector distances · lower is closer · click a match for 3D comparison").font(.caption2).foregroundStyle(.secondary)
        }.padding(12).background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 10))
    }
    func neighborList(_ title: String, key: String) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            let metric = store.dataset?.methods.first(where: { $0.id == key })?.metric ?? "euclidean"
            Text(title + " · " + (metric == "euclidean" ? "Euclidean" : metric == "cosine" ? "cosine" : "shape distance")).font(.caption.weight(.semibold)).foregroundStyle(.secondary)
            let neighbors = store.neighbors(method: key, count: 4)
            if let query = store.analysisSelected, query.representations[key] == nil {
                Text("No embedding for this object in this method").font(.caption).foregroundStyle(.orange)
            } else if neighbors.isEmpty { Text("No eligible matches").font(.caption).foregroundStyle(.secondary) }
            ForEach(Array(neighbors.enumerated()), id: \.element.id) { index, neighbor in
                Button {
                    store.compared = neighbor.item; store.showCompare = true
                } label: {
                    HStack { Text("\(index+1)").foregroundStyle(.secondary); Text(neighbor.id).monospaced(); Spacer(); Text(String(format: "%.3g", neighbor.distance)).monospacedDigit(); Image(systemName: "square.split.2x1") }.font(.caption)
                }.buttonStyle(.plain).padding(.vertical, 3)
            }
        }.frame(maxWidth: .infinity, alignment: .leading)
    }
}
