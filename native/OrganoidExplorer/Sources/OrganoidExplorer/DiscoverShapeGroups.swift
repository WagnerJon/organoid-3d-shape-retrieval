import SwiftUI

private struct MethodShapeMatches: Identifiable {
    let method: Method
    let matches: [Neighbor]
    var id: String { method.id }
}

struct DiscoverShapeGroupsView: View {
    @EnvironmentObject var store: ExplorerStore
    @State private var mode = "Similar cells"
    @State private var scope = "Primary methods"
    @State private var groupMethod = "point_mae_adapted_pose_average_raw"
    @State private var minimum = 10
    @State private var samples = 5
    @State private var groupSelection = "eom"
    @State private var allowSingle = false
    @State private var commonOnly = true
    @State private var matches: [MethodShapeMatches] = []
    @State private var groups: ShapeAnalysisResult?
    @State private var loading = false
    @State private var error: String?
    @State private var reset = 0
    @State private var linkCameras = true
    @StateObject private var cameras = CameraLink()
    private let primary: Set<String> = ["morph_all", "morph_adjusted", "spatial_robust", "spatial_adjusted", "spherical_harmonics", "spherical_harmonics_adjusted",
        "allen_head_decoder_raw", "allen_head_decoder_adjusted", "point_mae_adapted_pose_average_raw", "point_mae_adapted_pose_average_adjusted", "shapeembed3d_invariant_raw", "shapeembed3d_invariant_adjusted"]
    var availableMethods: [Method] {
        (store.dataset?.methods ?? []).filter { store.level == "Cells" || ($0.scope != "cells" && !["shape_hybrid95", "precomputed"].contains($0.metric)) }
    }
    var methods: [Method] {
        availableMethods.filter {
            if scope == "Validation reference" { return primary.contains($0.id) || $0.family == "validation" || $0.id == "cajal_qgw" }
            return $0.family != "validation" && (scope == "All methods" || (scope == "Adjusted" ? $0.id.contains("adjusted") : primary.contains($0.id)))
        }
    }
    var cohort: [Specimen] {
        (commonOnly || scope == "Validation reference") ? store.filtered.filter { item in methods.allSatisfy { item.representations[$0.id] != nil } } : store.filtered
    }
    var query: Specimen? { store.analysisSelected }
    var cohortKey: String {
        "\(store.folder?.path ?? "")|\(store.level)|\(store.analysisRevision)|\(scope)|\(commonOnly)|" + cohort.map(\.id).joined(separator: ",")
    }
    var neighborsKey: String { cohortKey + "|\(store.selectedID ?? "")|\(store.excludeSameStack)" }
    var groupsKey: String { cohortKey + "|\(groupMethod)|\(minimum)|\(samples)|\(groupSelection)|\(allowSingle)|\(mode)" }
    var sharedSpan: Double {
        let specimens = ([query].compactMap { $0 } + matches.flatMap { $0.matches.prefix(2).map(\.item) })
        if store.level == "Cells" { return (specimens.compactMap { $0.metrics["major_axis_um"] }.max() ?? 20) * 1.4 }
        let ids = Set(specimens.map(\.sample))
        return store.dataset?.samples.filter { ids.contains($0.id) }.flatMap(\.extent).max() ?? 80
    }
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Discover shape groups").font(.system(size: 26, weight: .semibold))
                Text("Compare actual shapes using each method’s full comparison distance. Automatic groups are exploratory and can leave cells unassigned.")
                    .foregroundStyle(.secondary)
                HStack {
                    Picker("View", selection: $mode) {
                        Text("Similar shapes by method").tag("Similar cells")
                        Text("Discover groups").tag("Discover groups")
                    }.pickerStyle(.segmented).frame(width: 400)
                    Picker("Methods", selection: $scope) {
                        Text("Primary methods").tag("Primary methods")
                        Text("Adjusted methods").tag("Adjusted")
                        Text("All methods").tag("All methods")
                        if store.level == "Cells" { Text("Validation reference").tag("Validation reference") }
                    }.frame(width: 250)
                    Toggle("Same objects across methods", isOn: $commonOnly).toggleStyle(.checkbox)
                        .disabled(scope == "Validation reference")
                }
                HStack {
                    Text("\(cohort.count) eligible \(store.level.lowercased()) · \(methods.count) methods").font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Toggle("Link rotation & zoom", isOn: $linkCameras).toggleStyle(.checkbox)
                        .onChange(of: linkCameras) { cameras.enabled = linkCameras }
                    Button("Reset shapes") { reset += 1 }
                }
                if cohort.isEmpty {
                    ContentUnavailableView("No shared objects", systemImage: "circle.dashed", description: Text("Change the filters or turn off ‘Same objects across methods’."))
                } else if mode == "Similar cells" {
                    similarShapes
                } else {
                    discoveredGroups
                }
                Text("Distances are specific to each method and cannot be compared numerically across methods. Adjusted spaces control for volume, elongation and sphericity. Excluded cells never enter the cohort. Galleries use original segmented surfaces, centered at a shared physical scale; linked cameras preserve orientation differences rather than aligning shapes automatically.")
                    .font(.caption).foregroundStyle(.secondary)
            }.padding(20)
        }
        .onAppear { ensureMethod(); cameras.enabled = linkCameras }
        .onChange(of: scope) { ensureMethod() }
        .onChange(of: store.level) { ensureMethod() }
        .task(id: neighborsKey) {
            let candidates = cohort, query = query, methods = methods, other = store.excludeSameStack
            matches = []
            guard let query else { return }
            let computed = await Task.detached(priority: .userInitiated) {
                methods.map { method in
                    MethodShapeMatches(method: method, matches: rankNeighbors(query: query,
                        candidates: candidates.filter { !other || $0.sample != query.sample }, method: method, count: 5))
                }
            }.value
            guard !Task.isCancelled else { return }
            matches = computed
        }
        .task(id: groupsKey) {
            groups = nil; error = nil; loading = false
            guard mode == "Discover groups", let folder = store.folder, !cohort.isEmpty else { return }
            loading = true
            do {
                let value = try await ShapeAnalysis.run(folder: folder, method: groupMethod, level: store.level,
                                                       action: "groups", ids: cohort.map(\.id), minimum: minimum, samples: samples,
                                                       selection: groupSelection, allowSingle: groupSelection == "eom" && allowSingle)
                guard !Task.isCancelled else { return }
                groups = value
            } catch { if !Task.isCancelled { self.error = error.localizedDescription } }
            if !Task.isCancelled { loading = false }
        }
    }
    func ensureMethod() {
        if store.level != "Cells" && scope == "Validation reference" { scope = "Primary methods" }
        if !methods.contains(where: { $0.id == groupMethod }), let first = methods.first { groupMethod = first.id }
    }
    var similarShapes: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                if let query {
                    ShapeCellPreview(item: query, title: "Query · \(query.id)", span: sharedSpan, cameras: cameras, reset: reset)
                        .frame(width: 245)
                }
                VStack(alignment: .leading, spacing: 10) {
                    Text("Closest shapes to \(query?.id ?? "the selected cell")").font(.title3.bold())
                    Text("Select any cell in a plot or table, then compare the top matches returned by every method. The first two appear as rotatable surfaces; additional matches are listed below.").foregroundStyle(.secondary)
                    Toggle("Other stacks only", isOn: $store.excludeSameStack).toggleStyle(.checkbox)
                    if let first = cohort.first, query == nil || !cohort.contains(where: { $0.id == query?.id }) {
                        Button("Use a query from this cohort") { store.select(first.id) }
                    }
                }
            }
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 450), spacing: 14)], spacing: 14) {
                ForEach(matches) { entry in
                    VStack(alignment: .leading, spacing: 10) {
                        Text(entry.method.name).font(.headline)
                        Text("\(cohort.filter { $0.representations[entry.id] != nil }.count) candidates · \(entry.method.metric)")
                            .font(.caption).foregroundStyle(.secondary)
                        if let query, query.representations[entry.id] == nil {
                            Text("No embedding for this query in this method.").foregroundStyle(.orange)
                        } else if entry.matches.isEmpty {
                            Text("No eligible matches with these filters.").foregroundStyle(.secondary)
                        } else {
                            HStack(spacing: 10) {
                                ForEach(Array(entry.matches.prefix(2))) { neighbor in
                                    ShapeCellPreview(item: neighbor.item, title: "\(neighbor.id) · \(String(format: "%.4g", neighbor.distance))",
                                                     span: sharedSpan, cameras: cameras, reset: reset, methodID: entry.id)
                                }
                            }
                            HStack {
                                ForEach(Array(entry.matches.dropFirst(2))) { neighbor in
                                    Button(neighbor.id) { compare(neighbor.item, method: entry.method) }
                                }
                                Spacer()
                                Button("Mark nearest 5") {
                                    var ids = Set(entry.matches.map(\.id))
                                    if let query { ids.insert(query.id) }
                                    store.markPlotItems(ids)
                                }.disabled(store.markedOnly)
                            }.font(.caption)
                        }
                    }.padding(14).background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 12))
                }
            }
        }
    }
    var discoveredGroups: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Picker("Method", selection: $groupMethod) { ForEach(methods) { Text($0.name).tag($0.id) } }.frame(maxWidth: 440)
                Picker("Minimum group", selection: $minimum) { ForEach([5, 10, 15, 25, 40], id: \.self) { Text("\($0) objects").tag($0) } }.frame(width: 200)
                Picker("Density neighbors", selection: $samples) { ForEach([3, 5, 10, 15], id: \.self) { Text("\($0)").tag($0) } }.frame(width: 190)
            }
            HStack {
                Picker("Grouping detail", selection: $groupSelection) {
                    Text("Balanced groups").tag("eom")
                    Text("Fine groups").tag("leaf")
                }.frame(width: 300)
                Toggle("Allow one broad group", isOn: $allowSingle).toggleStyle(.checkbox)
                    .disabled(groupSelection != "eom")
                    .help("Can select the parent of several smaller groups. Off by default so those groups can be explored.")
            }
            Text("HDBSCAN uses the full method distance, without a fixed group count or 2D projection. Balanced groups selects stable branches; Fine groups selects smaller terminal branches and may leave more objects unassigned. No groups is a valid result. Changing filters refits groups; saved analysis labels stay unchanged. ‘Other stacks only’ affects query matching, not fitting.")
                .font(.caption).foregroundStyle(.secondary)
            if loading { ProgressView("Discovering shape groups…") }
            if let error { Text(error).foregroundStyle(.orange).textSelection(.enabled) }
            if let groups {
                HStack {
                    Text("\(groups.groups?.count ?? 0) groups · \(groups.unassigned ?? 0) unassigned · \(groups.ids.count) objects").font(.headline)
                    Spacer()
                    if let method = methods.first(where: { $0.id == groupMethod }), let labels = groups.labels {
                        Button("Show discovered groups in 3D plot") {
                            PlotWindows.shared.open(method: method, store: store, threeD: true, cohort: Set(groups.ids),
                                                    groups: Dictionary(uniqueKeysWithValues: zip(groups.ids, labels)))
                        }
                    }
                    if let labels = groups.labels, labels.contains(-1) {
                        Button("Mark unassigned") { store.markPlotItems(Set(zip(groups.ids, labels).filter { $0.1 < 0 }.map { $0.0 })) }
                    }
                }
                if groups.groups?.isEmpty ?? true {
                    Text("No density groups were found at these settings. Try a smaller minimum group or fewer density neighbors, or use Similar shapes to explore continuous variation. These settings do not establish biological cell types.").foregroundStyle(.secondary)
                } else if groups.groups?.count == 1 {
                    Text("Only one density group was found. Try Fine groups, a smaller minimum group or fewer density neighbors. A single group does not mean all shapes are identical.").foregroundStyle(.secondary)
                }
                LazyVStack(spacing: 14) {
                    ForEach(groups.groups ?? []) { group in groupGallery(group) }
                }
            }
        }
    }
    func groupGallery(_ group: DiscoveredShapeGroup) -> some View {
        let ids = [group.representative] + group.nearest.prefix(3).map(\.id)
        let lookup = Dictionary(uniqueKeysWithValues: store.allItems.map { ($0.id, $0) })
        let specimens = ids.compactMap { lookup[$0] }
        let span = store.level == "Cells" ? (specimens.compactMap { $0.metrics["major_axis_um"] }.max() ?? 20) * 1.4
            : (store.dataset?.samples.filter { Set(specimens.map(\.sample)).contains($0.id) }.flatMap(\.extent).max() ?? 80)
        return VStack(alignment: .leading, spacing: 10) {
            HStack {
                Circle().fill(clusterColors[group.label % clusterColors.count]).frame(width: 9, height: 9)
                Text("Group \(group.label + 1) · \(group.members.count) objects").font(.headline)
                Text("Mean membership strength \(group.membershipStrength, specifier: "%.2f")").font(.caption).foregroundStyle(.secondary)
                    .help("HDBSCAN membership strength is not biological confidence or a resampling stability test.")
                Spacer()
                Button("Mark group") { store.markPlotItems(Set(group.members)) }
            }
            Text("Representative = smallest mean within-group distance. The other shapes are its closest group members.").font(.caption).foregroundStyle(.secondary)
            HStack(spacing: 12) {
                ForEach(specimens) { item in
                    ShapeCellPreview(item: item, title: (item.id == group.representative ? "Representative · " : "Similar · ") + item.id,
                                     span: span, cameras: cameras, reset: reset, methodID: groupMethod)
                }
            }
        }.padding(14).background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 12))
    }
    func compare(_ item: Specimen, method: Method) {
        if store.learnedMethods.contains(where: { $0.id == method.id }) { store.method = method.id }
        store.shapeComparisonMethodID = method.id
        store.compared = item; store.showCompare = true
    }
}

struct ShapeCellPreview: View {
    @EnvironmentObject var store: ExplorerStore
    let item: Specimen
    let title: String
    let span: Double
    let cameras: CameraLink
    let reset: Int
    var methodID: String? = nil
    @State private var error: String?
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.caption.monospaced()).lineLimit(2)
            if let folder = store.folder, let sample = store.dataset?.samples.first(where: { $0.id == item.sample }) {
                let cells = store.level == "Cells" ? [item] : store.dataset?.cells.filter { $0.sample == item.sample && $0.includedInAnalysis } ?? []
                ReconstructionView(folder: folder, sample: sample, cells: cells, selectedLabel: item.label,
                                   isolate: store.level == "Cells", hideExcluded: true, reset: reset, centered: true,
                                   cameraSpan: span, link: cameras, onError: { error = $0 })
                    .id(item.id).frame(height: 150).clipShape(RoundedRectangle(cornerRadius: 8))
            }
            if let error { Text(error).font(.caption2).foregroundStyle(.orange).lineLimit(2).help(error) }
            HStack {
                Button("Use as query") { store.select(item.id) }
                Spacer()
                Button("Compare") { store.shapeComparisonMethodID = methodID; store.compared = item; store.showCompare = true }
                    .disabled(store.analysisSelected == nil || store.selectedID == item.id)
            }.font(.caption2)
        }.frame(maxWidth: .infinity)
    }
}
