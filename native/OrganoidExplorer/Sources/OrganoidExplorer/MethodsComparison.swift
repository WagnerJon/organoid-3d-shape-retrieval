import SwiftUI

struct MethodsComparisonView: View {
    @EnvironmentObject var store: ExplorerStore
    @State private var agreement: [MethodAgreement] = []
    @State private var calculating = false
    @State private var comparisonScope = "Learned shape benchmarks"
    @State private var commonOnly = true
    private let columns = Array(repeating: GridItem(.flexible(minimum: 240), spacing: 12, alignment: .top), count: 3)
    private let benchmarkFamilies: Set<String> = ["allen", "point_mae", "shapeembed3d"]
    private let benchmarkKeys: Set<String> = ["morph_all", "morph_adjusted", "spatial_robust", "spatial_adjusted", "spherical_harmonics", "spherical_harmonics_adjusted",
        "allen_head_decoder_raw", "allen_head_decoder_adjusted", "point_mae_adapted_pose_average_raw", "point_mae_adapted_pose_average_adjusted", "shapeembed3d_invariant_raw", "shapeembed3d_invariant_adjusted"]
    var methods: [Method] {
        (store.dataset?.methods ?? []).filter { method in
            let available = store.level == "Cells" || (method.scope != "cells" && !["shape_hybrid95", "precomputed"].contains(method.metric))
            let validationKeys: Set<String> = ["morph_all", "morph_adjusted", "spatial_robust", "spatial_adjusted", "spherical_harmonics", "spherical_harmonics_adjusted", "cajal_qgw", "cajal_gw_validation"]
            let visible: Bool
            if comparisonScope == "Learned shape benchmarks" { visible = benchmarkKeys.contains(method.id) }
            else if benchmarkFamilies.contains(comparisonScope) { visible = method.family == comparisonScope || ["morph_all", "morph_adjusted"].contains(method.id) }
            else if comparisonScope == "Validation benchmarks" { visible = validationKeys.contains(method.id) || benchmarkKeys.contains(method.id) }
            else if comparisonScope == "All methods" { visible = method.family != "validation" }
            else if comparisonScope == "Shape and traditional" { visible = (method.family != nil && method.family != "validation") || method.id == "traditional" || method.id == "pointnet_shape128" }
            else { visible = method.id.contains("adjusted") }
            return available && visible
        }
    }
    var comparisonItems: [Specimen] {
        (commonOnly || comparisonScope == "Validation benchmarks") ? store.filtered.filter { item in methods.allSatisfy { item.representations[$0.id] != nil } } : store.filtered
    }
    var comparisonQuery: Specimen? {
        guard let query = store.analysisSelected, comparisonItems.contains(where: { $0.id == query.id }) else { return nil }
        return query
    }
    var plotCohortIDs: Set<String>? { (commonOnly || comparisonScope == "Validation benchmarks") ? Set(comparisonItems.map(\.id)) : nil }
    var referenceName: String { methods.first { $0.id == store.referenceMethod }?.name ?? "reference" }
    var calculationKey: String {
        [store.referenceMethod, store.selectedID ?? "", String(store.excludeSameStack), store.level, comparisonScope, String(commonOnly),
         store.filtered.map(\.id).joined(separator: ","), store.folder?.path ?? "", store.analysisRevision.uuidString].joined(separator: "|")
    }
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                heading
                HStack {
                    Picker("Compare", selection: $comparisonScope) {
                        if store.level == "Cells" {
                            Text("Learned shape benchmarks").tag("Learned shape benchmarks")
                            Text("Allen · released vs adapted").tag("allen")
                            Text("Point-MAE · pose and adaptation").tag("point_mae")
                            Text("ShapeEmbedLite3D · invariant vs CNN").tag("shapeembed3d")
                        }
                        Text("Shape and traditional").tag("Shape and traditional")
                        Text("After size / shape adjustment").tag("Adjusted")
                        Text("All methods").tag("All methods")
                        if store.level == "Cells" { Text("Validation benchmarks").tag("Validation benchmarks") }
                    }.frame(width: 390)
                    Toggle("Same objects across plots", isOn: $commonOnly).toggleStyle(.checkbox)
                        .disabled(comparisonScope == "Validation benchmarks")
                    Spacer()
                }
                controls
                if comparisonItems.isEmpty {
                    Text("No objects have results for every visible method. Change the filters or turn off ‘Same objects across plots’ to inspect available results.")
                        .font(.callout).foregroundStyle(.orange)
                } else if comparisonQuery == nil {
                    HStack {
                        Text("The selected object is outside this comparison cohort.").foregroundStyle(.secondary)
                        Button("Select an available object") { if let first = comparisonItems.first { store.select(first.id) } }
                    }.font(.caption)
                }
                legend
                LazyVGrid(columns: columns, spacing: 14) {
                    ForEach(methods) { method in methodCard(method) }
                }
                explanation
            }.padding(20)
        }
        .onChange(of: comparisonScope) { ensureReference() }
        .onChange(of: store.level) { ensureReference() }
        .onAppear { ensureReference() }
        .task(id: calculationKey) {
            calculating = true
            let methods = methods, reference = store.referenceMethod, items = comparisonItems
            let query = comparisonQuery, otherStacks = store.excludeSameStack
            let result = await Task.detached(priority: .userInitiated) {
                compareMethods(methods: methods, reference: reference, items: items, query: query, otherStacksOnly: otherStacks)
            }.value
            guard !Task.isCancelled else { return }
            agreement = result; calculating = false
        }
    }
    func ensureReference() {
        if store.level != "Cells" && (comparisonScope == "Validation benchmarks" || comparisonScope == "Learned shape benchmarks" || benchmarkFamilies.contains(comparisonScope)) { comparisonScope = "Shape and traditional" }
        if !methods.contains(where: { $0.id == store.referenceMethod }), let first = methods.first {
            store.referenceMethod = first.id
        }
        if comparisonScope == "Validation benchmarks", comparisonQuery == nil, let first = comparisonItems.first { store.select(first.id) }
    }
    var heading: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Compare shape and feature spaces").font(.system(size: 26, weight: .semibold))
                Text("\(comparisonItems.count) \(store.level.lowercased()) · \(methods.count) spaces · one linked selection")
                    .foregroundStyle(.secondary)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 5) {
                Text(store.selectedID ?? "Select a point").font(.headline.monospaced()).foregroundStyle(.teal)
                Button("Inspect in 3D") { store.activeTab = "explore" }.disabled(store.selected == nil)
            }
        }
    }
    var controls: some View {
        HStack(spacing: 16) {
            Picker("Reference", selection: $store.referenceMethod) {
                ForEach(methods) { Text($0.name).tag($0.id) }
            }.frame(maxWidth: 310)
                .onChange(of: store.referenceMethod) { store.clusterFilter = -1 }
            Picker("Default group colors", selection: $store.sharedMethodColors) {
                Text("Reference groups").tag(true)
                Text("Each method’s groups").tag(false)
            }.pickerStyle(.segmented).labelsHidden().frame(maxWidth: 320)
            Spacer(minLength: 0)
            Toggle("Other stacks only", isOn: $store.excludeSameStack)
                .toggleStyle(.checkbox).font(.caption)
                .help("Restrict nearest-neighbor overlap and neighbor marking to other stacks. ARI always uses all visible shared objects.")
        }
    }
    var legend: some View {
        HStack(spacing: 10) {
            ForEach(0..<6) { i in
                HStack(spacing: 4) { Circle().fill(clusterColors[i]).frame(width: 7, height: 7); Text("\(i+1)") }
            }
            Text(store.sharedMethodColors ? "Default colors follow \(referenceName) groups. Each plot can override its feature and palette." : "Default colors show each method’s own groups. Each plot can override its feature and palette.")
                .foregroundStyle(.secondary)
            Spacer(minLength: 0)
            Text("\(store.selection.count) marked").foregroundStyle(.teal)
        }.font(.caption2)
    }
    func methodCard(_ method: Method) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            ScatterPanel(methodID: method.id, title: method.name,
                         colorMethodID: store.sharedMethodColors ? store.referenceMethod : nil, cohortIDs: plotCohortIDs, featureColoring: true)
                .frame(height: 350)
            if method.id == store.referenceMethod {
                Label("Reference method", systemImage: "scope").font(.caption2).foregroundStyle(.teal)
            }
            if method.metric == "shape_hybrid95" {
                Text(store.analysisSelected?.representations[method.id]?.vector.first == 1
                     ? "Selected cell: ambiguous axes · surface fallback active"
                     : "Ambiguous cell pairs automatically use the surface fallback")
                    .font(.caption2).foregroundStyle(.teal)
            }
            if let description = method.description {
                Text(description).font(.caption2).foregroundStyle(.secondary).lineLimit(3)
                    .help(description)
            }
            if benchmarkFamilies.contains(method.family ?? "") {
                let covered = store.dataset?.cells.filter { $0.representations[method.id] != nil }.count ?? 0
                Text(method.coverage == "all_cells" ? "\(covered) cells encoded · frozen inference · cell-level" : "\(covered) cells encoded · train / validation only · cell-level")
                    .font(.caption2).foregroundStyle(.teal)
                if let query = store.analysisSelected, query.representations[method.id] == nil {
                    Text("No embedding for \(query.id)").font(.caption2).foregroundStyle(.orange)
                }
            }
            if let stats = agreement.first(where: { $0.id == method.id }) {
                HStack(alignment: .firstTextBaseline, spacing: 16) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Cluster agreement").foregroundStyle(.secondary)
                        Text(calculating ? "…" : stats.ari.map { String(format: "%.3f ARI", $0) } ?? "Unavailable")
                            .font(.caption.monospacedDigit().weight(.semibold))
                    }.help("Adjusted Rand index vs the reference on \(stats.sharedCount) shared visible objects. 1 = identical partitions, 0 = chance-level agreement; values can be negative.")
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Query neighbors").foregroundStyle(.secondary)
                        Text(calculating ? "…" : stats.neighborIntersection.map { "\($0)/\(stats.neighborCount) shared" } ?? "Unavailable")
                            .font(.caption.monospacedDigit().weight(.semibold))
                    }.help("Overlap of the selected object’s nearest 10 objects in full feature space with the reference, using a shared candidate pool. Reduced to the available count when fewer than 10 are eligible.")
                    Spacer(minLength: 0)
                }.font(.caption2)
            }
            HStack {
                Button("Mark nearest 10") {
                    if let query = comparisonQuery {
                        let candidates = comparisonItems.filter { !store.excludeSameStack || $0.sample != query.sample }
                        store.selection = Set(rankNeighbors(query: query, candidates: candidates, method: method, count: 10).map(\.id))
                    }
                    if let id = store.selectedID { store.selection.insert(id) }
                }.disabled(comparisonQuery?.representations[method.id] == nil || store.markedOnly)
                    .help("Mark this method’s closest objects across the visible plots, including the selected query. Clear ‘Show marked only’ before changing the neighbor set.")
                Spacer()
                Text("\(comparisonItems.filter { $0.representations[method.id] != nil }.count) objects").foregroundStyle(.secondary)
            }.font(.caption2)
            if benchmarkFamilies.contains(method.family ?? "") {
                Button("Explore this space in 3D") {
                    store.method = method.id
                    store.activeTab = "explore"
                }.font(.caption2).disabled(comparisonQuery?.representations[method.id] == nil)
            }
        }
    }
    var explanation: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Learned shape benchmarks compares adapted Allen, pose-averaged Point-MAE and the invariant ShapeEmbedLite3D pilot with traditional features, spatial PCA and spherical harmonics. Dedicated method views include released weights, native-orientation Point-MAE and the Lite3D CNN reference. Raw and adjusted distances match the saved benchmark readouts; display PCA and groups are descriptive fits, not new held-out results.")
            Text(store.dataset?.methods.contains(where: { $0.coverage == "all_cells" }) == true
                 ? "These shape encoders now cover every cell by frozen inference for exploration. Original 609-cell benchmark codes, weights and adjustment fits are preserved. Excluded and QC-flagged cells have codes for review; membership and QC filters still control analyses. These are cell-level codes, not organoid encodings. The separate validation view keeps the 48-cell unquantized CAJAL reference."
                 : "These new embeddings cover 609 training / validation cells, with no test-cell or organoid codes. Same-object filtering keeps plots and neighbor comparisons on a common cohort; validation benchmarks reduces this to the 48-cell CAJAL reference cohort.")
            Text("Allen uses vector norms, Point-MAE pose averaging is external, and Lite3D distance matrices discard chirality. Point-MAE patch completion is not global reconstruction; Lite3D reconstructions still underfit the shapes.")
            Text("Coverage refers to the shape benchmark’s split. The sidebar split filter retains Atlas’s original experiment assignments, which can differ from those benchmark assignments.")
            Text("Use the adjusted comparison to ask what remains beyond volume, elongation and sphericity. Both shape and traditional features use the same training-only adjustment. The surface signature is a geometric descriptor; spatial PCA is the learned reconstruction embedding. Separation alone does not demonstrate biological meaning; that needs independent condition or phenotype labels.")
            Text("Validation benchmarks includes the unquantized CAJAL reference for 48 validation cells only. Shared-cell filtering applies to every plot, agreement and neighbor marking. Full-dataset CAJAL is a faster quantized approximation; its neighbors can differ substantially. Check the validation view before interpreting its groups.")
            Text("Click any point to select the same object in every space. Drag to mark a group, or mark a method’s nearest neighbors and see where they land in the other plots.")
            Text("ARI compares the exploratory partitions, independently of group numbering. Query-neighbor overlap uses each method’s full comparison distance, including ambiguity-aware fallback for spatial PCA. Both respond to your filters. Display projections and cluster fits use main-organoid cells only and are refitted after membership edits; ordinary view filters keep those fits fixed. Pools of 10 or fewer candidates force complete neighbor overlap.")
            Text("PCA axes and explained variance are specific to each method. The ambiguity-aware spatial comparison and direct CAJAL comparisons use distance maps (MDS) and average-linkage groups; positive inertia is not PCA variance. Organoid views average included QC cell descriptors, rather than encoding the entire organoid surface. Rotation, reflection and projection can change the apparent layout. Agreement measures describe consistency between methods, not biological validity.")
        }.font(.caption).foregroundStyle(.secondary).padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
    }
}
