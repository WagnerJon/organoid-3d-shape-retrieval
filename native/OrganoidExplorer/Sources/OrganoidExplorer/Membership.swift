import SwiftUI

/// Review is separate from the analysis cohort, so inspecting a satellite never includes it.
struct MembershipView: View {
    @EnvironmentObject var store: ExplorerStore
    @State private var stack = "All stacks"
    @State private var excludedOnly = true
    var cells: [Specimen] {
        (store.dataset?.cells ?? []).filter {
            (stack == "All stacks" || $0.sample == stack) && (!excludedOnly || !$0.includedInAnalysis)
        }.sorted { $0.id.localizedStandardCompare($1.id) == .orderedAscending }
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Main organoid membership").font(.title.bold())
            Text("Detached cells are marked red and excluded from analyses. Original masks and surfaces are preserved.")
                .foregroundStyle(.secondary)
            HStack {
                Picker("Stack", selection: $stack) {
                    Text("All stacks").tag("All stacks")
                    ForEach(store.dataset?.samples ?? []) { Text($0.id).tag($0.id) }
                }.frame(width: 230)
                Toggle("Show excluded cells only", isOn: $excludedOnly)
                Spacer()
                Text("\(store.excludedCells.count) excluded · \((store.dataset?.cells.count ?? 0) - store.excludedCells.count) included").font(.headline)
            }
            HSplitView {
                VStack(alignment: .leading, spacing: 10) {
                    Table(cells, selection: Binding(get: { store.selectedID }, set: { if let id = $0 { store.select(id) } })) {
                        TableColumn("Cell") { Text($0.id).monospaced() }
                        TableColumn("Analysis") { cell in
                            Label(cell.includedInAnalysis ? "Included" : "Excluded", systemImage: cell.includedInAnalysis ? "checkmark.circle" : "minus.circle")
                                .foregroundStyle(cell.includedInAnalysis ? Color.teal : Color.red)
                        }
                        TableColumn("Decision") { cell in Text(cell.membershipOverride == nil ? "Automatic" : "Reviewed") }
                    }
                    Text("Automatic rule: largest connected tissue component in each native label stack. A cell is included when more than half its labelled voxels belong to that component. Review separated tissue pieces and override the decision beside the reconstruction.")
                        .font(.caption).foregroundStyle(.secondary)
                    Text("Membership edits are saved with this experiment and refit PCA, clustering, feature scaling and organoid summaries. Learned models are not retrained.")
                        .font(.caption).foregroundStyle(.secondary)
                }.frame(minWidth: 420).padding(.trailing, 10)
                Inspector().frame(minWidth: 400).padding(.leading, 10)
            }
        }.padding(20)
        .onAppear {
            if store.level != "Cells" { store.level = "Cells"; store.changeLevel() }
            if let first = cells.first { store.select(first.id) }
        }
    }
}
