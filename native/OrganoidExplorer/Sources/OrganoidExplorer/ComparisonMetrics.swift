import Foundation

/// Adjusted Rand index: agreement of partitions, invariant to cluster numbering.
/// Returns nil for fewer than two shared objects. Degenerate identical partitions score 1.
func adjustedRandIndex(_ left: [Int], _ right: [Int]) -> Double? {
    guard left.count == right.count, left.count > 1 else { return nil }
    var rows: [Int: Int] = [:], columns: [Int: Int] = [:]
    var table: [Int: [Int: Int]] = [:]
    for (a,b) in zip(left,right) {
        rows[a, default: 0] += 1; columns[b, default: 0] += 1
        table[a, default: [:]][b, default: 0] += 1
    }
    func pairs(_ n: Int) -> Double { Double(n) * Double(n-1) / 2 }
    let observed = table.values.flatMap { $0.values }.reduce(0.0) { $0 + pairs($1) }
    let rowPairs = rows.values.reduce(0.0) { $0 + pairs($1) }
    let columnPairs = columns.values.reduce(0.0) { $0 + pairs($1) }
    let expected = rowPairs * columnPairs / pairs(left.count)
    let denominator = (rowPairs + columnPairs) / 2 - expected
    if abs(denominator) < 1e-12 { return 1 }
    return (observed - expected) / denominator
}
struct MethodAgreement: Identifiable {
    let id: String
    let sharedCount: Int
    let ari: Double?
    let neighborIntersection: Int?
    let neighborCount: Int
}
func rankNeighbors(query: Specimen, candidates: [Specimen], method: Method, count: Int) -> [Neighbor] {
    guard query.includedInAnalysis, let queryVector = query.representations[method.id]?.vector else { return [] }
    return candidates.compactMap { item -> Neighbor? in
        guard item.includedInAnalysis, item.id != query.id, let vector = item.representations[method.id]?.vector else { return nil }
        let distance = representationDistance(queryVector, vector, metric: method.metric)
        return distance.isFinite ? Neighbor(item: item, distance: distance) : nil
    }.sorted { $0.distance == $1.distance ? $0.id < $1.id : $0.distance < $1.distance }
        .prefix(count).map { $0 }
}
func compareMethods(methods: [Method], reference: String, items: [Specimen], query: Specimen?, otherStacksOnly: Bool) -> [MethodAgreement] {
    guard let baseline = methods.first(where: { $0.id == reference }) else { return [] }
    return methods.map { method in
        let shared = items.filter { $0.includedInAnalysis && $0.representations[reference] != nil && $0.representations[method.id] != nil }
        let ari = adjustedRandIndex(shared.map { $0.representations[reference]!.cluster }, shared.map { $0.representations[method.id]!.cluster })
        var overlap: Int?, k = 0
        if let query, query.includedInAnalysis, query.representations[reference] != nil, query.representations[method.id] != nil {
            let eligible = shared.filter { $0.id != query.id && (!otherStacksOnly || $0.sample != query.sample) }
            let a = rankNeighbors(query: query, candidates: eligible, method: baseline, count: 10)
            let b = rankNeighbors(query: query, candidates: eligible, method: method, count: 10)
            k = min(a.count, b.count)
            if k > 0 {
                overlap = Set(a.prefix(k).map(\.id)).intersection(Set(b.prefix(k).map(\.id))).count
            }
        }
        return MethodAgreement(id: method.id, sharedCount: shared.count, ari: ari, neighborIntersection: overlap, neighborCount: k)
    }
}
