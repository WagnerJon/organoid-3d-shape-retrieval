import Foundation

@main struct SliceCacheWorker {
    @MainActor static func waitForVolume(_ store: ExplorerStore, sample: String) async throws -> VolumeData {
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline {
            if let message = store.volumeError { throw DataError.invalid(message) }
            if let volume = store.volume, volume.sample.id == sample { return volume }
            try await Task.sleep(nanoseconds:5_000_000)
        }
        throw DataError.invalid("Timed out loading " + sample)
    }
    @MainActor static func main() async throws {
        let folder = URL(fileURLWithPath:CommandLine.arguments[1])
        let dataset = try JSONDecoder().decode(Dataset.self,from:Data(contentsOf:folder.appendingPathComponent("dataset.json")))
        let store = ExplorerStore(); store.dataset = dataset; store.folder = folder
        let a = dataset.cells.first { $0.sample == dataset.samples[0].id }!
        let b = dataset.cells.first { $0.sample == dataset.samples[1].id }!
        let c = dataset.cells.first { $0.sample == dataset.samples[2].id }!
        store.select(a.id); store.loadVolume() // repeated selections during loading share the pending request
        let original = try await waitForVolume(store,sample:a.sample)
        let style = VolumeSliceStyle(axis:0,selected:a.label,overlay:true,contrast:1)
        original.prepareSlices(style:style,near:0.5)
        let firstImage = original.image(style:style,fraction:0.5)!
        store.select(b.id)
        _ = try await waitForVolume(store,sample:b.sample)
        store.select(a.id)
        guard store.volume?.cacheID == original.cacheID,
              firstImage === store.volume?.image(style:style,fraction:0.5) else {
            throw DataError.invalid("Returning to a stack did not reuse its volume and prepared images")
        }
        store.select(c.id); store.select(b.id)
        try await Task.sleep(nanoseconds:100_000_000)
        guard store.volume?.sample.id == b.sample else { throw DataError.invalid("A stale load replaced the current stack") }
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let sampleFolder = temporary.appendingPathComponent(a.sample)
        try FileManager.default.createDirectory(at:sampleFolder,withIntermediateDirectories:true)
        defer { try? FileManager.default.removeItem(at:temporary) }
        try FileManager.default.copyItem(at:folder.appendingPathComponent(a.sample + "/labels.u16"),to:sampleFolder.appendingPathComponent("labels.u16"))
        try Data(repeating:200,count:original.raw.count).write(to:sampleFolder.appendingPathComponent("raw.u8"))
        store.folder = temporary; store.select(a.id)
        let relocated = try await waitForVolume(store,sample:a.sample)
        guard relocated.cacheID != original.cacheID, relocated.raw.first == 200 else {
            throw DataError.invalid("Equal sample IDs from different experiments shared cached pixels")
        }
        print("PASS: stack revisits reuse volumes and frames; repeated selections share pending loads; stale loads cannot replace the selection; experiment folders cannot share cached pixels.")
    }
}
