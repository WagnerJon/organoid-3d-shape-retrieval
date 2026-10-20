import Foundation

@main struct RefinementWorkerCheck {
    @MainActor static func main() async throws {
        let root = URL(fileURLWithPath:FileManager.default.currentDirectoryPath)
        let source = try JSONDecoder().decode(SegSource.self,from:Data(contentsOf:root.appendingPathComponent(".cache/segmentation-check/source/source.json")))
        let parent = try JSONDecoder().decode(SegResult.self,from:Data(contentsOf:root.appendingPathComponent("results/segmentation_explorer/runs/cellpose-original-X001/result.json")))
        let post = SegmentationStore(projectRoot:root)
        post.comparisonMethod = "refined"
        post.adoptPreparedSource(source,raw:Array(try Data(contentsOf:URL(fileURLWithPath:source.folder).appendingPathComponent("raw.u8"))))
        post.choose(parent,side:1)
        post.refinementSettings.iterations = 2;post.refinementSettings.device = "cpu"
        post.refineDisplayed(side:1)
        let deadline = Date().addingTimeInterval(120)
        while post.busy && Date() < deadline {try await Task.sleep(nanoseconds:100_000_000)}
        guard !post.busy else {post.cancel();throw DataError.invalid("Native refinement timed out")}
        guard post.error == nil, post.resultA?.id == parent.id, let child = post.resultB,
              child.refinement?.parentID == parent.id, post.agreement != nil else {
            throw DataError.invalid("Native refinement failed: \(post.error ?? post.status)")
        }
        print("PASS: native app worker launch, fresh refinement folder, completed saved run, exact before/after pair and agreement.")
        print("Saved validation run: \(child.folder)")
    }
}
