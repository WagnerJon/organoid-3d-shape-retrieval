import XCTest
@testable import OrganoidExplorer

final class ExplorerTests: XCTestCase {
    func testAdjustedRandIndex() {
        XCTAssertEqual(adjustedRandIndex([0,0,1,1], [5,5,8,8]), 1)
        XCTAssertEqual(adjustedRandIndex([0,0,1,1], [0,1,0,1])!, -0.5, accuracy: 1e-10)
        XCTAssertEqual(adjustedRandIndex([0,0,0], [1,1,1]), 1)
        XCTAssertEqual(adjustedRandIndex([0,1,2], [3,4,5]), 1)
        XCTAssertNil(adjustedRandIndex([0], [0]))
        XCTAssertNil(adjustedRandIndex([0,1], [0]))
    }
    func testFullVectorDistances() {
        XCTAssertEqual(representationDistance([1,0], [0,1], metric: "cosine"), 1, accuracy: 1e-10)
        XCTAssertEqual(representationDistance([1,0], [5,0], metric: "cosine"), 0, accuracy: 1e-10)
        XCTAssertEqual(representationDistance([1,0], [-1,0], metric: "cosine"), 2, accuracy: 1e-10)
        XCTAssertEqual(representationDistance([0,0], [3,4], metric: "euclidean"), 5, accuracy: 1e-10)
        XCTAssertTrue(representationDistance([0,0], [3,4], metric: "cosine").isInfinite)
        XCTAssertTrue(representationDistance([1], [1,2], metric: "euclidean").isInfinite)
    }
    func testOrthogonalSlicesAndPickingUseSameVoxel() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let labels: [UInt16] = Array(0..<24)
        try labels.withUnsafeBufferPointer { try Data(buffer: $0).write(to: folder.appendingPathComponent("labels.u16")) }
        try Data(Array(UInt8(0)..<UInt8(24))).write(to: folder.appendingPathComponent("raw.u8"))
        let sample = Sample(id: "test", shape: [2,3,4], spacing: [1,1,1], extent: [1,2,3], rawAvailable: true, sourceImage: "", sourceLabels: "", meshCount: 1)
        let volume = try VolumeData(folder: folder, sample: sample)
        // The same physical voxel (z=1,y=2,x=3) in all three planes.
        XCTAssertEqual(volume.index(x: 3, y: 2, slice: 1, axis: 0), 23)
        XCTAssertEqual(volume.index(x: 3, y: 1, slice: 2, axis: 1), 23)
        XCTAssertEqual(volume.index(x: 2, y: 1, slice: 3, axis: 2), 23)
        XCTAssertEqual(volume.image(axis: 0, fraction: 1, selected: 23, overlay: true, contrast: 1)?.width, 4)
        XCTAssertEqual(volume.image(axis: 1, fraction: 1, selected: 23, overlay: true, contrast: 1)?.height, 2)
        XCTAssertEqual(volume.image(axis: 2, fraction: 1, selected: 23, overlay: true, contrast: 1)?.width, 3)
        try Data([0]).write(to: folder.appendingPathComponent("labels.u16"))
        XCTAssertThrowsError(try VolumeData(folder: folder, sample: sample))
    }
    func testRejectsTruncatedMesh() throws {
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: path) }
        try Data([1,2,3]).write(to: path)
        XCTAssertThrowsError(try MeshData(url: path))
    }
    func testPreparedDatasetIntegrity() throws {
        // Optional integration test against the real, ignored experiment export.
        guard let path = ProcessInfo.processInfo.environment["EXPLORER_DATASET"] else { throw XCTSkip("Set EXPLORER_DATASET to validate the local export") }
        let folder = URL(fileURLWithPath: path)
        let dataset = try JSONDecoder().decode(Dataset.self, from: Data(contentsOf: folder.appendingPathComponent("dataset.json")))
        try dataset.validate()
        XCTAssertEqual(dataset.cells.count, 1088)
        XCTAssertEqual(dataset.samples.count, 108)
        for sample in dataset.samples {
            let volume = try VolumeData(folder: folder.appendingPathComponent(sample.id), sample: sample)
            let labels = Set(dataset.cells.filter { $0.sample == sample.id }.map(\.label))
            XCTAssertTrue(Set(volume.labels.filter { $0 > 0 }.map(Int.init)).isSubset(of: labels))
        }
        for item in dataset.cells {
            let mesh = try MeshData(url: folder.appendingPathComponent(item.sample).appendingPathComponent("\(item.label).mesh"))
            XCTAssertFalse(mesh.vertices.isEmpty)
            XCTAssertEqual(mesh.faces.count % 3, 0)
        }
    }
}
