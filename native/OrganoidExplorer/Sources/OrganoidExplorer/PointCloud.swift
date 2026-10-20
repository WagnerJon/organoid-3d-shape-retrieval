import Foundation
import SceneKit

struct PointCloudData {
    let vertices: [SCNVector3]
    init(url: URL) throws {
        let data = try Data(contentsOf:url)
        guard data.count >= 4 else { throw DataError.invalid("Truncated point reconstruction") }
        let count = Int(data.prefix(4).withUnsafeBytes { $0.loadUnaligned(as:UInt32.self).littleEndian })
        guard count > 0, count <= 1_000_000, data.count == 4+count*12 else { throw DataError.invalid("Invalid point reconstruction length") }
        let values = data.subdata(in:4..<data.count).withUnsafeBytes { Array($0.bindMemory(to:Float.self)) }
        guard values.allSatisfy(\.isFinite) else { throw DataError.invalid("Non-finite reconstructed points") }
        vertices = (0..<count).map { SCNVector3(values[$0*3],values[$0*3+1],values[$0*3+2]) }
    }
    func geometry() -> SCNGeometry {
        let indices = (0..<vertices.count).map(UInt32.init)
        let bytes = indices.withUnsafeBufferPointer { Data(buffer:$0) }
        let element = SCNGeometryElement(data:bytes,primitiveType:.point,primitiveCount:vertices.count,bytesPerIndex:4)
        element.pointSize = 3
        element.minimumPointScreenSpaceRadius = 1
        element.maximumPointScreenSpaceRadius = 3
        return SCNGeometry(sources:[SCNGeometrySource(vertices:vertices)],elements:[element])
    }
}

enum ReconstructionGeometry {
    case mesh(MeshData)
    case points(PointCloudData)
    func geometry() -> SCNGeometry {
        switch self {
        case .mesh(let mesh): return mesh.geometry()
        case .points(let points): return points.geometry()
        }
    }
}
