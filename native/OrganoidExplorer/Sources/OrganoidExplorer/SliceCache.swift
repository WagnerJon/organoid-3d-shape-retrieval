import Foundation
import CoreGraphics

/// Every setting that changes pixels is part of the cache key; slice position is separate.
struct VolumeSliceStyle: Hashable, Sendable {
    let axis: Int
    let selected: Int
    let overlay: Bool
    let contrast: Double
    let excluded: Set<Int>
    let hideExcluded: Bool

    init(axis: Int, selected: Int, overlay: Bool, contrast: Double, excluded: Set<Int> = [], hideExcluded: Bool = false) {
        self.axis = axis
        self.selected = overlay ? selected : 0
        self.overlay = overlay
        self.contrast = contrast
        self.excluded = overlay ? excluded : []
        self.hideExcluded = overlay && hideExcluded
    }
}

struct SlicePreparation: Hashable {
    let volumeID: UUID
    let style: VolumeSliceStyle
}

private final class SliceImageKey: NSObject {
    let style: VolumeSliceStyle
    let index: Int
    init(style: VolumeSliceStyle, index: Int) { self.style = style; self.index = index }
    override var hash: Int {
        var hasher = Hasher(); hasher.combine(style); hasher.combine(index)
        return hasher.finalize()
    }
    override func isEqual(_ object: Any?) -> Bool {
        guard let other = object as? SliceImageKey else { return false }
        return index == other.index && style == other.style
    }
}

/// NSCache permits foreground lookups while a cancellable worker warms adjacent slices.
/// Images and volume arrays are immutable once published. Eviction bounds memory use.
final class VolumeSliceCache: @unchecked Sendable {
    static let byteLimit = 32 * 1024 * 1024
    private let images: NSCache<SliceImageKey, CGImage> = {
        let result = NSCache<SliceImageKey, CGImage>()
        result.totalCostLimit = byteLimit
        result.countLimit = 768
        return result
    }()
    func image(style: VolumeSliceStyle, index: Int) -> CGImage? {
        images.object(forKey: SliceImageKey(style: style, index: index))
    }
    func insert(_ image: CGImage, style: VolumeSliceStyle, index: Int) {
        images.setObject(image, forKey: SliceImageKey(style: style, index: index), cost: image.bytesPerRow * image.height)
    }
}

final class CachedVolume: NSObject {
    let volume: VolumeData
    init(_ volume: VolumeData) { self.volume = volume }
    var cost: Int { volume.raw.count + volume.labels.count * 2 + VolumeSliceCache.byteLimit }
}
