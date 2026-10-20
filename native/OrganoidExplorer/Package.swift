// swift-tools-version: 6.0
import PackageDescription
let package = Package(
    name: "OrganoidExplorer", platforms: [.macOS(.v14)],
    products: [.executable(name: "OrganoidExplorer", targets: ["OrganoidExplorer"])],
    targets: [.executableTarget(name: "OrganoidExplorer"),
              .testTarget(name: "OrganoidExplorerTests", dependencies: ["OrganoidExplorer"])],
    swiftLanguageModes: [.v5])
