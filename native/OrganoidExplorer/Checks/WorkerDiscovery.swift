import Foundation
import AppKit

@main struct WorkerDiscoveryChecks {
    @MainActor static func main() throws {
        func require(_ value: @autoclosure () -> Bool, _ message: String) { if !value() { fatalError(message) } }
        let root = URL(fileURLWithPath: CommandLine.arguments[1]).deletingLastPathComponent().deletingLastPathComponent()
        let temp = FileManager.default.temporaryDirectory.appendingPathComponent("atlas-bonjour-check-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
        for name in ["scripts", "vendor", ".venv"] { try FileManager.default.createSymbolicLink(at: temp.appendingPathComponent(name), withDestinationURL: root.appendingPathComponent(name)) }
        setenv("ATLAS_WORKER_CREDENTIALS", temp.appendingPathComponent("credentials.json").path, 1)
        setenv("ATLAS_WORKER_INSTANCE_DIR", temp.appendingPathComponent("singleton").path, 1)
        let publisher = AtlasWorkers(root: temp), browser = AtlasWorkers(root: temp)
        defer { browser.shutdown(); publisher.shutdown(); try? FileManager.default.removeItem(at: temp) }
        func spin(_ seconds: Double, until test: () -> Bool) {
            let end = Date().addingTimeInterval(seconds)
            while !test() && Date() < end { RunLoop.current.run(until: Date().addingTimeInterval(0.05)) }
        }
        let legacy = try JSONDecoder().decode(AtlasExecution.self, from: Data(#"{"useLocal":true,"workers":[]}"#.utf8))
        require(!legacy.splitStacks && legacy.planesPerBatch == 16, "Old preparation requests keep whole-stack execution")
        browser.splitStacks = true; browser.planesPerBatch = 32
        require(browser.execution?.splitStacks == true && browser.execution?.planesPerBatch == 32, "Local-only batch settings are saved")
        let roundTrip = try JSONDecoder().decode(AtlasExecution.self, from: JSONEncoder().encode(browser.execution!))
        browser.splitStacks = false; browser.restore(roundTrip)
        require(browser.splitStacks && browser.planesPerBatch == 32, "Batch settings restore on resume")
        publisher.startSharing()
        spin(20) { publisher.sharing }
        require(publisher.sharing, "Native worker starts and advertises: \(publisher.log)")
        browser.discover()
        spin(15) { !browser.nearby.isEmpty }
        require(!browser.nearby.isEmpty, "Bonjour finds the local test worker: \(browser.message)")
        let endpoint = browser.nearby.first { $0.fingerprint == publisher.fingerprint }
        require(endpoint != nil, "Discovery reports the server certificate fingerprint")
        browser.beginPair(endpoint!); browser.pairingCode = publisher.code; browser.pair()
        spin(15) { !browser.busy }
        require(browser.enabled.contains(endpoint!.id) && browser.paired.contains(endpoint!), "Native pairing persists a trusted worker: \(browser.message)")
        browser.remove(endpoint!)
        spin(15) { !browser.busy }
        require(browser.paired.isEmpty && browser.enabled.isEmpty, "Removing a worker forgets its credentials")
        browser.stopDiscovery(); publisher.stopSharing()
        spin(10) { !publisher.sharing && !publisher.starting }
        require(!publisher.sharing, "Worker stops cleanly")
        print("PASS: native worker launch; Bonjour publication/discovery; TLS fingerprint; pairing; credential removal; shutdown.")
    }
}
