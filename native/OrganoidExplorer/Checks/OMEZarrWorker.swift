import SwiftUI
import AppKit

@main struct OMEZarrChecks {
    @MainActor static func main() async throws {
        func require(_ condition: @autoclosure () -> Bool, _ message: String) { if !condition() { fatalError(message) } }
        let root = URL(fileURLWithPath:CommandLine.arguments[1]).deletingLastPathComponent().deletingLastPathComponent()
        let source = ProcessInfo.processInfo.environment["ATLAS_ZARR_CHECK_SOURCE"] ?? "/Volumes/CHUNKY/Scientific_data/001-small"
        let temp = FileManager.default.temporaryDirectory.appendingPathComponent("atlas-zarr-native-" + UUID().uuidString)
        try FileManager.default.createDirectory(at:temp,withIntermediateDirectories:true)
        defer { try? FileManager.default.removeItem(at:temp) }
        let request = temp.appendingPathComponent("inspect.json"), output = temp.appendingPathComponent("catalog.json")
        try JSONSerialization.data(withJSONObject:["paths":[source]]).write(to:request)
        let process = Process();process.executableURL = root.appendingPathComponent(".venv/bin/python");process.currentDirectoryURL = root
        process.arguments = ["scripts/prepare_atlas_dataset.py","inspect-zarr","--request",request.path,"--output",output.path]
        try process.run();process.waitUntilExit();require(process.terminationStatus == 0,"Native OME-Zarr inspection completes")
        let catalog = try JSONDecoder().decode(OMEZarrCatalog.self,from:Data(contentsOf:output))
        let image = catalog.images.first { $0.name == "deconv.ome.zarr" }!
        require(image.channels == ["mem9","H2B"] && image.timepoints == 50 && image.levels.count == 5,"Namespaced OME 0.5 metadata decodes into native controls")
        require(image.labels.contains { $0.name == "cell" && $0.instanceEligible } && image.labels.contains { $0.name == "lumen" && !$0.instanceEligible },"Instance choices omit semantic lumen labels")
        let store = DatasetPreparationStore(root:root)
        store.addOMEZarr(image,channel:0,level:2,first:0,last:2,step:2,label:"cell")
        require(store.inputs.count == 2 && Set(store.inputs.map(\.id)).count == 2,"Two time points from the same store remain independently selectable")
        store.addOMEZarr(image,channel:0,level:2,first:0,last:2,step:2,label:"cell")
        require(store.inputs.count == 2,"Reimporting an identical selection avoids duplicate samples")
        require(store.inputs[1].omeZarr?.time == 2 && store.inputs[0].shape == [76,96,152],"The requested frame range and level are retained")
        store.previewSlice(input:store.inputs[0],plane:"xy",index:30,segment:false)
        let started = Date()
        while store.busy && Date().timeIntervalSince(started)<30 { try await Task.sleep(nanoseconds:100_000_000) }
        require(store.sliceRaw != nil && store.error == nil,"OME-Zarr raw slice reaches the native preview: \(store.error ?? "")")
        store.output = temp.appendingPathComponent("prepared");store.harmonics = false;store.confirmed = true
        require(store.valid && !store.memoryBlocked,"Calibrated existing labels can prepare without Cellpose inference")
        store.start()
        let began = Date()
        while store.busy && Date().timeIntervalSince(began)<90 { try await Task.sleep(nanoseconds:100_000_000) }
        require(store.result != nil && store.error == nil,"Native OME-Zarr preparation completes: \(store.error ?? store.log)")
        let dataset = try JSONDecoder().decode(Dataset.self,from:Data(contentsOf:store.result!.appendingPathComponent("dataset.json")))
        try dataset.validate()
        require(dataset.samples.count == 2 && dataset.samples.allSatisfy { $0.name?.contains("mem9") == true },"The actual Atlas loader accepts the imported dataset and frame names")
        _ = NSApplication.shared
        let view = NSHostingView(rootView:OMEZarrImportView(catalog:catalog,add:{ _,_,_,_,_,_,_ in },close:{}).preferredColorScheme(.dark))
        let window = NSWindow(contentRect:NSRect(x:0,y:0,width:650,height:760),styleMask:[.titled],backing:.buffered,defer:false)
        window.contentView = view;window.orderFront(nil)
        try await Task.sleep(nanoseconds:500_000_000)
        view.layoutSubtreeIfNeeded()
        if let bitmap = view.bitmapImageRepForCachingDisplay(in:view.bounds) {
            view.cacheDisplay(in:view.bounds,to:bitmap)
            try bitmap.representation(using:.png,properties:[:])!.write(to:root.appendingPathComponent("results/ome-zarr-import-preview.png"))
        }
        window.orderOut(nil)
        print("PASS: real 5D metadata; native T/C/level/label controls; unique frame IDs; preview; full preparation; native Atlas loading; import-window render.")
    }
}
