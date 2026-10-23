import Foundation
import AppKit
import SceneKit
import SwiftUI

@main struct ReconstructionChecks {
    @MainActor static func main() async throws {
        func require(_ value: @autoclosure () -> Bool, _ message: String) { if !value() { fatalError(message) } }
        func close(_ a: NSColor, _ b: NSColor) -> Bool {
            let a=a.usingColorSpace(.sRGB)!, b=b.usingColorSpace(.sRGB)!
            return abs(a.redComponent-b.redComponent)+abs(a.greenComponent-b.greenComponent)+abs(a.blueComponent-b.blueComponent) < 0.001
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("atlas-reconstruction-"+UUID().uuidString)
        let folder = root.appendingPathComponent("sample")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        // Original physical meshes, with one object on either side of the origin.
        for label in 1...3 {
            var data=Data();let counts:[UInt32]=[4,4]
            let x=Float(label*4)
            let vertices:[Float]=[x,4,4, x+3,4,4, x,7,4, x,4,7]
            let normals:[Float]=[0,0,1,0,0,1,0,0,1,0,0,1]
            let faces:[UInt32]=[0,2,1,0,1,3,1,2,3,2,0,3]
            counts.withUnsafeBytes { data.append(contentsOf:$0) };vertices.withUnsafeBytes { data.append(contentsOf:$0) }
            normals.withUnsafeBytes { data.append(contentsOf:$0) };faces.withUnsafeBytes { data.append(contentsOf:$0) }
            try data.write(to:folder.appendingPathComponent("\(label).mesh"))
        }
        func item(_ label:Int, value:Double?, included:Bool=true) -> Specimen {
            Specimen(id:"sample:L\(label)",sample:"sample",split:"test",flags:"",label:label,qc:true,
                     metrics:value.map { ["sphericity":$0] } ?? [:],representations:[:],cellCount:nil,inMainOrganoid:included)
        }
        let cells=[item(1,value:0.2),item(2,value:0.8),item(3,value:nil,included:false)]
        let random=ReconstructionColorMap(cells:cells,settings:PlotColorSettings(),hideExcluded:false)
        let reordered=ReconstructionColorMap(cells:cells.reversed(),settings:PlotColorSettings(),hideExcluded:false)
        require(close(random.colors[1]!,reordered.colors[1]!) && !close(random.colors[1]!,random.colors[2]!),"Stable ID colors are independent of order and distinguish objects")
        require(close(random.colors[3]!,.systemRed),"Excluded cells retain their ID-mode warning color")
        let settings=PlotColorSettings(feature:"sphericity",palette:.viridis)
        let map=ReconstructionColorMap(cells:cells,settings:settings,hideExcluded:false)
        require(map.map.minimum==0.2 && map.map.maximum==0.8 && map.map.missing==1,"Feature range and missing values come from the reconstruction cohort")
        require(close(map.colors[1]!,PlotPalette.viridis.color(0)) && close(map.colors[2]!,PlotPalette.viridis.color(1)) && close(map.colors[3]!,.gray),"Features map to palette endpoints; missing values are gray")
        let sample=Sample(id:"sample",shape:[16,16,20],spacing:[1,1,1],extent:[16,16,20],rawAvailable:false,sourceImage:"",sourceLabels:"",meshCount:3)
        let capture=PlotSceneCapture()
        var config=ReconstructionView(folder:root,sample:sample,cells:cells,selectedLabel:1,coloring:settings,capture:capture)
        let view=PickingSceneView(frame:NSRect(x:0,y:0,width:900,height:650));view.autoenablesDefaultLighting=true
        let coordinator=ReconstructionView.Coordinator();coordinator.view=view;coordinator.key=root.path+sample.id;coordinator.current=config
        coordinator.load(view:view,config:config)
        let start=Date()
        while (coordinator.nodes.count<3 || capture.loading) && Date().timeIntervalSince(start)<10 { try await Task.sleep(nanoseconds:10_000_000) }
        require(coordinator.nodes.count==3,"Mesh load completes")
        let originalGeometry=coordinator.nodes[1]!.geometry!
        view.pointOfView?.position=SCNVector3(24,-28,24);view.pointOfView?.look(at:SCNVector3Zero)
        let originalCamera=view.pointOfView!.worldTransform
        config.coloring=PlotColorSettings(feature:"sphericity",palette:.magma);config.highlightSelection=false
        coordinator.current=config;coordinator.updateMaterials(config:config)
        require(coordinator.nodes[1]!.geometry === originalGeometry,"Color changes do not rebuild the geometry")
        require(SCNMatrix4EqualToMatrix4(originalCamera,view.pointOfView!.worldTransform),"Coloring preserves the camera")
        require(close(coordinator.nodes[1]!.geometry!.firstMaterial!.diffuse.contents as! NSColor,PlotPalette.magma.color(0)),"On-screen material uses the selected feature palette")
        let seed=capture.seed()!
        require(SCNMatrix4EqualToMatrix4(seed.camera.worldTransform,originalCamera),"Fullscreen starts at the current camera")
        try FileManager.default.removeItem(at:folder) // Fullscreen must not reload mesh files.
        let secondCapture=PlotSceneCapture();var expanded=config;expanded.capture=secondCapture;expanded.seed=seed
        let second=PickingSceneView(frame:NSRect(x:0,y:0,width:1000,height:700));second.autoenablesDefaultLighting=true
        let secondCoordinator=ReconstructionView.Coordinator();secondCoordinator.view=second;secondCoordinator.key=coordinator.key;secondCoordinator.current=expanded
        secondCoordinator.load(view:second,config:expanded)
        require(secondCoordinator.nodes.count==3,"Fullscreen reuses loaded surfaces without disk reads")
        require(secondCoordinator.nodes[1]!.geometry !== originalGeometry,"Fullscreen has independent material state")
        expanded.hideExcluded=true;expanded.clipping=true;expanded.axis=0;expanded.slice=0.5
        secondCoordinator.current=expanded;secondCoordinator.updateMaterials(config:expanded)
        require(secondCoordinator.nodes[3]!.isHidden && !coordinator.nodes[3]!.isHidden,"Exclusion visibility is correctly applied per scene")
        require(secondCoordinator.appliedClip?.position==8,"Slicing retains physical coordinates")
        if ProcessInfo.processInfo.environment["ATLAS_CHECK_RENDER"]=="1" {
            // Offscreen SceneKit needs access to the Mac GPU.
            var options=PlotRenderOptions();options.xLabel="X (µm)";options.yLabel="Y (µm)";options.zLabel="Z (µm)";options.showAxes=false;options.showGrid=false;options.showTitle=false;options.showLegend=false
            options.background = .transparent;options.highlightSelection=false
            let image=try secondCapture.snapshot(options:options,size:CGSize(width:640,height:480))
            let bitmap=NSBitmapImageRep(data:try PlotImageExport.pngData(image))!
            require(bitmap.pixelsWide==640 && bitmap.pixelsHigh==480,"Exports retain exact requested pixel dimensions")
            require((bitmap.colorAt(x:0,y:0)?.alphaComponent ?? 1)<0.01,"Transparent exports preserve alpha")
            var visible=0
            for y in stride(from:0,to:480,by:4) { for x in stride(from:0,to:640,by:4) {
                if (bitmap.colorAt(x:x,y:y)?.alphaComponent ?? 0)>0.2 { visible += 1 }
            } }
            require(visible>10,"PNG contains reconstructed surfaces")
            let output=URL(fileURLWithPath:CommandLine.arguments[1]).deletingLastPathComponent().appendingPathComponent("reconstruction-fullscreen-check")
            try FileManager.default.createDirectory(at:output,withIntermediateDirectories:true)
            try PlotImageExport.writePNG(image,to:output.appendingPathComponent("transparent.png"))
            options.showAxes=true;options.showLegend=true;options.showTitle=true;options.title="Cell morphology";options.background = .light
            let styled=try secondCapture.snapshot(options:options,size:CGSize(width:900,height:650))
            try PlotImageExport.writePNG(styled,to:output.appendingPathComponent("feature-render.png"))
            require(secondCoordinator.nodes[3]!.isHidden && close(secondCoordinator.nodes[1]!.geometry!.firstMaterial!.diffuse.contents as! NSColor,PlotPalette.magma.color(0)),"Rendering leaves live colors and visibility unchanged")
            let scatter=SCNView(frame:NSRect(x:0,y:0,width:640,height:480));scatter.scene=SCNScene()
            let sphere=SCNSphere(radius:2);sphere.firstMaterial?.lightingModel = .constant
            sphere.firstMaterial?.diffuse.contents=NSColor.systemOrange
            let point=SCNNode(geometry:sphere);point.name="cell:fixture";scatter.scene!.rootNode.addChildNode(point)
            let camera=SCNNode();camera.camera=SCNCamera();camera.camera?.usesOrthographicProjection=true
            camera.camera?.orthographicScale=5;camera.position=SCNVector3(0,0,12)
            scatter.scene!.rootNode.addChildNode(camera);scatter.pointOfView=camera
            let scatterCapture=PlotSceneCapture()
            scatterCapture.update(view:scatter,colors:["fixture":.systemBlue],baseScales:["fixture":1],map:PlotColorMap(settings:PlotColorSettings(),items:[]),groups:[0])
            options.showLegend=false;options.showTitle=false;options.showAxes=false
            let plot=try scatterCapture.snapshot(options:options,size:CGSize(width:640,height:480))
            let plotBitmap=NSBitmapImageRep(data:try PlotImageExport.pngData(plot))
            require(plotBitmap?.pixelsWide==640,"Shared 3D plot export still accepts cell-prefixed scatter scenes")
            require(close(sphere.firstMaterial!.diffuse.contents as! NSColor,.systemOrange),"Plot export also leaves live selection styling unchanged")
            print("PASS: camera-based PNG exports; physical axes; legend; transparent alpha; unchanged live materials.")
        }
        print("PASS: stable ID/feature colors; missing values; material-only updates; camera-preserving fullscreen without mesh rereads; exclusion visibility; calibrated slicing.")
    }
}
