import Foundation
import AppKit
import SceneKit
import Metal

struct CellNode: Decodable { let id,mesh:String; let x,y,radius:Double; let color:[Double] }
struct Spec: Decodable { let width,height,frames,fps:Int; let nodes:[CellNode]; let output:String }
let spec=try JSONDecoder().decode(Spec.self,from:Data(contentsOf:URL(fileURLWithPath:CommandLine.arguments[1])))
guard let device=MTLCreateSystemDefaultDevice() else { fatalError("Metal GPU is unavailable") }
let scene=SCNScene();scene.background.contents=NSColor(red:12/255,green:20/255,blue:31/255,alpha:1)
let camera=SCNNode();camera.camera=SCNCamera();camera.camera!.usesOrthographicProjection=true
camera.camera!.orthographicScale=Double(spec.height)/2;camera.camera!.zNear=1;camera.camera!.zFar=20000
camera.position=SCNVector3(0,0,10000);scene.rootNode.addChildNode(camera)
let ambient=SCNNode();ambient.light=SCNLight();ambient.light!.type = .ambient;ambient.light!.intensity=600;ambient.light!.color=NSColor.white;scene.rootNode.addChildNode(ambient)
let light=SCNNode();light.light=SCNLight();light.light!.type = .directional;light.light!.intensity=900
light.eulerAngles=SCNVector3(-0.5,-0.5,0);scene.rootNode.addChildNode(light)
var cache:[String:SCNGeometry]=[:];var rotations:[SCNNode]=[]
for item in spec.nodes {
    let geometry:SCNGeometry
    if let existing=cache[item.mesh] { geometry=existing } else {
        let data=try Data(contentsOf:URL(fileURLWithPath:item.mesh))
        let counts=data.prefix(8).withUnsafeBytes {Array($0.bindMemory(to:UInt32.self))};let n=Int(counts[0]),f=Int(counts[1])
        guard data.count==8+n*24+f*12 else {fatalError("Invalid mesh")}
        let values=data.subdata(in:8..<(8+n*24)).withUnsafeBytes {Array($0.bindMemory(to:Float.self))}
        let v=(0..<n).map {SCNVector3(values[$0*3],values[$0*3+1],values[$0*3+2])}
        let ns=(0..<n).map {SCNVector3(values[n*3+$0*3],values[n*3+$0*3+1],values[n*3+$0*3+2])}
        let element=SCNGeometryElement(data:data.suffix(f*12),primitiveType:.triangles,primitiveCount:f,bytesPerIndex:4)
        geometry=SCNGeometry(sources:[SCNGeometrySource(vertices:v),SCNGeometrySource(normals:ns)],elements:[element])
        let material=SCNMaterial();material.lightingModel = .blinn
        material.diffuse.contents=NSColor(red:item.color[0],green:item.color[1],blue:item.color[2],alpha:1)
        material.specular.contents=NSColor(white:0.3,alpha:1);material.shininess=0.35;material.isDoubleSided=false
        geometry.materials=[material];cache[item.mesh]=geometry
    }
    let position=SCNNode();position.position=SCNVector3(item.x,item.y,0)
    let rotating=SCNNode();rotating.scale=SCNVector3(item.radius,item.radius,item.radius)
    let pose=SCNNode(geometry:geometry);pose.eulerAngles=SCNVector3(0.32,-0.4,0)
    rotating.addChildNode(pose);position.addChildNode(rotating);scene.rootNode.addChildNode(position);rotations.append(rotating)
}
let renderer=SCNRenderer(device:device,options:nil);renderer.scene=scene;renderer.pointOfView=camera
renderer.autoenablesDefaultLighting=false;renderer.isPlaying=false
try FileManager.default.createDirectory(atPath:spec.output,withIntermediateDirectories:true)
for i in 0..<spec.frames {
    autoreleasepool {
        SCNTransaction.begin();SCNTransaction.animationDuration=0;SCNTransaction.disableActions=true
        let angle=CGFloat(i)*2*CGFloat.pi/CGFloat(spec.frames)
        for node in rotations {node.eulerAngles.y=angle}
        SCNTransaction.commit()
        let snapshot=renderer.snapshot(atTime:Double(i)/Double(spec.fps),with:CGSize(width:spec.width,height:spec.height),antialiasingMode:.multisampling2X)
        guard let tiff=snapshot.tiffRepresentation,let bitmap=NSBitmapImageRep(data:tiff),let png=bitmap.representation(using:.png,properties:[:]) else {fatalError("Snapshot failed")}
        try! png.write(to:URL(fileURLWithPath:spec.output).appendingPathComponent(String(format:"%04d.png",i)))
    }
    if i%12==0 {print("Rendered frame \(i+1)/\(spec.frames)");fflush(stdout)}
}
