import Foundation
import AppKit
import SceneKit
import Metal
import simd

struct Item:Decodable {
    let id,mesh:String;let source:Bool;let sourceOrder:Int
    let color,organoidColor,organoidPosition,target,quaternion:[Float]
    let organoidRadius,targetRadius:Float
}
struct Spec:Decodable {let width,height,fps,frames:Int;let nodes:[Item];let output:String}
struct Pose {var p:SIMD3<Float>;var r:Float;var q:simd_quatf;var alpha:Float;var colorProgress:Float}
let spec=try JSONDecoder().decode(Spec.self,from:Data(contentsOf:URL(fileURLWithPath:CommandLine.arguments[1])))
guard let device=MTLCreateSystemDefaultDevice() else {fatalError("Metal GPU unavailable")}
func smooth(_ x:Float)->Float {let u=max(0,min(1,x));return u*u*(3-2*u)}
func vector(_ a:[Float])->SIMD3<Float>{SIMD3(a[0],a[1],a[2])}
func mix(_ a:SIMD3<Float>,_ b:SIMD3<Float>,_ t:Float)->SIMD3<Float>{a*(1-t)+b*t}
let base=simd_quatf(angle:0.35,axis:SIMD3(1,0,0))*simd_quatf(angle:-0.45,axis:SIMD3(0,1,0))
let end=simd_quatf(angle:0.9,axis:SIMD3(0,1,0))*base
let offset=SIMD3<Float>(0,-20,0)
let sourceCount=max(1,spec.nodes.filter {$0.source}.count)
func pose(_ item:Item,_ t:Float)->Pose {
    let target=vector(item.target),original=vector(item.organoidPosition)
    let q=simd_quatf(vector:SIMD4(item.quaternion[0],item.quaternion[1],item.quaternion[2],item.quaternion[3]))
    let spin=simd_quatf(angle:2*Float.pi*smooth((t-7)/4.8),axis:SIMD3(0,1,0))*q
    var result=Pose(p:target,r:item.targetRadius,q:spin,alpha:0,colorProgress:1)
    if item.source {
        result.alpha=1
        result.colorProgress=t<3.5 ? 0:1
        if t<2 {
            let turn=simd_quatf(angle:0.9*smooth(t/2),axis:SIMD3(0,1,0))*base
            result.p=turn.act(original)+offset;result.r=item.organoidRadius;result.q=turn
        } else if t<3.5 {
            result.p=end.act(original)*(1+0.55*smooth((t-2)/1.5))+offset;result.r=item.organoidRadius;result.q=end
        } else if t<7 {
            let fraction=Float(item.sourceOrder)/Float(max(1,sourceCount-1))
            let u=smooth((t-3.5-fraction*0.55)/2.9)
            result.colorProgress=u
            result.p=mix(end.act(original)*1.55+offset,target,u)
            result.p.y += sin(u*Float.pi)*(65+fraction*40)
            result.p.z += sin(u*Float.pi)*90
            result.r=item.organoidRadius*(1-u)+item.targetRadius*u;result.q=simd_slerp(end,q,u)
        }
    } else {
        result.alpha=smooth((t-4.5)/2.5)
    }
    return result
}
let scene=SCNScene();scene.background.contents=NSColor(red:12/255,green:20/255,blue:31/255,alpha:1)
let camera=SCNNode();camera.camera=SCNCamera();camera.camera!.usesOrthographicProjection=true
camera.camera!.orthographicScale=Double(spec.height)/2;camera.camera!.zNear=1;camera.camera!.zFar=20000
camera.position=SCNVector3(0,0,10000);scene.rootNode.addChildNode(camera)
let ambient=SCNNode();ambient.light=SCNLight();ambient.light!.type = .ambient;ambient.light!.intensity=430;ambient.light!.color=NSColor.white;scene.rootNode.addChildNode(ambient)
let light=SCNNode();light.light=SCNLight();light.light!.type = .directional;light.light!.intensity=800;light.eulerAngles=SCNVector3(-0.5,-0.5,0);scene.rootNode.addChildNode(light)
var nodes:[SCNNode]=[]
for item in spec.nodes {
    let data=try Data(contentsOf:URL(fileURLWithPath:item.mesh))
    let counts=data.prefix(8).withUnsafeBytes {Array($0.bindMemory(to:UInt32.self))};let n=Int(counts[0]),f=Int(counts[1])
    guard data.count==8+n*24+f*12 else {fatalError("Invalid mesh")}
    let values=data.subdata(in:8..<(8+n*24)).withUnsafeBytes {Array($0.bindMemory(to:Float.self))}
    let v=(0..<n).map {SCNVector3(values[$0*3],values[$0*3+1],values[$0*3+2])}
    let ns=(0..<n).map {SCNVector3(values[n*3+$0*3],values[n*3+$0*3+1],values[n*3+$0*3+2])}
    let element=SCNGeometryElement(data:data.suffix(f*12),primitiveType:.triangles,primitiveCount:f,bytesPerIndex:4)
    let geometry=SCNGeometry(sources:[SCNGeometrySource(vertices:v),SCNGeometrySource(normals:ns)],elements:[element])
    let m=SCNMaterial();m.lightingModel = .blinn
    m.diffuse.contents=NSColor(red:CGFloat(item.color[0]),green:CGFloat(item.color[1]),blue:CGFloat(item.color[2]),alpha:1)
    m.specular.contents=NSColor(white:0.3,alpha:1);m.shininess=0.35;geometry.materials=[m]
    let node=SCNNode(geometry:geometry);scene.rootNode.addChildNode(node);nodes.append(node)
}
let renderer=SCNRenderer(device:device,options:nil);renderer.scene=scene;renderer.pointOfView=camera;renderer.autoenablesDefaultLighting=false
try FileManager.default.createDirectory(atPath:spec.output,withIntermediateDirectories:true)
let indices: [Int] = CommandLine.arguments.count>2 ? CommandLine.arguments[2].split(separator:",").compactMap {Int($0)}:Array(0..<spec.frames)
for i in indices {
    autoreleasepool {
        SCNTransaction.begin();SCNTransaction.animationDuration=0;SCNTransaction.disableActions=true
        for (j,item) in spec.nodes.enumerated() {
            let state=pose(item,Float(i)/Float(spec.fps)),node=nodes[j]
            let color=mix(vector(item.organoidColor),vector(item.color),state.colorProgress)
            node.geometry?.firstMaterial?.diffuse.contents=NSColor(red:CGFloat(color.x),green:CGFloat(color.y),blue:CGFloat(color.z),alpha:1)
            node.simdPosition=state.p;node.simdOrientation=state.q;node.simdScale=SIMD3(repeating:state.r);node.opacity=CGFloat(state.alpha);node.isHidden=state.alpha<0.001
        }
        SCNTransaction.commit()
        let snapshot=renderer.snapshot(atTime:Double(i)/Double(spec.fps),with:CGSize(width:spec.width,height:spec.height),antialiasingMode:.multisampling2X)
        guard let tiff=snapshot.tiffRepresentation,let bitmap=NSBitmapImageRep(data:tiff),let png=bitmap.representation(using:.png,properties:[:]) else {fatalError("Snapshot failed")}
        try! png.write(to:URL(fileURLWithPath:spec.output).appendingPathComponent(String(format:"%04d.png",i)))
    }
    if i%20==0 || indices.count<10 {print("Rendered story frame \(i+1)/\(spec.frames)");fflush(stdout)}
}
