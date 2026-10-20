import SwiftUI

struct LearnedShapeComparison: View {
    @EnvironmentObject var store: ExplorerStore
    @Environment(\.dismiss) private var dismiss
    @StateObject private var cameras = CameraLink()
    @State private var reset = 0
    @State private var error: String?
    let item: Specimen
    var body: some View {
        VStack(alignment:.leading,spacing:14) {
            HStack {
                VStack(alignment:.leading) {
                    Text("PointNet++ reconstruction").font(.title2.bold())
                    Text("\(item.id) · 128D surface embedding · shared physical scale").foregroundStyle(.secondary)
                }
                Spacer()
                Button("Reset views") { reset += 1 }
                Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            }
            if let folder = store.folder, let sample = store.dataset?.samples.first(where:{$0.id == item.sample}),
               let path = item.pointReconstructions?["pointnet_shape128"] {
                HStack(spacing:16) {
                    panel(title:"Original segmented surface",folder:folder,sample:sample,points:nil)
                    panel(title:"Decoded surface · 2,048 points",folder:folder,sample:sample,points:folder.appendingPathComponent(path))
                }
            }
            if let error { Text(error).foregroundStyle(.orange) }
            Text("The prediction is reconstructed from the shape embedding, not copied from the segmentation. The original surface, measurements and microscopy stay unchanged. The baseline is exploratory and is not rotation invariant.")
                .font(.caption).foregroundStyle(.secondary)
        }.padding(22).frame(width:1000,height:610)
    }
    func panel(title:String,folder:URL,sample:Sample,points:URL?) -> some View {
        VStack(alignment:.leading,spacing:8) {
            Text(title).font(.headline)
            ReconstructionView(folder:folder,sample:sample,cells:[item],selectedLabel:item.label,isolate:true,reset:reset,centered:true,
                               cameraSpan:max(1,item.metrics["major_axis_um"] ?? pow(item.volume,1.0/3.0))*1.5,
                               pointCloudURL:points,link:cameras,onError:{error=$0})
                .clipShape(RoundedRectangle(cornerRadius:10))
        }
    }
}
