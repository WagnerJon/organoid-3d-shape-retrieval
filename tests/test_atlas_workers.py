import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

import numpy as np
import tifffile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import atlas_worker as worker
import atlas_worker_client as client
from prepare_atlas_dataset import inspect,build


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='atlas-workers-test-');cls.root=Path(cls.temp.name)
        cls.servers=[];cls.endpoints=[];cls.credentials=cls.root/'credentials.json'
        for i in range(2):
            server,fp=worker.create_server(cls.root/f'worker{i}',host='127.0.0.1')
            server.state.machine=f'fixture-mac-{i}';server.state.computeDirectory=server.state.folder/'compute-slot'
            threading.Thread(target=server.serve_forever,daemon=True).start()
            endpoint=dict(id=server.state.id,name=f'Test Mac {i+1}',host='127.0.0.1',port=server.server_port,fingerprint=fp)
            client.pairing(endpoint,server.state.code,cls.credentials)
            cls.servers.append(server);cls.endpoints.append(endpoint)
        cls.old_credentials=client.CREDENTIALS;client.CREDENTIALS=cls.credentials
        z,y,x=np.indices((20,36,40));labels=np.zeros(z.shape,np.uint16)
        labels[(z-10)**2+(y-16)**2+(x-15)**2<36]=1
        labels[(z-10)**2+(y-16)**2+(x-25)**2<36]=2
        raw,mask=cls.root/'raw.ome.tif',cls.root/'labels.tif'
        tifffile.imwrite(raw,labels*1000,ome=True,photometric='minisblack',metadata=dict(axes='ZYX',PhysicalSizeZ=2.,PhysicalSizeY=.5,PhysicalSizeX=.5))
        tifffile.imwrite(mask,labels,photometric='minisblack',metadata=dict(axes='ZYX'))
        entry=inspect(raw);entry.update(labels=str(mask),labelStat=[mask.stat().st_size,mask.stat().st_mtime_ns])
        # Distinct source files represent different organoids, even when geometry is repeated.
        entries=[]
        for i in range(3):
            path=cls.root/f'organoid{i}.ome.tif';path.write_bytes(raw.read_bytes())
            a=inspect(path);a.update(labels=str(mask),labelStat=entry['labelStat']);entries.append(a)
        cls.request=dict(title='Distributed fixture',inputs=entries,settings=dict(model='cpsam_v2',scale=.35,diameter=30.,gpu=False,background=True),harmonics=False,excludeDetached=True,execution=dict(useLocal=False,workers=cls.endpoints))

    @classmethod
    def tearDownClass(cls):
        client.CREDENTIALS=cls.old_credentials
        for server in cls.servers:server.shutdown();server.server_close()
        cls.temp.cleanup()

    def test_ome_zarr_is_materialized_before_remote_dispatch(self):
        import zarr
        from atlas_ome_zarr import inspect_store,selection_entry
        source=self.root/'worker-import.ome.zarr';g=zarr.open_group(str(source),mode='w',zarr_format=3)
        masks=tifffile.imread(self.request['inputs'][0]['labels'])
        raw=np.stack([np.stack([masks*1000]),np.stack([masks*1000])])
        g.create_array('0',shape=raw.shape,dtype=raw.dtype,chunks=(1,1,20,36,40),dimension_names=['t','c','z','y','x'])[:]=raw
        axes=[dict(name=a,**(dict(unit='micrometer') if a in 'zyx' else {})) for a in 'tczyx']
        g.attrs.update(dict(ome=dict(multiscales=[dict(axes=axes,datasets=[dict(path='0',coordinateTransformations=[dict(type='scale',scale=[1,1,2,.5,.5])])])],omero=dict(channels=[dict(label='membrane')]))))
        labels=g.create_group('labels');labels.attrs.update(dict(ome=dict(labels=['cell'])))
        mask=labels.create_group('cell');data=np.stack([masks,masks]);mask.create_array('0',shape=data.shape,dtype=data.dtype,chunks=(1,20,36,40))[:]=data
        mask.attrs.update(dict(ome=dict(multiscales=[dict(axes=[a for a in axes if a['name']!='c'],datasets=[dict(path='0',coordinateTransformations=[dict(type='scale',scale=[1,2,.5,.5])])])],**{'image-label':{}})))
        image=inspect_store(source);request=copy.deepcopy(self.request)
        request['inputs']=[selection_entry(image,t,0,0,'cell') for t in [0,1]]
        request.update(harmonics=False)
        atlas=build(request,self.root/'zarr-distributed');dataset=json.loads((atlas/'dataset.json').read_text())
        self.assertEqual(len(dataset['cells']),4)
        self.assertEqual([s['sourceOMEZarr']['time'] for s in dataset['samples']],[0,1])
        for index in range(1,3):
            run=self.root/'zarr-distributed/stacks'/f'Stack_{index:03d}'
            self.assertTrue((run/'worker_provenance.json').exists())
            self.assertIn('/inputs/',json.loads((run/'run.json').read_text())['source'])

    def test_numpy_is_materialized_before_remote_dispatch(self):
        raw = self.root/'worker-import.npy'; mask = self.root/'worker-labels.npy'
        np.save(raw, tifffile.imread(self.request['inputs'][0]['path']))
        np.save(mask, tifffile.imread(self.request['inputs'][0]['labels']))
        entry = inspect(raw)
        entry.update(spacing=[2.,.5,.5],labels=str(mask),labelStat=[mask.stat().st_size,mask.stat().st_mtime_ns])
        request = copy.deepcopy(self.request); request['inputs'] = [entry]
        atlas = build(request,self.root/'numpy-distributed')
        dataset = json.loads((atlas/'dataset.json').read_text())
        self.assertEqual(len(dataset['cells']),2)
        self.assertEqual(Path(dataset['samples'][0]['sourceNumPy']['path']).resolve(),raw.resolve())
        self.assertEqual(Path(dataset['samples'][0]['sourceNumPy']['labels']).resolve(),mask.resolve())
        run = self.root/'numpy-distributed/stacks/Stack_001'
        self.assertTrue((run/'worker_provenance.json').exists())
        self.assertIn('/inputs/',json.loads((run/'run.json').read_text())['source'])
        self.assertTrue(Path(dataset['samples'][0]['sourceImage']).is_file())

    def test_cancellation_single_slot_and_checkpoint_retry(self):
        endpoint=self.endpoints[0];server=self.servers[0];token=client.token_for(endpoint)
        entry=self.request['inputs'][0]
        raw=client.upload(endpoint,entry['path'],token,lambda s:None)
        label=client.upload(endpoint,entry['labels'],token,lambda s:None)
        settings=dict(self.request['settings'],diameter=31.,linkingMode='streamed')
        payload=dict(spec=dict(rawHash=raw,labelHash=label,spacing=entry['spacing'],settings=settings),compatibility=worker.engine())
        job=client.api(endpoint,'POST','/jobs',payload,token)['id']
        other=copy.deepcopy(payload);other['spec']['settings']['diameter']=32.
        with self.assertRaisesRegex(RuntimeError,'another stack'):
            client.api(endpoint,'POST','/jobs',other,token)
        client.api(endpoint,'DELETE',f'/jobs/{job}',token=token)
        self.assertEqual(client.api(endpoint,'GET',f'/jobs/{job}',token=token)['status'],'cancelled')
        started=time.time()
        while server.state.active and time.time()-started<10:time.sleep(.1)
        self.assertIsNone(server.state.active)
        retried=client.api(endpoint,'POST','/jobs',payload,token)
        self.assertEqual(retried['id'],job)
        started=time.time()
        while time.time()-started<30:
            status=client.api(endpoint,'GET',f'/jobs/{job}',token=token)
            if status['status'] not in ('running','packaging'):break
            time.sleep(.2)
        self.assertEqual(status['status'],'complete',status)

    def test_mixed_local_and_remote(self):
        request=copy.deepcopy(self.request);request['execution']=dict(useLocal=True,workers=[self.endpoints[0]])
        atlas=build(request,self.root/'mixed')
        data=json.loads((atlas/'dataset.json').read_text())
        self.assertEqual(len(data['cells']),6)
        self.assertTrue((self.root/'mixed/stacks/Stack_001/complete.json').exists())
        self.assertTrue((self.root/'mixed/stacks/Stack_002/worker_provenance.json').exists())

    def test_pairing_pin_and_authentication(self):
        endpoint=self.endpoints[0]
        info=client.api(endpoint,'GET','/info');self.assertEqual(info['slots'],1)
        changed=dict(endpoint,fingerprint='0'*64)
        with self.assertRaises(Exception):client.api(changed,'GET','/info')
        with self.assertRaisesRegex(RuntimeError,'Pair this Atlas'):
            client.api(endpoint,'GET','/blobs/'+'0'*64,token='wrong')
        with self.assertRaisesRegex(RuntimeError,'invalid or expired'):
            client.api(endpoint,'POST','/pair',dict(code='000000'))

    def test_upload_cache_and_job_validation(self):
        endpoint=self.endpoints[0];token=client.token_for(endpoint)
        path=Path(self.request['inputs'][0]['path'])
        digest=client.upload(endpoint,path,token,lambda s:None)
        self.assertTrue(client.api(endpoint,'GET','/blobs/'+digest,token=token)['exists'])
        self.assertEqual(client.upload(endpoint,path,token,lambda s:None),digest)
        with self.assertRaisesRegex(RuntimeError,'versions differ'):
            client.api(endpoint,'POST','/jobs',dict(spec={},compatibility={}),token)
        with self.assertRaises(RuntimeError):client.api(endpoint,'GET','/jobs/../../private',token=token)

    def test_two_workers_dask_and_joint_atlas_results(self):
        output=self.root/'joint'
        cached_before=[{p.name for p in (server.state.folder/'blobs').iterdir()} for server in self.servers]
        expected_blobs={worker.sha(self.request['inputs'][0][key]) for key in ('path','labels')}
        atlas=build(self.request,output)
        data=json.loads((atlas/'dataset.json').read_text())
        self.assertEqual(len(data['cells']),6);self.assertEqual(len(data['samples']),3)
        provenance=[]
        for i,entry in enumerate(self.request['inputs']):
            run=output/'stacks'/f'Stack_{i+1:03d}'
            cfg=json.loads((run/'run.json').read_text());self.assertEqual(cfg['source'],entry['path'])
            provenance.append(json.loads((run/'worker_provenance.json').read_text())['worker'])
            self.assertTrue((atlas/f'Stack_{i+1:03d}/1.mesh').exists())
        self.assertEqual(set(provenance),{'Test Mac 1','Test Mac 2'})
        before=(atlas/'dataset.json').read_bytes()
        changed=copy.deepcopy(self.request);changed['execution']['useLocal']=True
        self.assertEqual(build(changed,output),atlas)
        self.assertEqual((atlas/'dataset.json').read_bytes(),before)
        for server,previous in zip(self.servers,cached_before):
            self.assertEqual({p.name for p in (server.state.folder/'blobs').iterdir()},previous | expected_blobs)
        self.assertFalse(any(s.state.active for s in self.servers))

if __name__=='__main__':unittest.main()
