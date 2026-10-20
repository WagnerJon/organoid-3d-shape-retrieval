import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import trimesh
from atlas_shape_space import signature, pack, distances, project_hybrid


def test_signature_rigid_rotation_translation_scale():
    mesh=trimesh.creation.icosphere(subdivisions=2)
    v=mesh.vertices * np.array([1,.8,.65]);v[:,0]+=.07*v[:,1]**2
    q,_=np.linalg.qr(np.random.default_rng(27).normal(size=(3,3)))
    np.testing.assert_allclose(signature(v,mesh.faces),signature(3*v@q+[7,2,9],mesh.faces),atol=2e-12)


def test_hybrid_fallback_and_orbit_permutation():
    rng=np.random.default_rng(5);z=rng.normal(size=(3,4,95));s=rng.normal(size=(3,95));g=np.array([.1,.1,.001])
    a=pack(g,z,s);d=distances(a,'shape_hybrid95')
    b=pack(g,z[:,[2,0,3,1]],s)
    np.testing.assert_allclose(distances(b,'shape_hybrid95'),d,atol=1e-12)
    assert abs(d[0,2]-np.linalg.norm(s[0]-s[2]))<1e-10
    assert abs(d[0,1]-min(np.linalg.norm(x-y) for x in z[0] for y in z[1]))<1e-10
    np.testing.assert_array_equal(np.diag(d),0)
    np.testing.assert_allclose(d,d.T)


def test_projection_uses_hybrid_distance_not_packed_flags():
    rng=np.random.default_rng(2);z=rng.normal(size=(8,4,95));s=rng.normal(size=(8,95));v=pack(np.zeros(8),z,s)
    records=[{'representations':{'test':{'vector':x.tolist(),'point':[0,0],'cluster':0}}} for x in v]
    variance=project_hybrid(records,'test')
    assert len(variance)==2 and 0 < sum(variance)<=1.000001
    assert all(np.isfinite(r['representations']['test']['point']).all() for r in records)
    assert len({r['representations']['test']['cluster'] for r in records})==6


def test_membership_rebuild_keeps_hybrid_cell_only():
    from organoid_membership import rebuild
    rng=np.random.default_rng(4);v=pack(np.zeros(4),rng.normal(size=(4,4,95)),rng.normal(size=(4,95)))
    cells=[dict(id=str(i),sample='stack',qc=True,inMainOrganoid=i<3,metrics={'volume_um3':float(i+1)},
                representations={'traditional':dict(vector=[0.],point=[0,0],cluster=0),
                                 'hybrid':dict(vector=x.tolist(),point=[999,999],cluster=0)}) for i,x in enumerate(v)]
    data=dict(cells=cells,organoids=[dict(id='stack',sample='stack',metrics={})],traditionalFeatures=['volume_um3'],
              methods=[dict(id='traditional',metric='euclidean'),dict(id='hybrid',metric='shape_hybrid95')],provenance='Test')
    rebuild(data)
    assert 'hybrid' not in data['organoids'][0]['representations']
    assert cells[-1]['representations']['hybrid']['point']==[999,999]
    assert data['organoids'][0]['metrics']['volume_um3']==6.
