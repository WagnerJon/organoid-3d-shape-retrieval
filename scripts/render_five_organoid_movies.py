"""Render and publish five movies sequentially, limiting temporary disk use."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'results/five_organoid_movies'
DEST=Path('/Volumes/CHUNKY/Scientific_data/GFP-24hrs-2ndy-cleared-MV_all5_analysis/renderings/traditional_global_umap')
FFMPEG='/opt/homebrew/bin/ffmpeg'
def run(args):
    subprocess.run([str(x) for x in args],check=True)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    DEST.mkdir(parents=True,exist_ok=True)
    validation=[]
    for n in range(1,6):
        p=WORK/f'MV{n}'
        name=f'MV{n}_traditional_global_umap'
        movie=DEST/f'{name}.mp4'
        print(f'Rendering organoid MV{n} (240 frames)',flush=True)
        run([ROOT/'.cache/render_five_organoid_movies',p/'scene.json'])
        run([FFMPEG,'-hide_banner','-loglevel','error','-y','-framerate','20','-i',p/'frames/%04d.png',
            '-framerate','20','-i',p/'overlays/%04d.png','-filter_complex','[0:v][1:v]overlay=0:0:format=auto,format=yuv420p[v]',
            '-map','[v]','-frames:v','240','-an','-c:v','libx264','-threads','4','-preset','slow','-crf','18','-movflags','+faststart',movie])
        run([FFMPEG,'-hide_banner','-loglevel','error','-y','-i',movie,'-ss','10','-frames:v','1',DEST/f'{name}_poster.jpg'])
        if n==1:
            run([FFMPEG,'-hide_banner','-loglevel','error','-y','-i',movie,'-filter_complex',
                'fps=12,scale=1120:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192:stats_mode=full[p];[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle',
                '-loop','0',DEST/f'{name}.gif'])
        info=json.loads(subprocess.check_output(['/opt/homebrew/bin/ffprobe','-v','error','-show_streams','-show_format','-of','json',str(movie)]))
        stream=next(s for s in info['streams'] if s['codec_type']=='video')
        assert int(stream['nb_frames'])==240 and stream['width']==1920 and stream['height']==1080
        assert abs(float(info['format']['duration'])-12)<.01
        run([FFMPEG,'-v','error','-i',movie,'-f','null','-'])
        validation.append(dict(file=movie.name,SHA256=digest(movie),frames=240,duration=12,width=1920,height=1080,decode='passed'))
        # Keep selected inspection frames, remove the other bulky intermediates.
        for frame in (p/'frames').glob('*.png'):
            if frame.stem not in {'0000','0065','0140','0200'}:
                frame.unlink()
        print(f'Finished and verified MV{n}: {movie}',flush=True)
    provenance=json.loads((WORK/'provenance.json').read_text())
    dataset=Path(provenance['dataset'])
    assert digest(dataset/'dataset.json')==provenance['datasetSHA256']
    data=json.loads((dataset/'dataset.json').read_text())
    for cell in data['cells']:
        assert digest(dataset/cell['sample']/f"{cell['label']}.mesh")==provenance['sourceMeshSHA256'][cell['id']]
    specs=[json.loads((WORK/f'MV{n}/scene.json').read_text()) for n in range(1,6)]
    target_maps=[{c['id']:c['target'][:2] for c in s['nodes']} for s in specs]
    assert all(m==target_maps[0] for m in target_maps)
    assert [sum(c['source'] for c in s['nodes']) for s in specs]==[201,210,179,185,186]
    for name in ['global_umap_coordinates.csv','provenance.json','global_umap.npz']:
        shutil.copy2(WORK/name,DEST/name)
    (DEST/'validation.json').write_text(json.dumps(dict(status='passed',movies=validation,shared_coordinates=True,source_dataset_and_all_meshes_unchanged=True),indent=2))
    (DEST/'README.md').write_text('''# Five organoids → a shared traditional-morphometry UMAP\n\nFive 12-second, 1920 × 1080 MP4 videos. Each begins with the original reconstructed organoid, separates its cells, and moves those cells into the same global UMAP. The other four organoids fade into the background. At the final positions the cell surfaces rotate through 360°.\n\nThe UMAP uses all 961 segmented objects and all nine traditional size/shape features: volume, surface area, sphericity, elongation, flatness, solidity, and three principal-axis lengths. It uses the existing morph_all vectors (original preprocessing and training-reference scaling), Euclidean distances, 30 neighbors, min_dist 0.3, seed 42. The single fitted map is rotated 90° for a landscape layout without changing distances or neighbor relationships.\n\nAll 214 QC-flagged candidates are retained; these are segmented objects rather than validated biological cell counts. Missing solidity is handled by the comparison pipeline's existing training-reference imputation, not by changing recorded measurements. UMAP is an exploratory projection, not a cell-type annotation.\n\nOriginal measured mesh topology and surfaces are preserved. Reconstruction positions and relative physical sizes are retained at the beginning; cell display sizes are normalized in the map for visibility. Motion is illustrative, not a biological process. No source dataset or measurements were changed.\n\nIn the organoid view, cells use the Atlas categorical label palette: hue (label modulo 11)/11, saturation 0.55, brightness 0.83. These identity colors do not encode a measurement. During movement into UMAP, each cell smoothly transitions to its measured sphericity color, synchronized with its own motion. The UMAP uses the same orange–lavender–teal sphericity scale in all five videos. The display range is 0.30–1.00. Values outside that range are clipped for color only; measured values are unchanged. Some segmentation-derived sphericities exceed the physical ideal of 1, so the legend does not imply these are validated spherical objects. All cells reach full opacity in the shared map. Coordinate tables and provenance are included. MV1 also has a compact looping GIF.\n''')
    print('All five videos complete; shared coordinates and all source geometry verified.',flush=True)
if __name__=='__main__':main()
