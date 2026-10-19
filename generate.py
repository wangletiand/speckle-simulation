"""WaveProp thin-screen PSF + spatially incoherent intensity superposition.
No autocorrelation extraction or image reconstruction is performed."""
import pathlib,sys,json,gzip,hashlib,time,platform,importlib.metadata as md
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'packages'))
import requests,numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from waveprop.fresnel import fresnel_one_step
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def writejson(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def targets():
    src=ROOT/'source';src.mkdir(exist_ok=True)
    p=src/'t10k-images-idx3-ubyte.gz';q=src/'t10k-labels-idx1-ubyte.gz'
    for file in [p,q]:
        if not file.exists():
            r=requests.get('https://storage.googleapis.com/cvdf-datasets/mnist/'+file.name,timeout=90);r.raise_for_status();file.write_bytes(r.content)
    ims=np.frombuffer(gzip.decompress(p.read_bytes()),np.uint8,offset=16).reshape(-1,28,28)
    lbs=np.frombuffer(gzip.decompress(q.read_bytes()),np.uint8,offset=8)
    out=[]
    for d in range(10):
        idx=int(np.flatnonzero(lbs==d)[0]);a=np.zeros((65,65),np.float32)
        a[16:48,16:48]=np.asarray(Image.fromarray(ims[idx]).resize((32,32),Image.Resampling.BILINEAR),np.float32)/255
        out.append(('mnist_'+str(d),a,dict(dataset='MNIST test',index=idx,label=d)))
    for name in ['three_points','cross','ring','triangle','leaf','chair']:
        im=Image.new('L',(65,65));dr=ImageDraw.Draw(im)
        if name=='three_points':
            for x,y in [(24,23),(42,28),(31,42)]:dr.point((x,y),fill=255)
        if name=='cross':dr.rectangle((29,18,35,46),fill=255);dr.rectangle((18,29,46,35),fill=255)
        if name=='ring':dr.ellipse((19,19,45,45),outline=255,width=3)
        if name=='triangle':dr.polygon([(32,18),(18,46),(46,46)],outline=255,width=3)
        if name=='leaf':dr.ellipse((23,18,41,44),fill=255);dr.line((32,35,32,48),fill=255,width=2)
        if name=='chair':dr.rectangle((22,18,27,43),fill=255);dr.rectangle((22,32,44,37),fill=255);dr.rectangle((39,35,44,47),fill=255)
        out.append((name,np.asarray(im,dtype=np.float32)/255,dict(dataset='procedural calibration target',name=name)))
    return out
def main():
    t=time.time()
    for s in ['targets','psfs','pairs','previews']: (ROOT/s).mkdir(exist_ok=True)
    objs=targets()
    for name,o,meta in objs:
        np.save(ROOT/'targets'/(name+'.npy'),o);Image.fromarray((o*255).astype('uint8')).save(ROOT/'targets'/(name+'.png'))
    N=1024;dx=4e-6;wv=532e-9;u=.3;diam=.0012
    x=(np.arange(N)-N//2)*dx;X,Y=np.meshgrid(x,x);r2=X*X+Y*Y;k=2*np.pi/wv
    aperture=(r2<(diam/2)**2).astype(float)
    rows=[];checks=[];preview=[]
    for seed in [1001,1002,1003]:
        rng=np.random.default_rng(seed)
        phase=gaussian_filter(rng.normal(size=(N,N)),.7)
        phase=phase/phase.std()*4.
        field=aperture*np.exp(1j*(phase+k*r2/(2*u)))
        for z in [.1,.2]:
            group='seed'+str(seed)+'_z'+str(round(z*1000))+'mm'
            U,x2,y2=fresnel_one_step(field,wv,dx,z)
            h=abs(U)**2;h=h/h.sum();dout=float(x2[0,1]-x2[0,0]);dobj=dout*u/z
            # Independently propagate an off-axis point source; compare interior PSFs after predicted shift.
            shift=8;tilted=field*np.exp(1j*2*np.pi*shift*X/(N*dx))
            Ut,_,_=fresnel_one_step(tilted,wv,dx,z);ht=abs(Ut)**2;ht/=ht.sum()
            ref=np.roll(h,shift,axis=1)
            sl=np.s_[128:-128,128:-128]
            err=float(np.linalg.norm(ht[sl]-ref[sl])/np.linalg.norm(ref[sl]))
            assert err<1e-8,err
            np.savez_compressed(ROOT/'psfs'/(group+'.npz'),psf=h.astype('float32'),phase=phase.astype('float32'),aperture=aperture.astype('uint8'))
            cfg=dict(group=group,seed=seed,wavelength_m=wv,screen_sampling_m=dx,screen_grid=N,screen_diameter_m=diam,object_to_screen_m=u,screen_to_camera_m=z,camera_sampling_m=dout,object_sampling_m=dobj,magnification=-z/u,phase_std_rad=4.,phase_smoothing_sigma_px=.7,spatial_coherence='zero between distinct object pixels; exact intensity sum',temporal_coherence='monochromatic',screen_model='single zero-thickness phase screen',memory_effect='ideal paraxial thin-screen shift invariance, not a calibrated finite-thickness ME range',sensor_shape=[512,512],convolution='linear full then central crop; no circular wrap',camera_peak_electrons=20000.,camera_read_noise_electrons=3.,camera_offset_electrons=100.)
            writejson(ROOT/'psfs'/(group+'.json'),cfg)
            checks.append(dict(group=group,tilt_shift_px=shift,relative_intensity_shift_error=err,psf_sum=float(h.sum()),psf_contrast=float(h.std()/h.mean())))
            for oi,(name,o,meta) in enumerate(objs):
                # Negative magnification reverses both object axes on detector.
                full=fftconvolve(o[::-1,::-1],h,mode='full');a=(full.shape[0]-512)//2
                clean=np.maximum(full[a:a+512,a:a+512],0)
                # Electron scale changes exposure per image; raw units and scale retained.
                scale=20000./max(float(clean.max()),1e-30)
                nrng=np.random.default_rng(seed*1000+round(z*100)+oi)
                electrons=nrng.poisson(clean*scale)+nrng.normal(0,3,clean.shape)+100
                noisy=(electrons-100)/scale
                ident=group+'_'+name;p=ROOT/'pairs'/(ident+'.npz')
                np.savez_compressed(p,speckle_clean=clean.astype('float32'),speckle_noisy=noisy.astype('float32'),electrons=electrons.astype('float32'),electrons_per_intensity=scale)
                # PNG is a preview scaled to per-image peak; quantitative arrays are NPZ.
                Image.fromarray(np.uint16(np.clip(electrons/65535,0,1)*65535)).save(ROOT/'previews'/(ident+'.png'))
                row=dict(id=ident,target='targets/'+name+'.npy',target_png='targets/'+name+'.png',measurement='pairs/'+ident+'.npz',psf='psfs/'+group+'.npz',parameters='psfs/'+group+'.json',object=meta,retained_energy_fraction=float(clean.sum()/full.sum()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),valid_for='ideal spatially incoherent thin-screen OME baseline',detector_object_orientation='object rotated 180 degrees by negative magnification')
                assert np.isfinite(clean).all() and np.isfinite(noisy).all() and clean.max()>0
                rows.append(row)
                if seed==1001 and z==.1 and oi in [0,1,2,10,12,15]:preview.append((name,o,clean,electrons))
            print('SIM_GROUP_DONE',group,flush=True)
    # Check intensity superposition independently using three points and shifted PSFs.
    o=[v[1] for v in objs if v[0]=='three_points'][0];direct=np.zeros((1088,1088))
    for yy,xx in np.argwhere(o[::-1,::-1]>0):direct[yy:yy+N,xx:xx+N]+=h*o[::-1,::-1][yy,xx]
    c=fftconvolve(o[::-1,::-1].astype('float64'),h,mode='full')
    superposition_error=float(np.linalg.norm(c-direct)/np.linalg.norm(direct));assert superposition_error<1e-12
    fig,axs=plt.subplots(len(preview),4,figsize=(12,15))
    for ax,(name,o,clean,e) in zip(axs,preview):
        for aa,img,title in zip(ax,[o,clean,e,clean[192:320,192:320]],['Target: '+name,'Clean intensity','Noisy electrons','Central speckle detail']):
            aa.imshow(img,cmap='gray');aa.set_title(title);aa.axis('off')
    fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=140);plt.close(fig)
    (ROOT/'manifest.jsonl').write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in rows),encoding='utf-8')
    versions={n:md.version(n) for n in ['numpy','scipy','torch','torchvision','Pillow','matplotlib','requests','waveprop']}
    writejson(ROOT/'environment.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),versions=versions))
    writejson(ROOT/'validation.json',dict(pairs=len(rows),unique_targets=len(objs),groups=len(checks),checks=checks,three_point_superposition_error=superposition_error,elapsed_seconds=time.time()-t,retained_energy_range=[min(v['retained_energy_fraction'] for v in rows),max(v['retained_energy_fraction'] for v in rows)]))
    print('SIM_COMPLETE',len(rows),'pairs',round(time.time()-t,2),'s',flush=True)
if __name__=='__main__':main()
