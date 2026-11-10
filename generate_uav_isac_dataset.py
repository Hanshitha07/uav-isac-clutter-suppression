# Dataset generator source used to create this sample bundle. Run with Python after installing numpy and pandas.
import os,json,zipfile
import numpy as np,pandas as pd
out='/mnt/data/synthetic_uav_isac_research';os.makedirs(out,exist_ok=True)
rng=np.random.default_rng(20261019); C=299792458.;fc=3.5e9;lam=C/fc;bs=np.array([0.,0.,25.]); dt=.1; nt,nf,nu=30,100,4
truth=[];ues=[];target=[];clutter=[]
def unit(a): return a/(np.linalg.norm(a)+1e-12)
def ang(a): return np.arctan2(a[1],a[0]),np.arctan2(a[2],np.hypot(a[0],a[1]))
for tr in range(nt):
 mode=['low','high','mixed'][tr%3];mob=['stationary','walking','vehicle'][tr%3]
 ue0=np.c_[rng.uniform(-300,300,nu),rng.uniform(-300,300,nu),rng.uniform(1,2,nu)]
 uev=rng.normal(0,.5,(nu,3))
 if mob=='stationary': uev[:]=0
 elif mob=='walking': uev*=2
 else: uev*=12
 p=np.zeros((nf,3));v=np.zeros((nf,3));p[0]=[rng.uniform(-100,100),rng.uniform(-100,100),rng.uniform(60,130)];h=rng.uniform(-np.pi,np.pi);sp=rng.uniform(8,18);v[0]=[sp*np.cos(h),sp*np.sin(h),rng.uniform(-1,1)]
 for t in range(1,nf):
  prev=v[t-1].copy()
  if mode=='high':
   turn=[0,.055,-.07,.09][(t//12)%4];c,s=np.cos(turn),np.sin(turn);prev[:2]=[c*prev[0]-s*prev[1],s*prev[0]+c*prev[1]];a=rng.normal(0,1.4,3);a[2]=rng.normal(0,.45)
  elif mode=='low': a=rng.normal(0,.45,3);a[2]=rng.normal(0,.12)
  else: a=rng.normal(0,.9,3);a[2]=rng.normal(0,.3)
  v[t]=prev+a*dt;hh=np.linalg.norm(v[t,:2])
  if hh>28:v[t,:2]*=28/hh
  if hh<3:v[t,:2]*=3/max(hh,1e-9)
  v[t,2]=np.clip(v[t,2],-4,4);p[t]=p[t-1]+v[t]*dt
  if p[t,2]<35:p[t,2]=35;v[t,2]=abs(v[t,2])
  if p[t,2]>180:p[t,2]=180;v[t,2]=-abs(v[t,2])
 for t in range(nf):
  tm=t*dt;truth.append(dict(trajectory_id=tr,frame=t,time_s=tm,maneuverability=mode,px_m=p[t,0],py_m=p[t,1],pz_m=p[t,2],vx_mps=v[t,0],vy_mps=v[t,1],vz_mps=v[t,2],speed_mps=np.linalg.norm(v[t])))
  for i in range(nu):
   up=ue0[i]+uev[i]*tm;ues.append(dict(trajectory_id=tr,frame=t,time_s=tm,ue_id=i,mobility=mob,ue_x_true_m=up[0],ue_y_true_m=up[1],ue_z_true_m=up[2],ue_vx_mps=uev[i,0],ue_vy_mps=uev[i,1],ue_vz_mps=uev[i,2],ue_x_noisy_m=up[0]+rng.normal(0,np.sqrt(5)),ue_y_noisy_m=up[1]+rng.normal(0,np.sqrt(5)),ue_z_noisy_m=up[2]+rng.normal(0,np.sqrt(5))))
   d1v=p[t]-up;d2v=p[t]-bs;d1=np.linalg.norm(d1v);d2=np.linalg.norm(d2v);path=d1+d2
   if t:
    uprev=ue0[i]+uev[i]*((t-1)*dt);pr=np.linalg.norm(p[t-1]-uprev)+np.linalg.norm(p[t-1]-bs);rate=(path-pr)/dt
   else:rate=np.dot(unit(d1v),v[t]-uev[i])+np.dot(unit(d2v),v[t])
   dop=-rate/lam;az,el=ang(d2v);snr=float(rng.choice([0,5,10,20]));scale=10**(-snr/20);rs=1+2*scale;ds=1+3*scale;aas=np.deg2rad(.5+2.5*scale)
   target.append(dict(trajectory_id=tr,frame=t,time_s=tm,ue_id=i,maneuverability=mode,ue_mobility=mob,snr_db=snr,bistatic_path_true_m=path,bistatic_path_measured_m=path+rng.normal(0,rs),path_rate_true_mps=rate,doppler_true_hz=dop,doppler_measured_hz=dop+rng.normal(0,ds),azimuth_true_rad=az,azimuth_measured_rad=az+rng.normal(0,aas),elevation_true_rad=el,elevation_measured_rad=el+rng.normal(0,aas),range_noise_sigma_m=rs,doppler_noise_sigma_hz=ds,angle_noise_sigma_rad=aas))
   for q in range(3):
    sc=np.array([rng.uniform(-350,350),rng.uniform(-350,350),rng.uniform(0,30)]);clp=np.linalg.norm(sc-up)+np.linalg.norm(sc-bs);clr=np.dot(unit(up-sc),uev[i]);cld=-clr/lam
    clutter.append(dict(trajectory_id=tr,frame=t,time_s=tm,ue_id=i,clutter_id=q,ue_mobility=mob,scatter_x_m=sc[0],scatter_y_m=sc[1],scatter_z_m=sc[2],clutter_path_m=clp,clutter_doppler_hz=cld,clutter_doppler_measured_hz=cld+rng.normal(0,2),clutter_power_db=rng.normal(-8,4)))
frames={'uav_ground_truth.csv':truth,'ue_positions.csv':ues,'target_measurements.csv':target,'clutter_measurements.csv':clutter}
for fn,rows in frames.items():pd.DataFrame(rows).to_csv(os.path.join(out,fn),index=False)
meta={'dataset_type':'synthetic prototype, not original paper data','seed':20261019,'n_trajectories':nt,'n_frames_per_trajectory':nf,'n_ues':nu,'dt_s':dt,'carrier_frequency_hz':fc,'wavelength_m':lam,'base_station_xyz_m':bs.tolist(),'ue_position_noise_variance_m2':5,'maneuverability_modes':['low','high','mixed'],'ue_mobility_profiles':['stationary','walking','vehicle'],'limitations':['No full complex OFDM CSI or ray-tracing scene.','Measurements are simplified synthetic observables.','Cannot reproduce exact Figure 5 without original model and algorithm implementations.']}
json.dump(meta,open(os.path.join(out,'scenario_metadata.json'),'w'),indent=2)
open(os.path.join(out,'README.md'),'w').write('Synthetic UAV ISAC dataset prototype, not original paper data. Files: uav_ground_truth.csv (truth states), ue_positions.csv (true/noisy UE positions), target_measurements.csv (noisy bistatic path, Doppler, azimuth/elevation), clutter_measurements.csv (simplified scatterer paths/Doppler), scenario_metadata.json. Sample: 30 trajectories x 100 frames, 4 UEs, dt=0.1s. Supports initial mobility/clutter/noise/tracking experiments. Does not contain complex OFDM CSI and is not an exact reproduction of paper Figure 5.')
# Include a script that documents how to regenerate via same bundled script (this is a self-contained copy).
source=open('/tmp/generate.py').read()
open(os.path.join(out,'generate_uav_isac_dataset.py'),'w').write('# Dataset generator source used to create this sample bundle. Run with Python after installing numpy and pandas.\n'+source)
zpath='/mnt/data/synthetic_uav_isac_research_bundle.zip'
with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
 for fn in os.listdir(out):z.write(os.path.join(out,fn),arcname=fn)
print(zpath,os.path.getsize(zpath))
for fn,rows in frames.items():print(fn,len(rows))
