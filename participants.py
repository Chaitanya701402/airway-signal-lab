"""Reconstruct checksummed real source clips and explicitly synthetic derivatives offline."""
from pathlib import Path
import json,base64,zlib,hashlib,math
import numpy as np
import soundfile as sf
from scipy.signal import butter,sosfilt

def participant_config(root: Path, entry: dict) -> tuple[Path,dict]:
    payload=json.loads((root/entry['path']).read_text(encoding='utf-8'))
    folder=root/'assets'/'participants'/entry['id'];folder.mkdir(parents=True,exist_ok=True)
    waves={}
    for key in ['breath','cough']:
        item=payload[key];raw=zlib.decompress(base64.b64decode(item['zlib_base64_float32']))
        if hashlib.sha256(raw).hexdigest()!=item['sha256_float32']:raise ValueError('Participant audio checksum mismatch')
        wave=np.frombuffer(raw,dtype='<f4').copy()
        if len(wave)!=item['sample_count'] or not np.isfinite(wave).all():raise ValueError('Invalid participant audio')
        waves[key]=wave
    base=np.tile(waves['breath'],math.ceil(96000/len(waves['breath'])))[:96000]
    t=np.arange(len(base))/16000;rng=np.random.default_rng(20261081)
    hz=float(rng.uniform(2.5,5.5));cut=float(rng.uniform(700,1600));phase=float(rng.uniform(0,6.28))
    movable=sosfilt(butter(3,cut,fs=16000,output='sos'),base)*(.035+.75*(.5+.5*np.sin(2*np.pi*hz*t+phase))**3)
    rng=np.random.default_rng(20261082);fixed_cut=float(rng.uniform(350,900));gain=float(rng.uniform(.025,.08))
    fixed=sosfilt(butter(3,fixed_cut,fs=16000,output='sos'),base)*gain
    for name,wave in [('normal',base),('cough',waves['cough']),('movable',movable),('fixed',fixed)]:
        sf.write(folder/(name+'.wav'),wave.astype('float32'),16000,subtype='FLOAT')
    config=json.loads((root/'config.ready.json').read_text(encoding='utf-8'))
    config['participant_id']=entry['id'];config['source_scope']='Breathing source participant only. Real cough is from a separate Coswara donor. Both obstruction clips are SYNTHETIC.'
    for event in config['scenario']:
        kind=event['kind']
        if kind!='ambient':event['path']=str((folder/(kind+'.wav')).relative_to(root)).replace('\\','/')
        if kind in ['normal','movable','fixed']:event['source_id']=entry['breath_source']
        if kind=='cough':event['source_id']=entry['cough_source'];event['repeat']=False
        if kind=='normal':event['duration_s']=6.0
        if kind=='ambient':event['duration_s']=5.0
        if kind=='movable':event['recipe_parameters']={'attenuation_modulation_hz':hz,'lowpass_hz':cut,'phase':phase,'minimum_gain':.035,'maximum_gain':.785}
        if kind=='fixed':event['recipe_parameters']={'lowpass_hz':fixed_cut,'gain':gain}
    # A per-session configuration path prevents concurrent viewers overwriting settings.
    return folder,config

# Originality checklist: standard decoding and provisional DSP recipe; no physiological interpretation.
# Authors must add their own reasoning, use their university checker and disclose AI assistance.