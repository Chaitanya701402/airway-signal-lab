"""Offline, illustrative 3D cutaway. No anatomy, flow or device measurements."""

import json


def render_airway(kind: str, clock_s: float) -> str:
    """Return a self-contained canvas scene for a Streamlit component iframe.

    Geometry follows the scenario label, never the model prediction. Scene
    dimensions, lighting and motion are decorative and must not be interpreted
    as physiological estimates. No external JavaScript or image assets are used.
    """
    if kind not in {"normal", "cough", "ambient", "movable", "fixed"}:
        kind = "normal"
    scene = json.dumps({"kind": kind, "time": float(clock_s)}, allow_nan=False)
    return _HTML.replace("__SCENE__", scene)


_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box}body{margin:0;background:#101e29;color:#e4eff5;font:13px/1.4 system-ui,sans-serif}
.scene{position:relative;border:1px solid #304656;border-radius:12px;overflow:hidden;background:radial-gradient(ellipse at 45% 42%,#213b4c,#101e29 75%)}
canvas{display:block;width:100%;height:266px;touch-action:none;cursor:grab}canvas:active{cursor:grabbing}
.label{position:absolute;left:16px;top:12px;pointer-events:none}.eyebrow{color:#a9c3d3;font-size:11px;letter-spacing:.08em}.name{font-size:17px;margin-top:5px}
.foot{display:flex;gap:10px;flex-wrap:wrap;justify-content:space-between;padding:8px 13px;background:#142532;color:#b9cbd6;font-size:11px}
button{color:#dcebf3;background:#213949;border:1px solid #496170;border-radius:5px;padding:6px 9px;font:inherit;cursor:pointer}
.controls{position:absolute;right:12px;bottom:48px;display:flex;gap:6px}.truth{color:#eed69d}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto}}@media(max-width:420px){.name{font-size:14px}.controls{bottom:62px}.label{max-width:90%}}
</style></head><body><div class="scene">
<div class="label"><div class="eyebrow">SCENARIO VIEW · ILLUSTRATIVE 3D</div><div class="name" id="name"></div></div>
<canvas id="airway" aria-label="Rotatable schematic cutaway airway. Geometry is illustrative, not a measured airway." role="img"></canvas>
<div class="controls"><button id="cutaway" type="button" aria-pressed="true">Cutaway on</button><button id="reset" type="button">Reset view</button></div>
<div class="foot"><span>Drag to rotate · shape and motion are not physiological measurements</span><span class="truth">Known scenario, not model prediction</span></div></div>
<script>
(() => {
  'use strict';
  const scene=__SCENE__;
  const names={normal:'Normal breathing',cough:'Real cough · ambient-class confounder',ambient:'Ambient / speech noise',movable:'Movable obstruction · SYNTHETIC',fixed:'Fixed obstruction · SYNTHETIC'};
  document.getElementById('name').textContent=names[scene.kind];
  const canvas=document.getElementById('airway'),ctx=canvas.getContext('2d');
  let yaw=-0.26,pitch=0.30,cutaway=true,drag=null,W=500,H=266;
  const pi=Math.PI;
  function transform(p){let [x,y,z]=p;const a=x*Math.cos(yaw)+z*Math.sin(yaw),b=-x*Math.sin(yaw)+z*Math.cos(yaw);return [a,y*Math.cos(pitch)-b*Math.sin(pitch),y*Math.sin(pitch)+b*Math.cos(pitch)];}
  function project(p){const [x,y,z]=transform(p),s=Math.min(W/6.8,62)*5/(5+z);return [W*.5+x*s,H*.54+y*s,z];}
  function poly(points,color,alpha=1){const out=points.map(project);return {pts:out,z:out.reduce((s,p)=>s+p[2],0)/out.length,color,alpha};}
  function cylPoint(x,a,r=0.68){return [x,Math.cos(a)*r+0.055*x*x,Math.sin(a)*r];}
  function draw(){
    ctx.clearRect(0,0,W,H);const surfaces=[];
    const segments=42,around=48;
    for(let j=0;j<segments;j++){
      const x=-2.65+j*5.3/segments,x2=-2.65+(j+1)*5.3/segments;
      for(let k=0;k<around;k++){
        const a=k*2*pi/around,a2=(k+1)*2*pi/around;
        if(cutaway&&a>pi*.92&&a<pi*1.85)continue;
        const ribs=(j%4===0),r=ribs?.76:.70;
        const light=.6+.4*Math.max(0,Math.cos(a-3.9));
        const red=Math.round((ribs?215:167)*light),green=Math.round((ribs?157:119)*light),blue=Math.round((ribs?155:128)*light);
        surfaces.push(poly([cylPoint(x,a,r),cylPoint(x2,a,r),cylPoint(x2,a2,r),cylPoint(x,a2,r)],`rgb(${red},${green},${blue})`,.95));
        if(cutaway&&!ribs)surfaces.push(poly([cylPoint(x,a,.62),cylPoint(x2,a,.62),cylPoint(x2,a2,.62),cylPoint(x,a2,.62)],'#734f57',.63));
      }
    }
    const obstructed=scene.kind==='movable'||scene.kind==='fixed';
    if(obstructed){
      const cx=scene.kind==='movable'?Math.sin(scene.time*.9)*.34:.55;
      const cy=scene.kind==='movable'?-.04:.02;
      for(let j=0;j<18;j++)for(let k=0;k<28;k++){
        const t=j*pi/18,t2=(j+1)*pi/18,a=k*2*pi/28,a2=(k+1)*2*pi/28;
        const sphere=(t,a)=>[cx+.51*Math.cos(t),cy+.55*Math.sin(t)*Math.cos(a),.50*Math.sin(t)*Math.sin(a)];
        const brightness=.60+.4*Math.max(0,Math.cos(a-4));
        surfaces.push(poly([sphere(t,a),sphere(t2,a),sphere(t2,a2),sphere(t,a2)],`rgb(${Math.round(240*brightness)},${Math.round(164*brightness)},${Math.round(105*brightness)})`));
      }
    }
    surfaces.sort((a,b)=>b.z-a.z);
    for(const p of surfaces){ctx.globalAlpha=p.alpha;ctx.fillStyle=p.color;ctx.beginPath();p.pts.forEach((q,i)=>i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]));ctx.closePath();ctx.fill();}
    ctx.globalAlpha=1;
    // Particle marks indicate the direction of the illustrative scene only.
    // They never report calibrated airflow, obstruction severity or physiology.
    if(cutaway){for(let i=0;i<14;i++){
      const x=-2.35+((i*.39+scene.time*.35)%4.7);
      if(obstructed&&x>-.20&&x<1.15)continue;
      const p=project([x,.08*Math.sin(i*2.1),-.45]);
      ctx.beginPath();ctx.fillStyle='#77d7e2';ctx.globalAlpha=.55;ctx.arc(p[0],p[1],2.3,0,2*pi);ctx.fill();
    }}ctx.globalAlpha=1;
  }
  function resize(){const r=canvas.getBoundingClientRect();W=r.width;H=r.height;const d=Math.min(window.devicePixelRatio||1,2);canvas.width=Math.round(W*d);canvas.height=Math.round(H*d);ctx.setTransform(d,0,0,d,0,0);draw();}
  canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});
  canvas.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*.008;pitch=Math.max(-1.1,Math.min(1.1,pitch+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY];draw();});
  canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('pointercancel',()=>drag=null);
  document.getElementById('cutaway').addEventListener('click',e=>{cutaway=!cutaway;e.currentTarget.textContent=cutaway?'Cutaway on':'Cutaway off';e.currentTarget.setAttribute('aria-pressed',String(cutaway));draw();});
  document.getElementById('reset').addEventListener('click',()=>{yaw=-.26;pitch=.30;draw();});
  new ResizeObserver(resize).observe(canvas);resize();
})();
</script></body></html>'''

# Originality checklist: The scene and controls are template-like illustrations.
# Rewrite the scene-design rationale and interpretation in your own voice.
# [STUDENT TO ADD: our own observation from run X / bench test Y]
# Run your university's plagiarism checker and disclose AI assistance per policy.
