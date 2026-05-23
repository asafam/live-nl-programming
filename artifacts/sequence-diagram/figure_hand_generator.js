// gen_figure.js  --  hand-drawn sequence diagram (rough.js -> SVG)
// Reproduces the locked customer-support live-modification trace
// in an Excalidraw-style sketch aesthetic.
const { JSDOM } = require('jsdom');
const rough = require('roughjs');
const fs = require('fs');

const W = 1180, H = 756;
const dom = new JSDOM(`<!DOCTYPE html><svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"></svg>`);
const doc = dom.window.document;
const svg = doc.querySelector('svg');
const rc = rough.svg(svg);

// embed Architects Daughter so the SVG is self-contained
const fontB64 = fs.readFileSync('ArchitectsDaughter.ttf').toString('base64');
const style = doc.createElementNS('http://www.w3.org/2000/svg','style');
style.textContent = `@font-face{font-family:'ArchitectHand';src:url(data:font/ttf;base64,${fontB64}) format('truetype');}`;
svg.appendChild(style);
const HAND = "Architects Daughter";

// glyph-width table (em fractions) produced by measure.py
const _wtab = JSON.parse(fs.readFileSync('widths.json','utf8')).widths;
function textWidth(s, size){
  let em=0;
  for(const ch of s){ const w=_wtab[ch.codePointAt(0)]; em += (w==null?0.5:w); }
  return em*size;
}
function labelArrow(cx,y,a,b,opts={}){
  const sz=opts.size||16, gap=30;
  const wa=textWidth(a,sz), wb=textWidth(b,sz);
  const total=wa+gap+wb, x0=cx-total/2;
  text(x0+wa/2,y,a,{size:sz,fill:opts.fill,weight:opts.weight});
  inlineArrow(x0+wa+gap/2,y-5);
  text(x0+wa+gap+wb/2,y,b,{size:sz,fill:opts.fill,weight:opts.weight});
}

const INK = '#1f2430';
const BOX = '#fdfdfb';
const PINK = '#fde7f0';
const NOTE_Y = '#fef3c7', NOTE_Y_S = '#d97706';
const NOTE_G = '#d1fae5', NOTE_G_S = '#059669';
const BAND = '#3b4a8a';

function add(node){ svg.appendChild(node); }
function text(x,y,s,opts={}){
  const t = doc.createElementNS('http://www.w3.org/2000/svg','text');
  t.setAttribute('x',x); t.setAttribute('y',y);
  t.setAttribute('font-family',HAND);
  t.setAttribute('font-size',opts.size||17);
  t.setAttribute('fill',opts.fill||INK);
  t.setAttribute('text-anchor',opts.anchor||'middle');
  if(opts.weight) t.setAttribute('font-weight',opts.weight);
  if(opts.style) t.setAttribute('font-style',opts.style);
  t.textContent = s;
  add(t); return t;
}
function rrect(x,y,w,h,fill,stroke){
  add(rc.rectangle(x,y,w,h,{
    roughness:1.6, bowing:1.4, fill:fill||BOX, fillStyle:'solid',
    stroke:stroke||INK, strokeWidth:1.5 }));
}
function line(x1,y1,x2,y2,opts={}){
  add(rc.line(x1,y1,x2,y2,{
    roughness:opts.r??1.4, bowing:opts.b??1.2,
    stroke:opts.stroke||INK, strokeWidth:opts.sw||1.4,
    strokeLineDash:opts.dash||undefined }));
}
function arrow(x1,y1,x2,y2,stroke,dashed){
  const s = stroke||INK;
  line(x1,y1,x2,y2,{stroke:s,sw:1.6,r:1.1,b:0.8,dash:dashed?[7,5]:undefined});
  const dir = x2>x1?1:-1, a=9;
  line(x2,y2,x2-dir*a,y2-5,{stroke:s,sw:1.6,r:0.8,b:0.5});
  line(x2,y2,x2-dir*a,y2+5,{stroke:s,sw:1.6,r:0.8,b:0.5});
}
// simplified robot icon
function robot(cx,cy){
  add(rc.rectangle(cx-13,cy-12,26,22,{roughness:1.5,fill:'#eef1f7',fillStyle:'solid',stroke:INK,strokeWidth:1.4}));
  add(rc.circle(cx-6,cy-2,5,{roughness:1.4,fill:INK,fillStyle:'solid',stroke:INK}));
  add(rc.circle(cx+6,cy-2,5,{roughness:1.4,fill:INK,fillStyle:'solid',stroke:INK}));
  line(cx,cy-12,cx,cy-19,{sw:1.3,r:1});
  add(rc.circle(cx,cy-21,4,{roughness:1.3,fill:INK,fillStyle:'solid',stroke:INK}));
}
// simplified human icon
function human(cx,cy){
  add(rc.circle(cx,cy-13,9,{roughness:1.4,fill:'#eef1f7',fillStyle:'solid',stroke:INK,strokeWidth:1.4}));
  line(cx,cy-8,cx,cy+8,{sw:1.4});
  line(cx-9,cy-1,cx+9,cy-1,{sw:1.4});
  line(cx,cy+8,cx-7,cy+18,{sw:1.4});
  line(cx,cy+8,cx+7,cy+18,{sw:1.4});
}
function num(x,y,n){ text(x,y,String(n),{size:15,weight:'bold',fill:'#7a7a7a',anchor:'start'}); }
function inlineArrow(cx,cy){
  line(cx-8,cy,cx+8,cy,{sw:1.3,r:0.9,b:0.5});
  line(cx+8,cy,cx+2,cy-4,{sw:1.3,r:0.5});
  line(cx+8,cy,cx+2,cy+4,{sw:1.3,r:0.5});
}

// ---- actor columns (Triage removed; ultra-simple) ----
const A = {
  CSH:{x:150,label:['Customer','Support','Handler'],icon:'robot'},
  T:{x:430,label:['Ticket #4471'],icon:'robot'},
  B:{x:680,label:['Billing','Team'],icon:'robot'},
  V:{x:860,label:['VIP','Team'],icon:'robot'},
  Ad:{x:1050,label:['Administrator'],icon:'human'},
};
const TOP=70, BOT=716;

// TIMELINE arrow far left
text(34,393,'TIMELINE',{size:17,weight:'bold'}); 
svg.lastChild.setAttribute('transform',`rotate(-90 34 393)`);
line(56,90,56,706,{sw:1.6,r:0.8});
line(56,706,51,694,{sw:1.6}); line(56,706,61,694,{sw:1.6});

// actor heads + lifelines
for(const k in A){
  const a=A[k];
  const w = Math.max(...a.label.map(s=>s.length))*9+34;
  if(a.icon==='robot') robot(a.x-w/2-26,32); else human(a.x-w/2-24,32);
  const isLLM = (a.icon==='robot');
  rrect(a.x-w/2,14,w,a.label.length>1?(a.label.length*20+8):34, isLLM?PINK:undefined);
  a.label.forEach((ln,i)=>text(a.x,34+i*20-(a.label.length>1?2:0),ln,{size:16,weight:'bold'}));
  line(a.x,TOP,a.x,BOT,{stroke:'#9aa0ad',sw:1.1,r:0.6,b:0.4});
}

function note(cx,y,label,green){
  const lines = Array.isArray(label) ? label : [label];
  const w = Math.max(...lines.map(l=>textWidth(l,14))) + 24;
  const h = lines.length>1 ? (lines.length*19+8) : 26;
  rrect(cx-w/2, y-h/2, w, h, green?NOTE_G:NOTE_Y, green?NOTE_G_S:NOTE_Y_S);
  const startY = y - (lines.length-1)*9.5;
  lines.forEach((l,i)=> text(cx, startY+i*19+4, l, {size:14}));
  return w;
}

// ---- sequence (ultra-simple: no Triage, no numbers) ----
// event in
{ const et='event: new request ("Change payment detail by Sara Chen (Platinum)")';
  const ew=textWidth(et,14)+28;
  add(rc.rectangle(70,86,ew,28,{roughness:1.5,fill:NOTE_Y,fillStyle:'solid',stroke:NOTE_Y_S,strokeWidth:1.4}));
  text(70+ew/2,105,et,{size:14}); }
arrow(A.CSH.x,156,A.T.x-8,156); text((A.CSH.x+A.T.x)/2,144,'create ticket',{size:14});
note(A.T.x,204,['type = "VIP"',"customer issue = 'Billing'"]);
arrow(A.T.x,274,A.B.x-8,274); text((A.T.x+A.B.x)/2,262,'take care of me',{size:14});
note(A.T.x,322,'owner = BillingTeam');

// ---- time band: continuous scribble, irregular soft-edged text clearing ----
const by=376;
{
  const label='time passes \u2013 requests keep arriving, system never stops';
  const tw=textWidth(label,17);
  const cx=(60+1040)/2;
  const L0=60, R0=1050, top=by-15, bot=by+15;
  const BG='#ffffff';

  const bp=[];const bs=Math.round((R0-L0)/24);
  for(let i=0;i<=bs;i++){const x=L0+(R0-L0)*i/bs;bp.push([x,top+(Math.random()*6-3)]);}
  for(let i=bs;i>=0;i--){const x=L0+(R0-L0)*i/bs;bp.push([x,bot+(Math.random()*6-3)]);}
  add(rc.polygon(bp,{roughness:2.4,bowing:1.8,fill:BAND,fillStyle:'hachure',
    hachureGap:4.5,fillWeight:1.1,hachureAngle:-41,stroke:BAND,strokeWidth:1.2}));
  for(let k=0;k<7;k++){
    const sx=L0+Math.random()*(R0-L0);
    line(sx,top-(2+Math.random()*4),sx+(Math.random()*10-5),
      bot+(2+Math.random()*4),{stroke:BAND,sw:0.9,r:2.4,b:1.8});
  }
  function blob(halfW,halfH,jit,op){
    const segs=12,pts=[];
    for(let i=0;i<=segs;i++){const t=i/segs;const x=cx-halfW+2*halfW*t;
      pts.push([x, by-halfH + (Math.random()*jit-jit/2)]);}
    for(let i=segs;i>=0;i--){const t=i/segs;const x=cx-halfW+2*halfW*t;
      pts.push([x, by+halfH + (Math.random()*jit-jit/2)]);}
    const pg=doc.createElementNS('http://www.w3.org/2000/svg','polygon');
    pg.setAttribute('points',pts.map(p=>p.map(v=>v.toFixed(1)).join(',')).join(' '));
    pg.setAttribute('fill',BG);pg.setAttribute('opacity',op);
    svg.appendChild(pg);
  }
  blob(tw/2+34, 19, 11, 0.32);
  blob(tw/2+26, 16, 9,  0.55);
  blob(tw/2+18, 14, 7,  0.80);
  blob(tw/2+12, 12, 5,  1.0);
  text(cx,by+6,label,{size:17});
}

// ---- modification: Admin rewrites Ticket #4471's behavior directly ----
arrow(A.Ad.x,432,A.T.x+8,432,NOTE_G_S);
text((A.T.x+A.Ad.x)/2,420,'VIP tickets should be handled by the VIPTeam',{size:14,fill:'#1f2430',weight:'bold'});
text((A.T.x+A.Ad.x)/2,452,'MODIFICATION',{size:15,weight:'bold',fill:NOTE_G_S});
note(A.T.x,488,'behavior updated',true);
arrow(A.T.x,536,A.B.x-8,536); text((A.T.x+A.B.x)/2,524,'stop working on me',{size:14});
arrow(A.T.x,576,A.V.x-8,576); text((A.T.x+A.V.x)/2,564,'take care of me',{size:14});
note(A.T.x,620,'owner = VIPTeam');
arrow(A.B.x,668,A.T.x+8,668,'#73726c',true); text((A.T.x+A.B.x)/2,656,'I stopped working on you',{size:14,fill:'#73726c'});

fs.writeFileSync('figure_hand.svg', `<?xml version="1.0" encoding="UTF-8"?>\n`+svg.outerHTML);
console.log('wrote figure_hand.svg (ultra-simple, ticket-routes, admin-direct)');
