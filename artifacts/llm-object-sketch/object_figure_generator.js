// gen_object_figure.js  --  LLM-object internal architecture
// hand-drawn (rough.js) sketch, Architects Daughter font, matching the
// style of figure_hand (the sequence diagram).
const { JSDOM } = require('jsdom');
const rough = require('roughjs');
const fs = require('fs');

const W = 1180, H = 760;
const dom = new JSDOM(`<!DOCTYPE html><svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"></svg>`);
const doc = dom.window.document;
const svg = doc.querySelector('svg');
const rc = rough.svg(svg);

const HAND = "Architects Daughter";
const INK = '#1f2430';
const PINK = '#fbf9fa';
const ENGINE_BG = '#e0e7ff';  // soft blue for engines
const STATE_BG = '#fef3c7';   // yellow for state-bearing boxes
const TOOLS_BG = '#dcfce7';   // soft green for tools
const MOD_BG = '#fee2e2';     // soft red for the modifier engine

// glyph-width table for proper text layout
const _wtab = JSON.parse(fs.readFileSync('widths.json','utf8')).widths;
function textWidth(s, size){
  let em=0;
  for(const ch of s){ const w=_wtab[ch.codePointAt(0)]; em += (w==null?0.5:w); }
  return em*size;
}

function add(node){ svg.appendChild(node); }
function text(x,y,s,opts={}){
  const t = doc.createElementNS('http://www.w3.org/2000/svg','text');
  t.setAttribute('x',x); t.setAttribute('y',y);
  t.setAttribute('font-family',HAND);
  t.setAttribute('font-size',opts.size||17);
  t.setAttribute('fill',opts.fill||INK);
  t.setAttribute('text-anchor',opts.anchor||'middle');
  if(opts.weight) t.setAttribute('font-weight',opts.weight);
  t.textContent = s;
  add(t); return t;
}
function rrect(x,y,w,h,fill,stroke,opts={}){
  add(rc.rectangle(x,y,w,h,{
    roughness:opts.r??1.6, bowing:opts.b??1.4,
    fill:fill||'#fdfdfb', fillStyle:'solid',
    stroke:stroke||INK, strokeWidth:opts.sw||1.5 }));
}
function line(x1,y1,x2,y2,opts={}){
  add(rc.line(x1,y1,x2,y2,{
    roughness:opts.r??1.2, bowing:opts.b??1.0,
    stroke:opts.stroke||INK, strokeWidth:opts.sw||1.4 }));
}
function arrow(x1,y1,x2,y2,stroke){
  const s = stroke||INK;
  line(x1,y1,x2,y2,{stroke:s,sw:1.5,r:1.0,b:0.7});
  const dx=x2-x1, dy=y2-y1, len=Math.hypot(dx,dy);
  const ux=dx/len, uy=dy/len, a=10;
  // arrowhead
  const px=-uy, py=ux;
  line(x2,y2,x2-ux*a+px*5,y2-uy*a+py*5,{stroke:s,sw:1.5,r:0.6});
  line(x2,y2,x2-ux*a-px*5,y2-uy*a-py*5,{stroke:s,sw:1.5,r:0.6});
}
// small hand-drawn envelope, centered on (cx,cy)
function envelope(cx, cy){
  const w=28, h=18;
  const x=cx-w/2, y=cy-h/2;
  rrect(x, y, w, h, '#ffffff', INK, {r:1.2, sw:1.3});
  // flap: two diagonals from top corners meeting at the middle top edge
  line(x, y, x+w/2, y+h*0.55, {sw:1.2, r:0.8, b:0.4});
  line(x+w, y, x+w/2, y+h*0.55, {sw:1.2, r:0.8, b:0.4});
}

// ---- outer LLM-OBJECT container (pink, like the sequence diagram heads) ----
const OX=200, OY=80, OW=920, OH=640;
rrect(OX, OY, OW, OH, PINK, INK, {sw:2.0, r:1.8});
text(OX+OW/2, OY+30, 'LLM-OBJECT', {size:22, weight:'bold'});

// ---- NAME / DESCRIPTION subsection (top) ----
const ndY = OY+55, ndH = 70;
rrect(OX+30, ndY, OW-60, ndH, '#ffffff', INK);
text(OX+50, ndY+25, 'name:  ...', {size:17, anchor:'start'});
text(OX+50, ndY+50, 'description:  ...', {size:17, anchor:'start'});

// ---- ENGINES subsection ----
const engY = ndY + ndH + 20, engH = 220;
rrect(OX+30, engY, OW-60, engH, '#f8f9ff', INK);
text(OX+OW/2, engY+22, 'ENGINES', {size:18, weight:'bold'});

// Row 1: Planner → Executor → Evaluator (ReAct loop)
const engBoxW = 140, engBoxH = 46, engRowY = engY+48;
const planX = OX+90;
const execX = planX + engBoxW + 40;
const evalX = execX + engBoxW + 40;
rrect(planX, engRowY, engBoxW, engBoxH, ENGINE_BG, INK);
text(planX+engBoxW/2, engRowY+29, 'planner', {size:17, weight:'bold'});
rrect(execX, engRowY, engBoxW, engBoxH, ENGINE_BG, INK);
text(execX+engBoxW/2, engRowY+29, 'executor', {size:17, weight:'bold'});
rrect(evalX, engRowY, engBoxW, engBoxH, ENGINE_BG, INK);
text(evalX+engBoxW/2, engRowY+29, 'evaluator', {size:17, weight:'bold'});
// arrows between them
arrow(planX+engBoxW+4, engRowY+engBoxH/2, execX-4, engRowY+engBoxH/2);
arrow(execX+engBoxW+4, engRowY+engBoxH/2, evalX-4, engRowY+engBoxH/2);
// loop from evaluator back to executor (re-act)
line(evalX+engBoxW/2, engRowY+engBoxH+4, evalX+engBoxW/2, engRowY+engBoxH+22, {r:0.7});
line(evalX+engBoxW/2, engRowY+engBoxH+22, execX+engBoxW/2, engRowY+engBoxH+22, {r:0.7});
arrow(execX+engBoxW/2, engRowY+engBoxH+22, execX+engBoxW/2, engRowY+engBoxH+5);

// Row 2: Self-Modifier — its own row, visually distinct (red-pink), separated by clear gap
const smY = engRowY + engBoxH + 56;
const smW = 220, smX = OX + (OW - smW)/2;
rrect(smX, smY, smW, engBoxH, MOD_BG, '#b91c1c');
text(smX+smW/2, smY+29, 'self-modifier', {size:17, weight:'bold', fill:'#7f1d1d'});

// ---- INTERNAL STATE (yellow box) ----
const stY = engY + engH + 28, stH = 110;
const stW = 360, stX = OX + 40;
rrect(stX, stY, stW, stH, STATE_BG, '#d97706');
text(stX+stW/2, stY+26, 'internal state (obj)', {size:17, weight:'bold', fill:'#92400e'});
// squiggle lines suggesting NL content
for(let i=0;i<3;i++){
  const yy = stY+48+i*18;
  line(stX+30, yy, stX+stW-30, yy, {stroke:'#92400e', sw:1.0, r:2.5, b:1.8});
}

// ---- PLANS DICTIONARY (next to state) ----
const pdX = stX + stW + 40, pdW = OW - 60 - stW - 40 - 40;
rrect(pdX, stY, pdW, stH, '#fef3c7', '#d97706');
text(pdX+pdW/2, stY+30, 'plans dictionary', {size:17, weight:'bold', fill:'#92400e'});
text(pdX+pdW/2, stY+58, 'for execution', {size:17, weight:'bold', fill:'#92400e'});

// arrows from state/plans up to engines (land at engines bottom border, not through it)
arrow(stX+stW/2, stY-6, stX+stW/2, engY+engH+2);
arrow(pdX+pdW/2, stY-6, pdX+pdW/2, engY+engH+2);
// bidirectional arrows between state and plans (separated vertically so both visible)
arrow(stX+stW+6, stY+stH/2-7, pdX-4, stY+stH/2-7);
arrow(pdX-4, stY+stH/2+7, stX+stW+6, stY+stH/2+7);

// ---- TOOLS subsection ----
const tlY = stY + stH + 20, tlH = 90;
const tlW = 400, tlX = OX + (OW-tlW)/2;
rrect(tlX, tlY, tlW, tlH, TOOLS_BG, '#059669');
text(tlX+28, tlY+24, 'tools', {size:17, weight:'bold', fill:'#065f46', anchor:'start'});
// two tool rows (modify_behavior dropped — self-modifier engine handles that)
const toolRows = ['message_peer', 'api_tools'];
toolRows.forEach((label, i) => {
  text(tlX+50, tlY+52+i*22, label, {size:16, anchor:'start', fill:'#064e3b'});
  add(rc.circle(tlX+38, tlY+47+i*22, 6, {roughness:1.0, fill:'#059669', fillStyle:'solid', stroke:'#059669'}));
});

// ---- MAILBOX (vertical strip on the left of LLM-OBJECT) ----
const mbX = OX-28, mbY = OY+55, mbW = 48, mbH = OH-75;
rrect(mbX, mbY, mbW, mbH, '#f3f4f6', '#6b7280');
const mbLabel = doc.createElementNS('http://www.w3.org/2000/svg','text');
mbLabel.setAttribute('x', mbX+mbW/2); mbLabel.setAttribute('y', mbY+mbH/2);
mbLabel.setAttribute('font-family', HAND);
mbLabel.setAttribute('font-size', 20); mbLabel.setAttribute('font-weight', 'bold');
mbLabel.setAttribute('fill', '#374151'); mbLabel.setAttribute('text-anchor', 'middle');
mbLabel.setAttribute('transform', `rotate(-90 ${mbX+mbW/2} ${mbY+mbH/2})`);
mbLabel.textContent = 'mailbox';
add(mbLabel);

// arrow from mailbox to engines: horizontal, entering planner cleanly from the left
arrow(mbX+mbW+4, engRowY+engBoxH/2, planX-6, engRowY+engBoxH/2);

// ---- MESSAGES arriving on the left (3 envelopes, each with an arrow into the mailbox) ----
text(80, mbY-8, 'messages', {size:19, weight:'bold', anchor:'start'});
const inputYs = [mbY+mbH*0.18, mbY+mbH*0.50, mbY+mbH*0.82];
inputYs.forEach(y => {
  envelope(95, y);            // envelope on the left
  arrow(115, y, mbX-3, y);    // arrow from just right of envelope into mailbox
});

fs.writeFileSync('object_figure.svg', `<?xml version="1.0" encoding="UTF-8"?>\n`+svg.outerHTML);
console.log('wrote object_figure.svg');
