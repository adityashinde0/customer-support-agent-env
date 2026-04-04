from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from environment import CustomerSupportEnv
from models import Action

app = FastAPI(title="Customer Support OpenEnv API")
env_instance = CustomerSupportEnv()

@app.post("/reset")
def reset_environment():
    try:
        obs = env_instance.reset()
        return {"observation": obs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
def step_environment(action: Action):
    if env_instance.obs is None:
        raise HTTPException(status_code=400, detail="You must call /reset before calling /step.")
    try:
        obs, reward, done, info = env_instance.step(action)
        return {"observation": obs, "reward": reward, "done": done, "info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
def get_current_state():
    if env_instance.obs is None:
        raise HTTPException(status_code=400, detail="Environment has not been initialized. Call /reset.")
    return {"observation": env_instance.state()}

@app.get("/", response_class=HTMLResponse)
def home():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Customer Support OpenEnv — AI Agent Playground</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#ffffff;
  --black:#0a0a0a;
  --gray:#f2f2f2;
  --gray2:#e8e8e8;
  --gray3:#999;
  --lime:#c8ff00;
  
  /* GenZ Vibe Colors */
  --purple:#a855f7; 
  --cyan:#00f3ff;
  --blue:#3b82f6;
  --orange:#f97316;
  --text:#0a0a0a;
  
  /* Bouncy animation timing */
  --bounce: cubic-bezier(0.34, 1.56, 0.64, 1);
}
html{scroll-behavior:smooth}
body{
  font-family:'Outfit',sans-serif;
  background:var(--bg);
  color:var(--text);
  overflow-x:hidden;
}

/* ── NOISE TEXTURE OVERLAY ── */
body::after{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:9999;
  opacity:.025;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
}

/* ── NAV ── */
nav{
  position:fixed;top:0;left:0;right:0;z-index:100;
  display:flex;align-items:center;justify-content:space-between;
  padding:18px 48px;
  background:rgba(255,255,255,.92);
  backdrop-filter:blur(12px);
  border-bottom:2px solid var(--black);
}
.nav-logo{
  font-family:'Bebas Neue',sans-serif;
  font-size:26px;letter-spacing:2px;color:var(--black);
}
.nav-logo span{color:var(--purple)}
.nav-right{display:flex;align-items:center;gap:12px}

/* DEVELOPER CREDITS ANIMATIONS */
.nav-devs {
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 1px;
  text-transform: uppercase;
  padding: 6px 14px;
  border-radius: 100px;
  background: var(--black);
  color: #fff;
  border: 2px solid var(--cyan);
  animation: pulseNeon 3s infinite alternate, floatDev 4s ease-in-out infinite;
  display: flex;
  align-items: center;
  gap: 4px;
}
.nav-devs span {
  color: var(--cyan);
  animation: textShift 3s infinite alternate;
}
@keyframes pulseNeon {
  0% { box-shadow: 0 0 5px var(--cyan); border-color: var(--cyan); }
  100% { box-shadow: 0 0 15px var(--purple); border-color: var(--purple); }
}
@keyframes floatDev {
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-3px); }
}
@keyframes textShift {
  0% { color: var(--cyan); }
  100% { color: var(--purple); }
}

.nav-tag{
  font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;
  padding:6px 14px;border-radius:100px;
  background:var(--lime);color:var(--black);
  border:2px solid var(--black);
}
.nav-docs{
  font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;
  padding:6px 16px;border-radius:100px;
  background:var(--black);color:#fff;
  text-decoration:none;transition:all 0.3s var(--bounce);
  border:2px solid var(--black);
}
.nav-docs:hover{
  background:var(--purple);
  border-color:var(--purple);
  box-shadow: 0 0 15px rgba(168, 85, 247, 0.6);
  transform: translateY(-2px);
}

/* ── HERO ── */
.hero{
  min-height:100vh;
  display:flex;flex-direction:column;
  align-items:center;justify-content:center;
  padding:120px 48px 80px;
  text-align:center;
  position:relative;
  overflow:hidden;
}
.hero-bg-blob{
  position:absolute;
  border-radius:50%;
  filter:blur(80px);
  pointer-events:none;
  z-index:0;
}
.blob1{width:500px;height:500px;background:rgba(200,255,0,.25);top:-100px;right:-80px;animation:drift 8s ease-in-out infinite alternate}
.blob2{width:400px;height:400px;background:rgba(168, 85, 247,.15);bottom:-80px;left:-60px;animation:drift 10s ease-in-out infinite alternate-reverse}
.blob3{width:300px;height:300px;background:rgba(0, 243, 255,.1);top:40%;left:60%;animation:drift 12s ease-in-out infinite alternate}
@keyframes drift{from{transform:translate(0,0) scale(1)}to{transform:translate(30px,20px) scale(1.05)}}

.hero-inner{position:relative;z-index:1;max-width:900px}

.hero-badge{
  display:inline-flex;align-items:center;gap:8px;
  background:var(--black);color:#fff;
  font-size:12px;font-weight:700;letter-spacing:2px;text-transform:uppercase;
  padding:8px 18px;border-radius:100px;margin-bottom:32px;
  animation:fadeUp .5s ease both;
}
.hero-badge-dot{
  width:7px;height:7px;border-radius:50%;background:var(--cyan);
  animation:blink 1.5s ease infinite;
  box-shadow: 0 0 10px var(--cyan);
}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

.hero-title{
  font-family:'Bebas Neue',sans-serif;
  font-size:clamp(80px,14vw,160px);
  line-height:.92;letter-spacing:2px;
  color:var(--black);
  animation:fadeUp .6s .1s ease both;
}
.hero-title .stroke{
  -webkit-text-stroke:3px var(--black);
  color:transparent;
}
.hero-title .fill-lime{color:var(--lime);-webkit-text-stroke:3px var(--black)}
.hero-title .fill-purple{color:var(--purple)}

.hero-sub{
  margin:28px auto 0;max-width:580px;
  font-size:18px;font-weight:500;line-height:1.6;
  color:#444;
  animation:fadeUp .6s .2s ease both;
}
.hero-sub strong{color:var(--black);font-weight:800}

.hero-actions{
  margin-top:40px;display:flex;gap:14px;justify-content:center;flex-wrap:wrap;
  animation:fadeUp .6s .3s ease both;
}
.btn-hero{
  font-family:'Outfit',sans-serif;font-size:15px;font-weight:800;
  letter-spacing:.5px;text-transform:uppercase;
  padding:16px 36px;border-radius:100px;
  border:3px solid var(--black);cursor:pointer;
  transition:all 0.3s var(--bounce);
}
.btn-hero.primary{background:var(--black);color:#fff}
.btn-hero.primary:hover{
  background:var(--purple);
  border-color:var(--purple);
  transform:translateY(-5px) scale(1.05);
  box-shadow:0 12px 24px rgba(168, 85, 247,.4);
}
.btn-hero.secondary{background:var(--lime);color:var(--black)}
.btn-hero.secondary:hover{
  background:#d4ff1a;
  transform:translateY(-5px) scale(1.05);
  box-shadow:0 12px 24px rgba(200,255,0,.4);
}

/* ── MARQUEE ── */
.marquee-wrap{
  border-top:3px solid var(--black);
  border-bottom:3px solid var(--black);
  background:var(--black);
  padding:14px 0;overflow:hidden;white-space:nowrap;
}
.marquee-track{
  display:inline-flex;gap:0;
  animation:marquee 18s linear infinite;
}
.marquee-track span{
  font-family:'Bebas Neue',sans-serif;
  font-size:22px;letter-spacing:3px;color:var(--cyan);
  padding:0 32px;
}
.marquee-track span.sep{color:#fff;padding:0}
@keyframes marquee{from{transform:translateX(0)}to{transform:translateX(-50%)}}

/* ── TASKS SECTION ── */
.tasks-section{padding:80px 48px;border-bottom:3px solid var(--black)}
.section-header{text-align:center;margin-bottom:56px}
.section-eyebrow{
  font-size:12px;font-weight:800;letter-spacing:3px;text-transform:uppercase;
  color:var(--purple);margin-bottom:12px;
}
.section-title{
  font-family:'Bebas Neue',sans-serif;
  font-size:clamp(52px,7vw,96px);
  letter-spacing:2px;line-height:.95;
  color:var(--black);
}
/* Highlight Title matches the screenshot */
.highlight-bg {
  background: #2563eb; 
  color: #fff;
  padding: 0 16px;
  display: inline-block;
  margin-bottom: 4px;
}

/* Updated Task Grid for 5 Items */
.tasks-grid{
  display:flex;
  flex-wrap:wrap;
  gap:16px;
}
.task-card{
  flex: 1 1 calc(33.333% - 16px);
  min-width: 280px;
  padding:36px 32px;
  border:3px solid var(--black);
  border-radius:12px;
  position:relative;transition:all 0.3s var(--bounce);cursor:pointer;
  background: var(--bg);
}
.task-card:hover{background:var(--black);color:#fff; transform:scale(1.02); z-index:10; box-shadow:0 10px 30px rgba(0,0,0,0.15);}
.task-card:hover .task-num{color:var(--cyan)}
.task-card:hover .task-name{color:#fff}
.task-card:hover .task-desc{color:#aaa}
.task-card:hover .task-tag{background:var(--cyan);color:var(--black);border-color:var(--cyan);box-shadow:0 0 10px var(--cyan);}
.task-num{
  font-family:'Bebas Neue',sans-serif;
  font-size:64px;line-height:1;color:var(--gray2);
  margin-bottom:12px;transition:color .2s;
}
.task-tag{
  display:inline-block;font-size:10px;font-weight:800;letter-spacing:2px;text-transform:uppercase;
  padding:4px 12px;border-radius:100px;margin-bottom:16px;
  border:2px solid var(--black);transition:all .2s;
}
.tag-easy{background:rgba(200,255,0,.3);color:var(--black)}
.tag-med{background:rgba(249, 115, 22,.15);color:var(--orange)}
.tag-hard{background:rgba(168, 85, 247,.12);color:var(--purple)}
.task-name{font-size:22px;font-weight:800;margin-bottom:10px;transition:color .2s}
.task-desc{font-size:13px;font-weight:500;line-height:1.6;color:#555;transition:color .2s}

/* ── ACTIONS SECTION ── */
.actions-section{padding:80px 48px;background:var(--black);border-bottom:3px solid var(--black)}
.actions-section .section-eyebrow{color:var(--cyan)}
.actions-section .section-title{color:#fff}
.actions-grid{
  display:grid;grid-template-columns:repeat(3,1fr);gap:3px;
  margin-top:48px;
}
.action-card{
  background:#111;padding:28px 24px;
  transition:all 0.3s var(--bounce);cursor:default;
  border-radius:4px;
}
.action-card:hover{background:var(--purple);transform:translateY(-6px); box-shadow:0 10px 20px rgba(168, 85, 247, 0.2);}
.action-card:hover .ac-name{color:#fff}
.action-card:hover .ac-desc{color:rgba(255,255,255,0.8)}
.action-card:hover .ac-icon{background:var(--black);color:var(--cyan);box-shadow:0 0 15px rgba(0, 243, 255, 0.5);}
.ac-icon{
  width:44px;height:44px;border-radius:10px;
  background:rgba(255,255,255,.06);
  display:flex;align-items:center;justify-content:center;
  font-size:20px;margin-bottom:16px;transition:all .3s;
}
.ac-name{font-size:14px;font-weight:800;color:#fff;letter-spacing:.3px;margin-bottom:6px;transition:color .2s;font-family:'Outfit',monospace}
.ac-desc{font-size:12px;color:#666;line-height:1.6;transition:color .2s}

/* ── PLAYGROUND SECTION ── */
.playground{padding:80px 48px;border-bottom:3px solid var(--black)}
.playground .section-header{text-align:center;margin-bottom:48px}

.pg-grid{display:grid;grid-template-columns:380px 1fr;gap:3px;border:3px solid var(--black);border-radius:12px;overflow:hidden}

/* LEFT CONTROLS */
.pg-left{background:var(--gray);border-right:3px solid var(--black);display:flex;flex-direction:column}
.pg-block{padding:24px;border-bottom:3px solid var(--black)}
.pg-block:last-child{border-bottom:none;flex:1}

.pg-label{font-size:10px;font-weight:800;letter-spacing:2.5px;text-transform:uppercase;color:#888;margin-bottom:14px}

/* TICKET CARD */
.tkt{background:#fff;border:3px solid var(--black);border-radius:8px;padding:16px}
.tkt-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:14px}
.tkt-id{font-size:11px;font-weight:700;letter-spacing:1px;color:#888}
.tkt-badge{font-size:9px;font-weight:800;letter-spacing:1.5px;text-transform:uppercase;padding:4px 10px;border-radius:100px;border:2px solid var(--black)}
.tkt-badge.idle{background:var(--gray2);color:#888}
.tkt-badge.active{background:var(--cyan);color:var(--black);box-shadow:0 0 8px rgba(0,243,255,0.4);}
.tkt-badge.done{background:var(--purple);color:#fff;border-color:var(--purple)}
.tkt-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;font-size:13px}
.tkt-row:last-of-type{margin-bottom:0}
.tkt-k{font-weight:600;color:#888}
.tkt-v{font-weight:800;color:var(--black)}
.tkt-v.vip{color:var(--orange)}.tkt-v.ent{color:var(--blue)}.tkt-v.std{color:var(--black)}

.prog-bar{margin-top:14px}
.prog-labels{display:flex;justify-content:space-between;font-size:11px;font-weight:700;color:#888;margin-bottom:6px}
.prog-track{height:6px;background:var(--gray2);border-radius:100px;overflow:hidden;border:2px solid var(--black)}
.prog-fill{height:100%;background:var(--black);border-radius:100px;transition:width .5s ease;width:0%}

/* SENTIMENT */
.sent-pills{display:flex;gap:6px}
.sp{
  flex:1;padding:6px 4px;border-radius:100px;
  font-size:9px;font-weight:800;letter-spacing:1px;text-transform:uppercase;
  text-align:center;border:2px solid var(--black);
  background:#fff;color:#888;transition:all .3s var(--bounce);
}
.sp.a-happy{background:var(--lime);color:var(--black);transform:scale(1.05);box-shadow:0 4px 10px rgba(200,255,0,0.3);}
.sp.a-neutral{background:var(--black);color:#fff;transform:scale(1.05);}
.sp.a-frustrated{background:var(--orange);color:#fff;border-color:var(--orange);transform:scale(1.05);box-shadow:0 4px 10px rgba(249,115,22,0.3);}
.sp.a-angry{background:var(--purple);color:#fff;border-color:var(--purple);transform:scale(1.05);box-shadow:0 4px 10px rgba(168,85,247,0.3);}

/* KB */
.kb-result{
  background:#fff;border:3px solid var(--black);border-left:6px solid var(--cyan);
  border-radius:0 8px 8px 0;padding:12px;
  font-size:12px;font-weight:500;line-height:1.7;color:#333;
  display:none;
}
.kb-result.on{display:block; animation: msgPop 0.4s var(--bounce) both;}
.kb-ph{
  font-size:12px;font-weight:600;color:#888;
  padding:10px;background:#fff;border:2px dashed var(--gray2);border-radius:8px;text-align:center;
}
.kb-ph span{color:var(--purple)}

/* FORM */
.fg{margin-bottom:14px}
.fl{font-size:10px;font-weight:800;letter-spacing:2px;text-transform:uppercase;color:#888;margin-bottom:6px;display:block}
select,input[type=text]{
  width:100%;padding:11px 14px;
  background:#fff;border:2px solid var(--black);
  border-radius:8px;color:var(--black);
  font-family:'Outfit',sans-serif;font-size:13px;font-weight:600;
  outline:none;transition:all 0.3s;
  appearance:none;-webkit-appearance:none;
}
select{
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%230a0a0a' stroke-width='2' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 12px center;padding-right:36px;
}
select:focus,input:focus{border-color:var(--cyan);box-shadow:0 0 0 4px rgba(0, 243, 255,.15); transform:scale(1.01);}
.fhide{display:none}

.btn-submit{
  width:100%;padding:14px;
  background:var(--black);color:#fff;
  border:3px solid var(--black);border-radius:100px;
  font-family:'Outfit',sans-serif;font-size:14px;font-weight:800;
  letter-spacing:1px;text-transform:uppercase;
  cursor:pointer;transition:all 0.3s var(--bounce);
}
.btn-submit:hover{
  background:var(--purple);
  border-color:var(--purple);
  transform:translateY(-3px);
  box-shadow:0 8px 20px rgba(168, 85, 247,.4);
}
.btn-submit:active{transform:translateY(0)}
.btn-submit:disabled{opacity:.4;cursor:not-allowed;transform:none;box-shadow:none;}

.btn-row{display:flex;gap:8px;margin-top:10px}
.bsm{
  flex:1;padding:10px;
  background:#fff;border:2px solid var(--black);border-radius:100px;
  font-family:'Outfit',sans-serif;font-size:11px;font-weight:800;
  letter-spacing:1px;text-transform:uppercase;
  cursor:pointer;transition:all 0.3s var(--bounce);color:var(--black);
}
.bsm:hover{background:var(--black);color:#fff;transform:translateY(-2px);}
.bsm.new:hover{
  background:var(--cyan);
  color:var(--black);
  border-color:var(--cyan);
  box-shadow: 0 5px 15px rgba(0,243,255,0.4);
}

/* RIGHT CONVO */
.pg-right{display:flex;flex-direction:column;background:#fff;min-height:600px}

.convo-area{
  flex:1;overflow-y:auto;padding:28px;display:flex;flex-direction:column;gap:14px;min-height:0;
}
/* Cool custom scrollbar for GenZ vibe */
.convo-area::-webkit-scrollbar {
  width: 8px;
}
.convo-area::-webkit-scrollbar-track {
  background: var(--gray);
  border-radius: 8px;
}
.convo-area::-webkit-scrollbar-thumb {
  background: var(--purple);
  border-radius: 8px;
  border: 2px solid var(--gray);
}
.convo-area::-webkit-scrollbar-thumb:hover {
  background: var(--cyan);
}

.convo-empty{
  flex:1;display:flex;flex-direction:column;
  align-items:center;justify-content:center;
  text-align:center;gap:12px;padding:48px;
}
.ce-emoji{font-size:56px; animation: bounce 2s infinite;}
@keyframes bounce { 0%, 100% {transform: translateY(0);} 50% {transform: translateY(-10px);} }
.ce-title{font-family:'Bebas Neue',sans-serif;font-size:36px;letter-spacing:1px;color:var(--black)}
.ce-sub{font-size:14px;font-weight:500;color:#888;max-width:280px;line-height:1.6}

.msg{display:flex;gap:12px;animation:msgPop .3s var(--bounce) both}
@keyframes msgPop{from{opacity:0;transform:translateY(12px) scale(0.95)}to{opacity:1;transform:translateY(0) scale(1)}}
.mavatar{
  width:32px;height:32px;border-radius:8px;
  border:2px solid var(--black);
  display:flex;align-items:center;justify-content:center;
  font-size:14px;flex-shrink:0;
}
.mavatar.cu{background:var(--black);color:#fff}
.mavatar.ag{background:var(--cyan);color:var(--black)}
.mavatar.sy{background:var(--gray2)}
.mbody{flex:1}
.msender{font-size:10px;font-weight:800;letter-spacing:2px;text-transform:uppercase;margin-bottom:5px}
.msender.cu{color:var(--black)}.msender.ag{color:var(--purple)}.msender.sy{color:#888}
.mtext{
  font-size:13px;font-weight:500;line-height:1.6;color:var(--black);
  background:var(--gray);border:2px solid var(--black);
  border-radius:0 10px 10px 10px;padding:10px 14px;
  display:inline-block;max-width:90%;
}
.mtext.sys{background:var(--gray2);border-color:var(--gray2);color:#555;font-size:12px}

/* REWARD LOG */
.rlog{border-top:3px solid var(--black);background:var(--gray);max-height:220px;overflow-y:auto}
.rlog::-webkit-scrollbar { width: 6px; }
.rlog::-webkit-scrollbar-track { background: transparent; }
.rlog::-webkit-scrollbar-thumb { background: #ccc; border-radius: 10px; }

.rlog-head{
  padding:12px 24px;border-bottom:2px solid var(--black);
  display:flex;justify-content:space-between;align-items:center;
  position:sticky;top:0;background:var(--gray);z-index:5;
}
.rlog-title{font-size:10px;font-weight:800;letter-spacing:2px;text-transform:uppercase;color:#888}
.rtotal{font-size:15px;font-weight:900;color:var(--black)}

/* Custom History Log Entry based on Image */
.history-log {
  background: #fff;
  display: flex;
  flex-direction: column;
}
.log-entry {
  padding: 14px 24px;
  border-bottom: 1px solid var(--gray2);
  font-family: 'Outfit', sans-serif;
  font-size: 13px;
  animation: msgPop .3s var(--bounce) both;
  background: #fff;
}
.log-entry:last-child {
  border-bottom: none;
}
.log-action {
  font-size: 14px;
  margin-bottom: 6px;
  color: #38bdf8; /* Light blue */
  font-weight: 500;
}
.log-action strong { font-weight: 700; }
.log-reward {
  font-weight: 700;
  color: #475569;
}


/* SETUP SECTION */
.setup-section{padding:80px 48px;background:var(--gray);border-bottom:3px solid var(--black)}
.setup-section .section-header{text-align:center;margin-bottom:48px}
.setup-grid{display:grid;grid-template-columns:1fr 1fr;gap:24px;max-width:900px;margin:0 auto}
.code-card{background:#fff;border:3px solid var(--black);border-radius:12px;overflow:hidden; transition:transform 0.3s var(--bounce);}
.code-card:hover { transform: translateY(-4px); box-shadow: 0 10px 25px rgba(0,0,0,0.1); }
.code-head{
  background:var(--black);padding:14px 20px;
  display:flex;justify-content:space-between;align-items:center;
}
.code-head-title{font-size:11px;font-weight:800;letter-spacing:2px;text-transform:uppercase;color:var(--cyan)}
.copy-btn{
  font-size:10px;font-weight:800;letter-spacing:1px;text-transform:uppercase;
  padding:4px 12px;border-radius:100px;
  background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);
  color:#fff;cursor:pointer;transition:all 0.3s var(--bounce);
  font-family:'Outfit',sans-serif;
}
.copy-btn:hover{background:var(--purple);color:#fff;border-color:var(--purple);box-shadow:0 0 10px var(--purple);}
.copy-btn.done{background:var(--cyan);color:var(--black); border-color:var(--cyan);}
.code-body{padding:20px;font-size:12px;line-height:1.9;font-family:'Outfit',monospace;background:#fff}
.c-cmd{color:var(--purple);font-weight:700}
.c-val{color:var(--blue)}
.c-cmt{color:#aaa}

/* FOOTER */
footer{
  padding:48px;
  background:var(--black);
  border-top:3px solid var(--black);
  display:flex;justify-content:space-between;align-items:center;
  flex-wrap:wrap;gap:20px;
}
.foot-logo{font-family:'Bebas Neue',sans-serif;font-size:28px;letter-spacing:2px;color:var(--cyan)}
.foot-text{font-size:13px;font-weight:500;color:#555}

/* FOOTER DEV ANIMATION */
.foot-devs {
  font-size: 14px;
  font-weight: 700;
  color: #888;
  display: flex;
  align-items: center;
  gap: 8px;
}
.foot-devs span {
  background: linear-gradient(90deg, var(--cyan), var(--purple), var(--cyan));
  background-size: 200% auto;
  color: transparent;
  -webkit-background-clip: text;
  animation: shine 3s linear infinite;
  font-weight: 900;
  font-size: 16px;
}
@keyframes shine {
  to { background-position: 200% center; }
}

.foot-links{display:flex;gap:20px}
.foot-links a{font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#555;text-decoration:none;transition:color .3s}
.foot-links a:hover{color:var(--purple)}

/* TOAST */
.toast{
  position:fixed;bottom:24px;right:24px;z-index:999;
  background:var(--black);color:#fff;border:3px solid var(--black);
  border-radius:100px;padding:12px 22px;
  font-size:13px;font-weight:700;
  box-shadow:0 8px 28px rgba(0,0,0,.2);
  transform:translateY(16px);opacity:0;
  transition:all .3s var(--bounce);pointer-events:none;
}
.toast.show{transform:translateY(0);opacity:1}
.toast.ok{background:var(--cyan);color:var(--black)}
.toast.err{background:var(--purple); color:#fff; border-color:var(--purple);}

@media(max-width:900px){
  .task-card{border-right:none;border-bottom:3px solid var(--black)}
  .task-card:last-child{border-bottom:none}
  .actions-grid{grid-template-columns:1fr 1fr}
  .pg-grid{grid-template-columns:1fr}
  .pg-left{border-right:none;border-bottom:3px solid var(--black)}
  .setup-grid{grid-template-columns:1fr}
  nav{padding:14px 24px}
  .hero{padding:100px 24px 60px}
  footer{padding:32px 24px; flex-direction:column; text-align:center;}
}
</style>
</head>
<body>

<nav>
  <div class="nav-logo">Customer Support <span>OpenEnv</span></div>
  <div class="nav-right">
    <div class="nav-devs">Developed by <span>Hrushikesh Sarode</span> & <span>Aditya Shinde</span></div>
    <span class="nav-tag">🟢 Live</span>
    <a href="/docs" class="nav-docs" target="_blank">API Docs ↗</a>
  </div>
</nav>

<section class="hero">
  <div class="hero-bg-blob blob1"></div>
  <div class="hero-bg-blob blob2"></div>
  <div class="hero-bg-blob blob3"></div>
  <div class="hero-inner">
    <div class="hero-badge"><div class="hero-badge-dot"></div> OpenEnv Hackathon · v1.0</div>
    <h1 class="hero-title">
      <span class="fill-purple">AI</span><br>
      <span class="stroke">SUPPORT</span><br>
      <span class="fill-lime">AGENT</span>
    </h1>
    <p class="hero-sub">
      Train AI to handle <strong>real customer tickets</strong> — classify issues, search policy, read logs, and resolve or escalate. Shaped rewards. Sentiment scoring. No cap.
    </p>
    <div class="hero-actions">
      <button class="btn-hero primary" onclick="document.getElementById('playground').scrollIntoView({behavior:'smooth'})">▶ Try It Now</button>
      <button class="btn-hero secondary" onclick="document.getElementById('tasks').scrollIntoView({behavior:'smooth'})">See Tasks →</button>
    </div>
  </div>
</section>

<div class="marquee-wrap">
  <div class="marquee-track">
    <span>CLASSIFY ISSUE</span><span class="sep">·</span>
    <span>SEARCH KB</span><span class="sep">·</span>
    <span>CHECK LOGS</span><span class="sep">·</span>
    <span>RESOLVE TICKET</span><span class="sep">·</span>
    <span>ESCALATE TO HUMAN</span><span class="sep">·</span>
    <span>SENTIMENT SCORE</span><span class="sep">·</span>
    <span>REWARD SIGNAL</span><span class="sep">·</span>
    <span>CLASSIFY ISSUE</span><span class="sep">·</span>
    <span>SEARCH KB</span><span class="sep">·</span>
    <span>CHECK LOGS</span><span class="sep">·</span>
    <span>RESOLVE TICKET</span><span class="sep">·</span>
    <span>ESCALATE TO HUMAN</span><span class="sep">·</span>
    <span>SENTIMENT SCORE</span><span class="sep">·</span>
    <span>REWARD SIGNAL</span><span class="sep">·</span>
  </div>
</div>

<section class="tasks-section" id="tasks">
  <div class="section-header">
    <div class="section-eyebrow">Difficulty Levels</div>
    <div class="section-title">
      <span class="">5 TASKS</span><br>
      
    </div>
  </div>
  <div class="tasks-grid">
    <div class="task-card">
      <div class="task-num">01</div>
      <div class="task-tag tag-easy">Easy</div>
      <div class="task-name">Password Reset</div>
      <div class="task-desc">Deterministic resolution requiring a simple KB search.</div>
    </div>
    <div class="task-card">
      <div class="task-num">02</div>
      <div class="task-tag tag-med">Medium</div>
      <div class="task-name">Standard Refund</div>
      <div class="task-desc">Requires classification, KB search for policy checking, and resolution.</div>
    </div>
    <div class="task-card">
      <div class="task-num">03</div>
      <div class="task-tag tag-med">Medium</div>
      <div class="task-name">Vague Complaint</div>
      <div class="task-desc">The user says "It is broken." The agent must proactively ask a clarifying question before proceeding.</div>
    </div>
    <div class="task-card">
      <div class="task-num">04</div>
      <div class="task-tag tag-hard">Medium/Hard</div>
      <div class="task-name">VIP Outage</div>
      <div class="task-desc">Tests if the agent can correctly classify high-urgency technical issues without being distracted by account status keywords.</div>
    </div>
    <div class="task-card">
      <div class="task-num">05</div>
      <div class="task-tag tag-hard">Hard</div>
      <div class="task-name">Hostile/Policy Violation</div>
      <div class="task-desc">The user demands a refund for a non-refundable item. The agent must realize the KB contradicts the user and gracefully use escalate_to_human rather than arguing.</div>
    </div>
  </div>
</section>

<section class="actions-section">
  <div class="section-header">
    <div class="section-eyebrow">What The Agent Can Do</div>
    <div class="section-title">5 ACTIONS.<br>ZERO HALLUCINATIONS.</div>
  </div>
  <div class="actions-grid">
    <div class="action-card">
      <div class="ac-icon">🏷️</div>
      <div class="ac-name">classify_issue</div>
      <div class="ac-desc">Tag the ticket type first. Billing, Technical, or Refund_Request. Miss this and lose points.</div>
    </div>
    <div class="action-card">
      <div class="ac-icon">📚</div>
      <div class="ac-name">search_kb</div>
      <div class="ac-desc">Query the knowledge base with keywords. Returns the exact policy the agent needs to follow.</div>
    </div>
    <div class="action-card">
      <div class="ac-icon">💬</div>
      <div class="ac-name">ask_clarifying_question</div>
      <div class="ac-desc">Talk to the customer. Warning: ask too many times and their patience runs out fast.</div>
    </div>
    <div class="action-card">
      <div class="ac-icon">✅</div>
      <div class="ac-name">resolve_ticket</div>
      <div class="ac-desc">Close the ticket with a solution. Only works if the classification was correct. No shortcuts.</div>
    </div>
    <div class="action-card">
      <div class="ac-icon">🚨</div>
      <div class="ac-name">escalate_to_human</div>
      <div class="ac-desc">Hand off to a real person. Safe fallback — but costs points. Use only when policy demands it.</div>
    </div>
    <div class="action-card" style="background:rgba(168, 85, 247,.08)">
      <div class="ac-icon">⚡</div>
      <div class="ac-name">Reward Signal</div>
      <div class="ac-desc">Every action scores you. Correct classification +0.2. Wrong one -0.2. Perfect resolve +1.0. Clock is ticking.</div>
    </div>
  </div>
</section>

<section class="playground" id="playground">
  <div class="section-header">
    <div class="section-eyebrow">Interactive Demo</div>
    <div class="section-title">PLAY THE<br>ENVIRONMENT.</div>
  </div>

  <div class="pg-grid">
    <div class="pg-left">
      <div class="pg-block">
        <div class="pg-label">Ticket Status</div>
        <div class="tkt">
          <div class="tkt-top">
            <span class="tkt-id" id="ticketId">NO TICKET</span>
            <span class="tkt-badge idle" id="ticketStatus">IDLE</span>
          </div>
          <div class="tkt-row"><span class="tkt-k">Tier</span><span class="tkt-v" id="tierVal">—</span></div>
          <div class="tkt-row"><span class="tkt-k">Category</span><span class="tkt-v" id="catVal" style="color:#888">Not classified</span></div>
          <div class="tkt-row"><span class="tkt-k">Steps</span><span class="tkt-v" id="stepVal">0 / 10</span></div>
          <div class="prog-bar">
            <div class="prog-labels"><span>Progress</span><span id="progPct">0%</span></div>
            <div class="prog-track"><div class="prog-fill" id="progFill"></div></div>
          </div>
        </div>

        <div style="margin-top:16px">
          <div class="pg-label" style="margin-bottom:8px">Sentiment</div>
          <div class="sent-pills">
            <div class="sp" id="sp-happy">😊 Happy</div>
            <div class="sp a-neutral" id="sp-neutral">😐 Neutral</div>
            <div class="sp" id="sp-frustrated">😤 Upset</div>
            <div class="sp" id="sp-angry">😡 Angry</div>
          </div>
        </div>

        <div style="margin-top:16px">
          <div class="pg-label" style="margin-bottom:8px">Knowledge Base</div>
          <div class="kb-result" id="kbResult"></div>
          <div class="kb-ph" id="kbPh">Use <span>search_kb</span> to pull policy</div>
        </div>
      </div>

      <div class="pg-block" style="flex:1">
        <div class="pg-label">Submit Action</div>

        <div class="fg">
          <label class="fl">Action Type</label>
          <select id="actionType" onchange="toggleFields()">
            <option value="classify_issue">classify_issue</option>
            <option value="search_kb">search_kb</option>
            <option value="ask_clarifying_question">ask_clarifying_question</option>
            <option value="resolve_ticket">resolve_ticket</option>
            <option value="escalate_to_human">escalate_to_human</option>
          </select>
        </div>
        <div class="fg fhide" id="f-cat">
          <label class="fl">Category</label>
          <select id="catGuess">
            <option value="Billing">Billing</option>
            <option value="Technical">Technical</option>
            <option value="Refund_Request">Refund_Request</option>
          </select>
        </div>
        <div class="fg fhide" id="f-search">
          <label class="fl">Search Query</label>
          <input type="text" id="searchQ" placeholder="e.g. billing receipt">
        </div>
        <div class="fg fhide" id="f-msg">
          <label class="fl">Message</label>
          <input type="text" id="msgInput" placeholder="Type your message…">
        </div>

        <button class="btn-submit" id="submitBtn" onclick="submitAction()">→ Submit Action</button>
        <div class="btn-row">
          <button class="bsm new" onclick="resetEnv()">⟳ New Episode</button>
          <button class="bsm" onclick="refreshState()">↺ Refresh</button>
        </div>
      </div>
    </div>

    <div class="pg-right">
      <div class="convo-area" id="convoWrap">
        <div class="convo-empty" id="convoEmpty">
          <div class="ce-emoji">🎧</div>
          <div class="ce-title">No Ticket Yet</div>
          <div class="ce-sub">Hit "New Episode" to load a customer scenario and start playing.</div>
        </div>
      </div>
      <div class="rlog">
        <div class="rlog-head">
          <span class="rlog-title">Reward Log</span>
          <span class="rtotal" id="rtotal">+0.00</span>
        </div>
        <div class="history-log" id="historyLog"></div>
      </div>
    </div>
  </div>
</section>

<section class="setup-section" id="setup">
  <div class="section-header">
    <div class="section-eyebrow">Get Running</div>
    <div class="section-title">SETUP IN<br>60 SECONDS.</div>
  </div>
  <div class="setup-grid">
    <div class="code-card">
      <div class="code-head">
        <span class="code-head-title">Install & Run</span>
        <button class="copy-btn" onclick="copyCode(this,'install')">Copy</button>
      </div>
      <div class="code-body" id="install">
        <span class="c-cmt"># Install dependencies</span><br>
        <span class="c-cmd">pip install</span> fastapi uvicorn openai pydantic<br><br>
        <span class="c-cmt"># Start the server</span><br>
        <span class="c-cmd">uvicorn</span> api:app --reload
      </div>
    </div>
    <div class="code-card">
      <div class="code-head">
        <span class="code-head-title">Run Inference</span>
        <button class="copy-btn" onclick="copyCode(this,'inference')">Copy</button>
      </div>
      <div class="code-body" id="inference">
        <span class="c-cmd">set</span> <span class="c-val">HF_TOKEN</span>=your_token<br>
        <span class="c-cmd">set</span> <span class="c-val">API_BASE_URL</span>=https://router.huggingface.co/v1<br>
        <span class="c-cmd">set</span> <span class="c-val">MODEL_NAME</span>=Qwen/Qwen2.5-72B-Instruct<br><br>
        <span class="c-cmd">python</span> inference.py
      </div>
    </div>
  </div>
</section>

<footer>
  <div class="foot-logo">Customer Support Openenv</div>
  <div class="foot-text">Built for the OpenEnv Hackathon 2024</div>
  <div class="foot-devs">Developed by <span>Hrushikesh Sarode</span> & <span>Aditya Shinde</span></div>
  <div class="foot-links">
    <a href="/docs" target="_blank">API Docs</a>
    <a href="/reset" onclick="event.preventDefault();fetch('/reset',{method:'POST'})">Reset Env</a>
  </div>
</footer>

<div class="toast" id="toast"></div>

<script>
let totalRew=0,stepNum=0,lastHistLen=0;

function toggleFields(){
  const a=document.getElementById('actionType').value;
  ['f-cat','f-search','f-msg'].forEach(id=>document.getElementById(id).classList.add('fhide'));
  if(a==='classify_issue') document.getElementById('f-cat').classList.remove('fhide');
  if(a==='search_kb') document.getElementById('f-search').classList.remove('fhide');
  if(a==='ask_clarifying_question'||a==='resolve_ticket') document.getElementById('f-msg').classList.remove('fhide');
}

function showToast(msg,type=''){
  const t=document.getElementById('toast');
  t.textContent=msg;t.className='toast show '+type;
  setTimeout(()=>t.className='toast',2600);
}

function addMsg(sender,text,cls){
  const wrap=document.getElementById('convoWrap');
  const em=document.getElementById('convoEmpty');if(em)em.remove();
  const icons={cu:'👤',ag:'🤖',sy:'⚙'};
  const d=document.createElement('div');d.className='msg';
  d.innerHTML=`<div class="mavatar ${cls}">${icons[cls]||'•'}</div>
    <div class="mbody">
      <div class="msender ${cls}">${sender}</div>
      <div class="mtext ${cls==='sy'?'sys':''}">${text}</div>
    </div>`;
  wrap.appendChild(d);wrap.scrollTop=wrap.scrollHeight;
}

// Replaced addReward with logHistory from your prompt, fully integrated to update total points
function logHistory(actionPayload, reward, done) {
    const logDiv = document.getElementById('historyLog');
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    
    // Extract the reward value and reason beautifully
    let rewardText = reward;
    let rewardValNum = 0;

    if (typeof reward === 'object' && reward !== null) {
        rewardValNum = parseFloat(reward.value) || 0;
        if (reward.value !== undefined && reward.reason !== undefined) {
            let color = reward.value >= 0 ? '#22c55e' : '#ef4444'; 
            rewardText = `<span style="color:${color}; font-weight:bold;">Reward: ${reward.value}</span> | <span style="color:#94a3b8; font-weight:normal;">Reason: ${reward.reason}</span>`;
        } else {
            rewardText = JSON.stringify(reward); 
        }
    } else {
        rewardValNum = parseFloat(reward) || 0;
        let color = rewardValNum >= 0 ? '#22c55e' : '#ef4444'; 
        rewardText = `<span style="color:${color}; font-weight:bold;">Reward: ${rewardValNum}</span>`;
    }

    // Translate the raw JSON Action into beautiful English Badges!
    let formattedAction = "";
    if (actionPayload.action_type === 'ask_clarifying_question') {
        formattedAction = `🗣️ <strong>Asked Question:</strong> "${actionPayload.message_to_customer}"`;
    } else if (actionPayload.action_type === 'search_kb') {
        formattedAction = `🔍 <strong>Searched KB:</strong> "${actionPayload.search_query}"`;
    } else if (actionPayload.action_type === 'classify_issue') {
        formattedAction = `🗂️ <strong>Classified Issue:</strong> ${actionPayload.category_guess}`;
    } else if (actionPayload.action_type === 'resolve_ticket') {
        formattedAction = `✅ <strong>Resolved Ticket:</strong> "${actionPayload.message_to_customer}"`;
    } else if (actionPayload.action_type === 'escalate_to_human') {
        formattedAction = `🚨 <strong>Escalated to Human</strong>`;
    } else if (actionPayload.command) {
        formattedAction = `🔄 <strong>${actionPayload.command}</strong>`;
    } else {
        formattedAction = JSON.stringify(actionPayload);
    }

    entry.innerHTML = `
        <div class="log-action">${formattedAction}</div>
        <div class="log-reward">${rewardText !== null ? rewardText : '0'} ${done ? '<br><span style="color:#ef4444; font-size:12px; margin-top:6px; display:inline-block; font-weight:bold;">[EPISODE FINISHED]</span>' : ''}</div>
    `;
    logDiv.prepend(entry);

    // Update total score UI
    totalRew += rewardValNum;
    const col = totalRew >= 0 ? '#16a34a' : '#ef4444';
    const s = totalRew >= 0 ? '+' : '';
    document.getElementById('rtotal').innerHTML = `<span style="color:${col};font-weight:900">${s}${totalRew.toFixed(2)}</span>`;
}

function updateStatus(obs,done){
  if(!obs||typeof obs!=='object')return;
  document.getElementById('ticketId').textContent=obs.ticket_id||'N/A';
  const st=document.getElementById('ticketStatus');
  if(done){st.textContent='DONE';st.className='tkt-badge done';}
  else{st.textContent='ACTIVE';st.className='tkt-badge active';}
  const tier=obs.customer_tier||'—';
  const te=document.getElementById('tierVal');
  te.textContent=tier;
  te.className='tkt-v '+(tier==='VIP'?'vip':tier==='Enterprise'?'ent':'std');
  const ce=document.getElementById('catVal');
  ce.textContent=obs.issue_category||'Not classified';
  ce.style.color=obs.issue_category?'var(--black)':'#888';
  const sc=obs.step_count||0;
  document.getElementById('stepVal').textContent=`${sc} / 10`;
  const pct=Math.min((sc/10)*100,100);
  document.getElementById('progFill').style.width=pct+'%';
  document.getElementById('progPct').textContent=Math.round(pct)+'%';
  const sent=(obs.user_sentiment||'Neutral').toLowerCase();
  ['happy','neutral','frustrated','angry'].forEach(s=>{
    const el=document.getElementById('sp-'+s);
    el.className='sp';
    if(sent===s)el.classList.add('a-'+s);
  });
  if(obs.knowledge_base_result){
    document.getElementById('kbResult').textContent=obs.knowledge_base_result;
    document.getElementById('kbResult').classList.add('on');
    document.getElementById('kbPh').style.display='none';
  }
}

function renderNewLines(history){
  if(!Array.isArray(history))return;
  const newLines=history.slice(lastHistLen);
  lastHistLen=history.length;
  newLines.forEach(line=>{
    if(typeof line!=='string')return;
    if(line.startsWith('Customer:')) addMsg('Customer',line.replace('Customer: ',''),'cu');
    else if(line.startsWith('Agent:')) addMsg('Agent',line.replace('Agent: ',''),'ag');
    else addMsg('System',line,'sy');
  });
}

async function resetEnv(){
  document.getElementById('historyLog').innerHTML='';
  document.getElementById('kbResult').classList.remove('on');
  document.getElementById('kbPh').style.display='';
  totalRew=0;stepNum=0;lastHistLen=0;
  document.getElementById('rtotal').innerHTML='+0.00';
  document.getElementById('convoWrap').innerHTML=`<div class="convo-empty" id="convoEmpty">
    <div class="ce-emoji">⏳</div>
    <div class="ce-title">Loading…</div></div>`;
  try{
    const res=await fetch('/reset',{method:'POST'});
    const data=await res.json();
    const obs=data.observation||data;
    document.getElementById('convoWrap').innerHTML='';
    updateStatus(obs,false);
    lastHistLen=0;
    renderNewLines(obs.conversation_history||[]);
    
    // Log initial reset state
    logHistory({command: 'System Reset - New Episode Started'}, {value: 0, reason: 'New episode started'}, false);
    
    stepNum=0;
    showToast('Episode started — '+(obs.ticket_id||''),'ok');
    document.getElementById('playground').scrollIntoView({behavior:'smooth'});
  }catch(e){showToast('Reset failed: '+e.message,'err');}
}

async function refreshState(){
  try{
    const res=await fetch('/state');
    if(!res.ok){showToast('No active state. Start a new episode first.','err');return;}
    const data=await res.json();
    updateStatus(data.observation||data,false);
    showToast('State refreshed');
  }catch(e){showToast('Error: '+e.message,'err');}
}

async function submitAction(){
  const a=document.getElementById('actionType').value;
  const payload={action_type:a};
  if(a==='classify_issue') payload.category_guess=document.getElementById('catGuess').value;
  if(a==='search_kb') payload.search_query=document.getElementById('searchQ').value;
  if(a==='ask_clarifying_question'||a==='resolve_ticket')
    payload.message_to_customer=document.getElementById('msgInput').value;
  const btn=document.getElementById('submitBtn');
  btn.disabled=true;btn.textContent='⟳ Processing…';
  try{
    const res=await fetch('/step',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(payload)
    });
    if(!res.ok){
      const err=await res.json();
      showToast(err.detail||'Error','err');
      return;
    }
    const data=await res.json();
    const obs=data.observation||{};
    updateStatus(obs,data.done);
    renderNewLines(obs.conversation_history||[]);
    
    // Using the custom logHistory format requested
    logHistory(payload, data.reward, data.done);

    if(data.done){
      const v=typeof data.reward==='object'?data.reward.value:(data.reward||0);
      addMsg('System',`Episode finished. Reward: ${parseFloat(v).toFixed(2)}`,'sy');
      showToast(v>=0.5?'🎉 Episode complete!':'❌ Episode ended — review policy','');
    }
  }catch(e){showToast('Network error: '+e.message,'err');}
  finally{btn.disabled=false;btn.textContent='→ Submit Action';}
}

function copyCode(btn,id){
  const el=document.getElementById(id);
  const text=el.innerText;
  navigator.clipboard.writeText(text).then(()=>{
    btn.textContent='Copied!';btn.classList.add('done');
    setTimeout(()=>{btn.textContent='Copy';btn.classList.remove('done');},2000);
  });
}

toggleFields();
</script>
</body>
</html>"""