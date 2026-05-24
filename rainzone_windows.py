
import os, sys, time, re, json, threading, logging
from tkinter import *
from PIL import ImageGrab, ImageOps
import numpy as np
try:
    from paddleocr import PaddleOCR
    HAS_PADDLEOCR = True
except Exception:
    PaddleOCR = None
    HAS_PADDLEOCR = False
from pynput import keyboard as pynput_kb
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, KeyCode, Controller as KeyboardController

APP_NAME = "RainZone Itadori OCR Test"
BG="#07070e"; PANEL="#0d0d18"; CYAN="#00cccc"; PURPLE="#cc44ff"; GREEN="#00e87a"; RED="#ff3355"; DIM="#777799"; TEXT="#e8e8ff"
def app_dir():
    base=os.environ.get("APPDATA") or os.path.expanduser("~")
    p=os.path.join(base,"RainZoneItadoriTest"); os.makedirs(p,exist_ok=True); return p
APP_DIR=app_dir(); LOG_FILE=os.path.join(APP_DIR,"itadori_ocr_test.log"); CFG_FILE=os.path.join(APP_DIR,"config.json"); DEBUG_IMAGE=os.path.join(APP_DIR,"itadori_ocr_debug.png")
logging.basicConfig(filename=LOG_FILE,level=logging.INFO,format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s")
def log(msg):
    try: logging.info(msg)
    except Exception: pass
    print(msg)
mouse=MouseController(); kb_ctrl=KeyboardController(); pressed_keys=set(); pressed_lock=threading.Lock()
itadori_active=[False]; paused=[False]; modo_z=[False]; ocr_active=[False]; ocr_region=[None]
debug_ocr=[False]; ocr_raw=[""]; ocr_accepted=[""]; ocr_state=["none"]; ocr_phrase=[""]; ocr_score=[0.0]; key1_mode=["unknown"]
KEY1_BLOW_MS=[400]; KEY2_MS=[698]; KEY3_Z_MS=[150]; KEY4_MS=[1688]
_paddle=[None]; stop_event=threading.Event(); root_ref=[None]

def save_config():
    data={"ocr_region":ocr_region[0],"ocr_active":ocr_active[0],"key1_blow_ms":KEY1_BLOW_MS[0],"key3_z_ms":KEY3_Z_MS[0]}
    try:
        with open(CFG_FILE,"w",encoding="utf-8") as f: json.dump(data,f,indent=2)
    except Exception as e: log(f"save_config erro: {e}")

def load_config():
    try:
        if not os.path.exists(CFG_FILE): return
        with open(CFG_FILE,"r",encoding="utf-8") as f: data=json.load(f)
        reg=data.get("ocr_region")
        if isinstance(reg,(list,tuple)) and len(reg)==4: ocr_region[0]=tuple(int(v) for v in reg)
        ocr_active[0]=bool(data.get("ocr_active",ocr_active[0])); KEY1_BLOW_MS[0]=int(data.get("key1_blow_ms",KEY1_BLOW_MS[0])); KEY3_Z_MS[0]=int(data.get("key3_z_ms",KEY3_Z_MS[0]))
    except Exception as e: log(f"load_config erro: {e}")

def get_pressed():
    with pressed_lock: return pressed_keys.copy()
def click_left():
    try: mouse.press(Button.left); mouse.release(Button.left)
    except Exception as e: log(f"click_left erro: {e}")
def start_thread(fn,name):
    t=threading.Thread(target=fn,daemon=True,name=name); t.start(); return t
def update_status(text,color=CYAN):
    root=root_ref[0]
    if root and root.winfo_exists():
        try: root.after(0,lambda: root.status_lbl.config(text=text,fg=color))
        except Exception: pass

def normalize_text(txt):
    txt=(txt or "").strip().lower()
    txt=(txt.replace("0","o").replace("1","i").replace("|","l").replace("!","i").replace("@","a").replace("5","s").replace("$","s").replace("8","b").replace("manji","maji"))
    txt=re.sub(r"[^a-z\s]"," ",txt); return re.sub(r"\s+"," ",txt).strip()
def similarity(a,b):
    try:
        import difflib; return difflib.SequenceMatcher(None,a or "",b or "").ratio()
    except Exception: return 0.0
def fuzzy_has(txt,word,ratio=0.70):
    txt=normalize_text(txt)
    if word in txt: return True
    for token in re.findall(r"[a-z]+",txt):
        if similarity(token,word)>=ratio: return True
    return False
def state_score(txt):
    txt=normalize_text(txt); targets={"maji kick":"kick","maji block":"block","divergent blow":"blow","divergent fist":"fist","kick":"kick","block":"block","blow":"blow","fist":"fist"}
    best_state=None; best_score=0.0; best_phrase=""
    for phrase,state in targets.items():
        toks=phrase.split(); hits=sum(1 for tok in toks if fuzzy_has(txt,tok,0.70)); ratio=similarity(txt,phrase); score=ratio+hits*0.40
        if len(toks)>=2 and hits>=2: score+=0.60
        if len(toks)==1 and hits>=1: score+=0.80
        if len(txt.replace(" ",""))>22 and hits<=1: score-=0.25
        if score>best_score: best_state=state; best_score=score; best_phrase=phrase
    return best_state,best_score,best_phrase

def get_paddle():
    if not HAS_PADDLEOCR: return None
    if _paddle[0] is not None: return _paddle[0]
    try:
        try: _paddle[0]=PaddleOCR(use_angle_cls=False,lang="en",use_gpu=False,show_log=False)
        except TypeError:
            try: _paddle[0]=PaddleOCR(use_angle_cls=False,lang="en")
            except TypeError: _paddle[0]=PaddleOCR(lang="en")
        log("PaddleOCR carregado."); return _paddle[0]
    except Exception as e:
        log(f"Erro carregando PaddleOCR: {e}"); return None

def extract_paddle_texts(result):
    found=[]
    def walk(obj):
        if obj is None: return
        if isinstance(obj,tuple) and len(obj)>=1 and isinstance(obj[0],str):
            conf=0.0
            if len(obj)>1:
                try: conf=float(obj[1])
                except Exception: pass
            found.append((obj[0],conf)); return
        if isinstance(obj,dict):
            texts=obj.get("rec_texts") or obj.get("texts"); scores=obj.get("rec_scores") or obj.get("scores") or []
            if isinstance(texts,list):
                for i,t in enumerate(texts):
                    conf=0.0
                    try: conf=float(scores[i]) if i<len(scores) else 0.0
                    except Exception: pass
                    found.append((str(t),conf))
            for v in obj.values(): walk(v)
            return
        if isinstance(obj,list):
            for item in obj: walk(item)
    walk(result); return found

def prepare_image_for_paddle(img):
    img=ImageOps.expand(img.convert("RGB"),border=16,fill="white")
    img=img.resize((max(1,img.width*3),max(1,img.height*3)))
    return ImageOps.autocontrast(img)

def run_paddle_ocr(img):
    ocr=get_paddle()
    if ocr is None: return "",None,0.0,""
    img=prepare_image_for_paddle(img); arr=np.array(img); candidates=[]
    try:
        try: result=ocr.ocr(arr,det=False,cls=False)
        except TypeError: result=ocr.ocr(arr,det=False)
        candidates.extend(extract_paddle_texts(result))
    except Exception as e: log(f"Paddle det=False erro: {e}")
    if not candidates and hasattr(ocr,"predict"):
        try: candidates.extend(extract_paddle_texts(ocr.predict(arr)))
        except Exception as e: log(f"Paddle predict erro: {e}")
    best_txt=""; best_state=None; best_score=-1.0; best_phrase=""; raw_parts=[]
    for raw,conf in candidates:
        if raw: raw_parts.append(f"{raw} ({conf:.2f})")
        txt=normalize_text(raw); st,sc,phr=state_score(txt)
        try: sc+=float(conf)*0.35
        except Exception: pass
        if sc>best_score: best_txt,best_state,best_score,best_phrase=txt,st,sc,phr
    ocr_raw[0]=" | ".join(raw_parts)[:240]; ocr_state[0]=best_state or "none"; ocr_score[0]=float(best_score if best_score>0 else 0.0); ocr_phrase[0]=best_phrase or ""
    if debug_ocr[0]:
        try:
            img.save(DEBUG_IMAGE); log(f"OCR DEBUG IMAGE SAVED: {DEBUG_IMAGE}"); log(f"OCR RAW: {ocr_raw[0]} TXT: {best_txt} STATE: {best_state} PHRASE: {best_phrase} SCORE: {best_score:.2f}")
        except Exception as e: log(f"debug save erro: {e}")
    if best_score<1.05: return "",None,best_score,best_phrase
    return best_txt,best_state,best_score,best_phrase

def apply_state(state,score):
    if not state: return
    if score>=1.55:
        if state=="kick": modo_z[0]=True; update_status("ITADORI — MODO Z ON / KICK",PURPLE)
        elif state=="block": modo_z[0]=False; update_status("ITADORI — MODO Z OFF / BLOCK",GREEN)
        elif state=="blow": key1_mode[0]="blow"
        elif state=="fist": key1_mode[0]="fist"

def ocr_loop():
    while not stop_event.is_set():
        try:
            if not (itadori_active[0] and ocr_active[0] and ocr_region[0] and not paused[0]): time.sleep(0.15); continue
            x1,y1,x2,y2=ocr_region[0]; img=ImageGrab.grab(bbox=(x1,y1,x2,y2)).convert("RGB")
            txt,st,sc,phr=run_paddle_ocr(img); ocr_accepted[0]=txt; apply_state(st,sc)
        except Exception as e: log(f"ocr_loop erro: {e}")
        time.sleep(0.13)

def itadori_macro_loop():
    prev=get_pressed(); k1_running=[False]; k4_running=[False]
    def key1_blow():
        if k1_running[0] or key1_mode[0]!="blow": return
        def worker():
            k1_running[0]=True
            try:
                time.sleep(max(0,KEY1_BLOW_MS[0])/1000)
                if itadori_active[0] and not paused[0]: click_left()
            finally: k1_running[0]=False
        start_thread(worker,"itadori-key1-blow")
    while not stop_event.is_set():
        try:
            if not itadori_active[0]: prev=get_pressed(); time.sleep(0.05); continue
            cur=get_pressed(); just=cur-prev
            if "g" in just: paused[0]=not paused[0]; update_status("PAUSADO" if paused[0] else "ITADORI ACTIVE",RED if paused[0] else GREEN)
            if paused[0]: prev=cur; time.sleep(0.05); continue
            if "z" in just: modo_z[0]=not modo_z[0]; update_status("ITADORI — MODO Z ON" if modo_z[0] else "ITADORI ACTIVE",PURPLE if modo_z[0] else GREEN)
            if "1" in just: key1_blow()
            if "2" in just:
                def k2():
                    time.sleep(KEY2_MS[0]/1000)
                    if itadori_active[0] and not paused[0]: click_left()
                start_thread(k2,"itadori-key2")
            if "3" in just and modo_z[0]:
                def k3():
                    time.sleep(KEY3_Z_MS[0]/1000)
                    if itadori_active[0] and not paused[0] and modo_z[0]: click_left()
                start_thread(k3,"itadori-key3-z")
            if "4" in just and not k4_running[0]:
                def k4():
                    k4_running[0]=True
                    try:
                        time.sleep(KEY4_MS[0]/1000)
                        if itadori_active[0] and not paused[0]: click_left()
                    finally: k4_running[0]=False
                start_thread(k4,"itadori-key4")
            prev=cur
        except Exception as e: log(f"macro_loop erro: {e}")
        time.sleep(0.01)

def on_press(key):
    try:
        k=key.char.lower() if hasattr(key,"char") and key.char else str(key).replace("Key.","").lower()
        with pressed_lock: pressed_keys.add(k)
    except Exception: pass
def on_release(key):
    try:
        k=key.char.lower() if hasattr(key,"char") and key.char else str(key).replace("Key.","").lower()
        with pressed_lock: pressed_keys.discard(k)
    except Exception: pass

def draw_main_button(canvas,hover=False):
    canvas.delete("all"); w,h=int(canvas["width"]),int(canvas["height"]); active=itadori_active[0]
    bg="#1e0a2e" if active else ("#141428" if hover else "#101020"); outline=PURPLE if active or hover else "#222244"; label="ITADORI  ON" if active else "ITADORI"
    canvas.create_rectangle(2,2,w-2,h-2,fill=bg,outline=outline,width=2); canvas.create_text(w//2,h//2,text=label,fill=PURPLE if active else TEXT,font=("Courier New",15,"bold"))

def open_config(root):
    if hasattr(root,"_cfg") and root._cfg.winfo_exists(): root._cfg.lift(); return
    cfg=Toplevel(root); root._cfg=cfg; cfg.title("Config — ITADORI OCR TEST"); cfg.geometry("460x590+540+170"); cfg.configure(bg=BG); cfg.resizable(False,False)
    Label(cfg,text="── ITADORI OCR TEST ──",bg=BG,fg=PURPLE,font=("Courier New",14,"bold")).pack(pady=(16,4))
    status_var=StringVar(); read_var=StringVar()
    def refresh():
        area=str(ocr_region[0]) if ocr_region[0] else "não calibrada"
        status_var.set(f"Itadori: {'ON' if itadori_active[0] else 'OFF'} | OCR: {'ON' if ocr_active[0] else 'OFF'} | Z: {'ON' if modo_z[0] else 'OFF'}\nÁrea: {area}\nKey1: {key1_mode[0]} | Debug: {'ON' if debug_ocr[0] else 'OFF'}")
        read_var.set(f"Bruto: {ocr_raw[0] or '—'}\nAceito: {ocr_accepted[0] or '—'}\nEstado: {ocr_state[0]} | Alvo: {ocr_phrase[0] or '—'} | Score: {ocr_score[0]:.2f}")
        if cfg.winfo_exists(): cfg.after(250,refresh)
    Label(cfg,textvariable=status_var,bg=BG,fg=DIM,font=("Courier New",8),justify=LEFT,wraplength=400).pack(fill=X,padx=20,pady=(0,8))
    Label(cfg,textvariable=read_var,bg=PANEL,fg=CYAN,font=("Courier New",8,"bold"),justify=LEFT,anchor="nw",wraplength=390,padx=10,pady=10).pack(fill=X,padx=20,pady=(0,10))
    def calibrate():
        box=Toplevel(root); box.title("Calibrar OCR Itadori"); box.geometry("460x120+300+300"); box.configure(bg=PURPLE); box.attributes("-topmost",True)
        try: box.attributes("-alpha",0.38)
        except Exception: pass
        Label(box,text="Arraste em cima do nome do ataque e confirme",bg=PURPLE,fg="#000000",font=("Courier New",9,"bold")).pack(pady=(8,2))
        drag={"x":0,"y":0}
        box.bind("<Button-1>",lambda e: drag.update({"x":e.x,"y":e.y}))
        box.bind("<B1-Motion>",lambda e: box.geometry(f"+{e.x_root-drag['x']}+{e.y_root-drag['y']}"))
        def confirm():
            x,y=box.winfo_rootx(),box.winfo_rooty(); w,h=box.winfo_width(),box.winfo_height(); ocr_region[0]=(x,y,x+w,y+h); save_config(); update_status("OCR CALIBRADO",PURPLE); box.destroy()
        Button(box,text="✓ CONFIRMAR",command=confirm,bg=BG,fg=PURPLE,font=("Courier New",8,"bold"),relief=FLAT,cursor="hand2").pack(pady=6)
    def toggle_ocr(): ocr_active[0]=not ocr_active[0]; save_config(); update_status("OCR ON" if ocr_active[0] else "OCR OFF",CYAN)
    def toggle_debug(): debug_ocr[0]=not debug_ocr[0]; update_status("DEBUG OCR ON" if debug_ocr[0] else "DEBUG OCR OFF",PURPLE)
    Button(cfg,text="CALIBRAR OCR",command=calibrate,bg="#1e0a2e",fg=PURPLE,font=("Courier New",9,"bold"),relief=FLAT,cursor="hand2",height=2).pack(fill=X,padx=30,pady=4)
    Button(cfg,text="OCR ON/OFF",command=toggle_ocr,bg="#001a1a",fg=CYAN,font=("Courier New",9,"bold"),relief=FLAT,cursor="hand2",height=2).pack(fill=X,padx=30,pady=4)
    Button(cfg,text="DEBUG OCR ON/OFF",command=toggle_debug,bg="#1a1a30",fg=PURPLE,font=("Courier New",9,"bold"),relief=FLAT,cursor="hand2",height=2).pack(fill=X,padx=30,pady=4)
    def slider(parent,title,ref,start,end):
        row=Frame(parent,bg=BG); row.pack(fill=X,padx=28,pady=(8,0)); lbl=Label(row,text=f"{title}: {ref[0]}",bg=BG,fg=TEXT,font=("Courier New",8,"bold")); lbl.pack(anchor="w")
        def change(v): ref[0]=int(float(v)); lbl.config(text=f"{title}: {ref[0]}"); save_config()
        sc=Scale(row,from_=start,to=end,orient=HORIZONTAL,bg=BG,fg=PURPLE,troughcolor="#1a1a2a",highlightthickness=0,bd=0,command=change); sc.pack(fill=X); sc.set(ref[0])
    slider(cfg,"KEY 1 BLOW DELAY",KEY1_BLOW_MS,0,1000); slider(cfg,"KEY 3 Z DELAY",KEY3_Z_MS,10,1000)
    Label(cfg,text="Fixos: KEY 2 = 698ms | KEY 4 = 1688ms\nG pausa | Z alterna modo Z manual",bg=BG,fg=DIM,font=("Courier New",8),justify=CENTER).pack(pady=(10,8))
    Label(cfg,text=f"Debug image: {DEBUG_IMAGE}\nLog: {LOG_FILE}",bg=BG,fg="#555577",font=("Courier New",7),wraplength=400,justify=CENTER).pack(pady=(0,8))
    Button(cfg,text="FECHAR",command=cfg.destroy,bg="#220008",fg=RED,font=("Courier New",9,"bold"),relief=FLAT,cursor="hand2",width=20).pack(pady=(0,10)); refresh()

def main():
    load_config(); root=Tk(); root_ref[0]=root; root.title(APP_NAME); root.geometry("430x310+560+260"); root.configure(bg=BG); root.resizable(False,False)
    Label(root,text="RAINZONE",bg=BG,fg=TEXT,font=("Courier New",22,"bold")).pack(pady=(28,0)); Label(root,text="itadori only • paddleocr test • sem key • sem assets",bg=BG,fg=DIM,font=("Courier New",8)).pack(pady=(2,18))
    btn=Canvas(root,width=310,height=60,bg=BG,highlightthickness=0,cursor="hand2"); btn.pack(pady=(0,12))
    def upd(hover=False): draw_main_button(btn,hover)
    def toggle(_=None): itadori_active[0]=not itadori_active[0]; update_status("ITADORI ACTIVE" if itadori_active[0] else "ITADORI OFF",GREEN if itadori_active[0] else DIM); upd()
    btn.bind("<ButtonRelease-1>",toggle); btn.bind("<Enter>",lambda e: upd(True)); btn.bind("<Leave>",lambda e: upd(False)); upd()
    Button(root,text="CONFIG ITADORI",command=lambda:open_config(root),bg="#1e0a2e",fg=PURPLE,font=("Courier New",10,"bold"),relief=FLAT,cursor="hand2",width=24,height=2).pack(pady=(0,12))
    root.status_lbl=Label(root,text="ITADORI OFF",bg=BG,fg=DIM,font=("Courier New",9,"bold")); root.status_lbl.pack(pady=(0,8))
    Label(root,text="Key 1: Blow pelo OCR | Key 3: só no modo Z | Key 2/4 fixos",bg=BG,fg="#555577",font=("Courier New",8),wraplength=370,justify=CENTER).pack()
    listener=pynput_kb.Listener(on_press=on_press,on_release=on_release); listener.daemon=True; listener.start()
    start_thread(ocr_loop,"ocr-loop"); start_thread(itadori_macro_loop,"itadori-macro-loop")
    def close(): stop_event.set(); save_config(); root.destroy()
    root.protocol("WM_DELETE_WINDOW",close); root.mainloop()
if __name__=="__main__": main()
