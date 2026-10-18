import os, sys, json, argparse, math
import numpy as np
import torch, torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    roc_curve, f1_score, precision_score, recall_score,
    average_precision_score, precision_recall_curve,
    classification_report
)
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from risk_engine import estimate_ita, get_skin_tone, apply_clahe, compute_risk_score, score_predictions
DATA_DIR   = os.environ.get("DATA_DIR", "data/processed")
print(f"[CONFIG] DATA_DIR = {DATA_DIR}")
MODEL_DIR  = "models"
OUTPUT_DIR = "outputs"
GRAPH_DIR  = os.path.join(OUTPUT_DIR, "graphs")
BATCH_SIZE = 32
IMG_SIZE   = 224
SEED       = 42
CYAN="#00D4FF"; RED="#FF3B5C"; GREEN="#00E5A0"; WARN="#FFB800"
ORG="#FF6400";  PUR="#818CF8"; BG="#0A101E";   BG2="#060D1A"
MUTED="#475569";TEXT="#CBD5E1"
os.makedirs(GRAPH_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def plt_dark():
    plt.rcParams.update({
        "figure.facecolor":BG,"axes.facecolor":BG2,
        "axes.edgecolor":"#1E293B","axes.labelcolor":MUTED,
        "xtick.color":MUTED,"ytick.color":MUTED,"text.color":TEXT,
        "grid.color":"#1E293B","grid.alpha":.4,"font.family":"monospace",
        "axes.spines.top":False,"axes.spines.right":False,
        "legend.facecolor":BG,"legend.edgecolor":"#1E293B","legend.labelcolor":TEXT,
    })
def save_fig(fig, name):
    path = os.path.join(GRAPH_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"   {path}")
    return path
def build_model():
    base = models.efficientnet_b3(weights=None)
    nf   = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.Dropout(.5), nn.Linear(nf,512), nn.ReLU(),
        nn.BatchNorm1d(512), nn.Dropout(.3), nn.Linear(512,256),
        nn.ReLU(), nn.Dropout(.2), nn.Linear(256,1), nn.Sigmoid())
    return base
def load_best_model(model_path=None):
    model = build_model()
    if model_path is None:
        candidates = []
        for d in [".", MODEL_DIR]:
            for n in ["melanoma_final.pth","best_phase2.pth","best_phase1.pth"]:
                p = os.path.join(d, n)
                if os.path.exists(p): candidates.append((os.path.getsize(p), p))
        if not candidates: raise FileNotFoundError("No model checkpoint found.")
        candidates.sort(reverse=True)
        model_path = candidates[0][1]
    ck = torch.load(model_path, map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    model = model.to(device).eval()
    print(f"   Loaded: {model_path}")
    return model, model_path
class SkinDataset(Dataset):
    def __init__(self, root, transform, indices):
        self.base=ImageFolder(root); self.transform=transform; self.indices=indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        path, label = self.base.samples[self.indices[idx]]
        img = np.array(Image.open(path).convert("RGB"))
        ita = estimate_ita(cv2.resize(img,(IMG_SIZE,IMG_SIZE)))
        aug = self.transform(image=img)["image"]
        return aug, label, ita, path
def get_val_loader():
    tf = A.Compose([A.Resize(IMG_SIZE,IMG_SIZE),
        A.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225]),ToTensorV2()])
    base = ImageFolder(DATA_DIR)
    torch.manual_seed(SEED)
    perm    = torch.randperm(len(base)).tolist()
    val_idx = perm[int(0.8*len(base)):]
    return DataLoader(SkinDataset(DATA_DIR,tf,val_idx),
                      batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
def run_inference(model, loader):
    all_probs,all_labels,all_itas = [],[],[]
    print(f"  Running on {len(loader.dataset)} validation samples...")
    with torch.no_grad():
        for imgs,labs,itas,_ in loader:
            imgs  = imgs.to(device)
            preds = model(imgs).squeeze()
            if preds.dim()==0: preds=preds.unsqueeze(0)
            all_probs.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labs.numpy().tolist())
            all_itas.extend(itas.numpy().tolist())
    return np.array(all_probs), np.array(all_labels), np.array(all_itas)
def find_threshold(probs, labels):
    print(f"\n  {'Threshold':>10} | {'Accuracy':>8} | {'Sensitivity':>11} | {'Specificity':>11} | {'F1':>8}")
    print("  "+"-"*58)
    best_thresh=0.5; best_f1=0.0; results=[]
    for t in np.arange(0.20,0.71,0.05):
        preds = (probs>=t).astype(int)
        acc   = accuracy_score(labels,preds)
        cm    = confusion_matrix(labels,preds)
        if cm.shape==(2,2):
            tn,fp,fn,tp=cm.ravel()
            sens=tp/(tp+fn) if (tp+fn)>0 else 0
            spec=tn/(tn+fp) if (tn+fp)>0 else 0
        else: sens=spec=0
        f1=f1_score(labels,preds,zero_division=0)
        results.append(dict(t=round(t,2),acc=acc,sens=sens,spec=spec,f1=f1))
        marker="  BEST" if f1>best_f1 else ""
        if f1>best_f1: best_f1=f1; best_thresh=t
        print(f"  {t:>10.2f} | {acc*100:>7.2f}% | {sens*100:>10.2f}% | {spec*100:>10.2f}% | {f1:>8.4f}{marker}")
    print(f"\n   Best threshold: {best_thresh:.2f}  F1={best_f1:.4f}")
    return best_thresh, results
def plot_confusion_matrix(cm):
    plt_dark()
    fig,ax=plt.subplots(figsize=(6,5))
    tn,fp,fn,tp=cm.ravel()
    vals=np.array([[tn,fp],[fn,tp]],dtype=float)
    cell_c=[[(0,.56,.38,.15),(.98,.23,.36,.15)],[(.99,.39,0,.15),(0,.52,.94,.15)]]
    cell_b=[["#00E5A0","#FF3B5C"],["#FF6400","#00D4FF"]]
    cell_l=[["TN","FP"],["FN","TP"]]
    for i in range(2):
        for j in range(2):
            ax.add_patch(FancyBboxPatch((j-.44,i-.44),.88,.88,
                boxstyle="round,pad=0.04",facecolor=cell_c[i][j],
                edgecolor=cell_b[i][j],lw=1.8,transform=ax.transData))
            ax.text(j,i+.1,str(int(vals[i,j])),ha="center",va="center",
                    fontsize=26,fontweight="bold",color=cell_b[i][j])
            ax.text(j,i-.16,cell_l[i][j],ha="center",va="center",
                    fontsize=10,color=cell_b[i][j],alpha=.8)
    ax.set_xlim(-.62,1.62); ax.set_ylim(-.62,1.62)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred: Benign","Pred: Malignant"],fontsize=9)
    ax.set_yticklabels(["Act: Benign","Act: Malignant"],fontsize=9)
    ax.set_title("Confusion Matrix",color="white",fontsize=13,fontweight="bold",pad=14)
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_visible(False)
    prec=tp/(tp+fp) if (tp+fp)>0 else 0
    rec=tp/(tp+fn) if (tp+fn)>0 else 0
    fig.text(.5,-.04,f"Precision:{prec:.1%}  Recall:{rec:.1%}  N={int(tn+fp+fn+tp)}",
             ha="center",color=MUTED,fontsize=9)
    plt.tight_layout()
    return save_fig(fig,"confusion_matrix.png")
def plot_roc(probs,labels):
    plt_dark()
    fpr,tpr,_=roc_curve(labels,probs)
    auc=roc_auc_score(labels,probs)
    fig,ax=plt.subplots(figsize=(6,5))
    ax.fill_between(fpr,tpr,alpha=.1,color=CYAN)
    ax.plot(fpr,tpr,color=CYAN,lw=2.5,label=f"AUC={auc:.4f}")
    ax.plot([0,1],[0,1],"--",color="#1E293B",lw=1.5,label="Random")
    ax.set_xlabel("False Positive Rate",fontsize=9.5)
    ax.set_ylabel("True Positive Rate",fontsize=9.5)
    ax.set_title("ROC Curve",color="white",fontsize=13,fontweight="bold",pad=12)
    ax.legend(fontsize=8.5); ax.grid(True,alpha=.2)
    ax.set_xlim([-.01,1.01]); ax.set_ylim([-.01,1.03])
    plt.tight_layout()
    return save_fig(fig,"roc_curve.png")
def plot_precision_recall(probs,labels):
    plt_dark()
    prec_c,rec_c,_=precision_recall_curve(labels,probs)
    ap=average_precision_score(labels,probs)
    f1v=2*prec_c*rec_c/(prec_c+rec_c+1e-9)
    bi=np.argmax(f1v)
    fig,ax=plt.subplots(figsize=(6,5))
    ax.fill_between(rec_c,prec_c,alpha=.08,color=GREEN)
    ax.plot(rec_c,prec_c,color=GREEN,lw=2.5,label=f"AP={ap:.4f}")
    ax.scatter([rec_c[bi]],[prec_c[bi]],color=WARN,s=100,zorder=6,
               label=f"Best F1={f1v[bi]:.3f}")
    ax.axhline(labels.mean(),color="#1E293B",ls="--",lw=1.5,label="Baseline")
    ax.set_xlabel("Recall",fontsize=9.5); ax.set_ylabel("Precision",fontsize=9.5)
    ax.set_title("Precision-Recall Curve",color="white",fontsize=13,fontweight="bold",pad=12)
    ax.legend(fontsize=8.5); ax.grid(True,alpha=.2)
    plt.tight_layout()
    return save_fig(fig,"precision_recall.png")
def plot_threshold_analysis(results,best):
    plt_dark()
    T=np.array([r["t"] for r in results])
    ACC=np.array([r["acc"] for r in results])*100
    SENS=np.array([r["sens"] for r in results])*100
    SPEC=np.array([r["spec"] for r in results])*100
    F1=np.array([r["f1"] for r in results])
    fig,axes=plt.subplots(1,3,figsize=(15,5)); fig.patch.set_facecolor(BG)
    ax=axes[0]; ax.set_facecolor(BG2)
    ax.plot(T,ACC,"^-",color=CYAN,lw=2,ms=5,label="Accuracy",alpha=.8)
    ax.plot(T,SENS,"o-",color=RED,lw=2,ms=5,label="Sensitivity")
    ax.plot(T,SPEC,"s-",color=GREEN,lw=2,ms=5,label="Specificity")
    ax.axvline(best,color=WARN,ls="--",lw=2,label=f"Selected({best:.2f})")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score(%)")
    ax.set_title("Metrics vs Threshold",color="white",fontsize=10,fontweight="bold")
    ax.legend(fontsize=7.5); ax.set_ylim(40,102); ax.grid(True,alpha=.2)
    ax=axes[1]; ax.set_facecolor(BG2)
    ax.plot(T,F1*100,"D-",color=PUR,lw=2.5,ms=5)
    bi=np.argmax(F1)
    ax.scatter(T[bi],F1[bi]*100,color=WARN,s=130,zorder=6,
               label=f"Best F1={F1[bi]:.3f}@{T[bi]:.2f}")
    ax.axvline(best,color=WARN,ls="--",lw=2,label=f"Selected({best:.2f})")
    ax.fill_between(T,F1*100,alpha=.08,color=PUR)
    ax.set_xlabel("Threshold"); ax.set_ylabel("F1 Score(%)")
    ax.set_title("F1 vs Threshold",color="white",fontsize=10,fontweight="bold")
    ax.legend(fontsize=7.5); ax.grid(True,alpha=.2)
    ax=axes[2]; ax.set_facecolor(BG2)
    ax.axvspan(.20,.40,alpha=.07,color=RED,label="High sensitivity")
    ax.axvspan(.40,.55,alpha=.07,color=WARN,label="Balanced")
    ax.axvspan(.55,.70,alpha=.07,color=GREEN,label="High specificity")
    ax.plot(T,SENS,"o-",color=RED,lw=2,ms=4)
    ax.plot(T,SPEC,"s-",color=GREEN,lw=2,ms=4)
    ci=np.argmin(np.abs(SENS-SPEC))
    ax.scatter([T[ci]],[SENS[ci]],color=CYAN,s=120,zorder=6,
               label=f"Equal@{T[ci]:.2f}")
    ax.axvline(best,color=WARN,ls="--",lw=2)
    ax.set_xlabel("Threshold")
    ax.set_title("Clinical Decision Zones",color="white",fontsize=10,fontweight="bold")
    ax.legend(fontsize=7); ax.set_ylim(40,102); ax.grid(True,alpha=.2)
    plt.suptitle("Threshold Calibration",color="white",fontsize=13,fontweight="bold",y=1.01)
    plt.tight_layout()
    return save_fig(fig,"threshold_analysis.png")
def plot_radar(metrics_dict):
    plt_dark()
    import math
    cats=["Accuracy","AUC-ROC","Sensitivity","Specificity","F1-Score","Precision"]
    vals=[metrics_dict.get(k,0)*100 for k in
          ["accuracy","auc_roc","sensitivity","specificity","f1_score","precision"]]
    N=len(cats)
    angles=[n/float(N)*2*math.pi for n in range(N)]; angles+=angles[:1]
    vn=[v/100 for v in vals]; vn+=vn[:1]
    fig,ax=plt.subplots(figsize=(6.5,6.5),subplot_kw=dict(polar=True))
    ax.set_facecolor(BG2); fig.patch.set_facecolor(BG)
    for r in [.2,.4,.6,.8,1.0]:
        ax.plot(angles,[r]*(N+1),color="#1E293B",lw=.8)
        ax.text(0,r+.03,f"{int(r*100)}%",color=MUTED,fontsize=7,ha="center")
    ax.plot(angles,[.90]*(N+1),color=GREEN,lw=1.2,ls="--",alpha=.5,label="90% target")
    ax.fill(angles,vn,color=CYAN,alpha=.12)
    ax.plot(angles,vn,color=CYAN,lw=2.5)
    ax.scatter(angles[:-1],vn[:-1],color=CYAN,s=60,zorder=5)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats,fontsize=9,color=TEXT)
    ax.set_yticklabels([]); ax.spines["polar"].set_color("#1E293B"); ax.grid(False)
    ax.set_title("Performance Radar",color="white",fontsize=13,fontweight="bold",pad=20)
    for a,v,vval in zip(angles[:-1],vals,vn[:-1]):
        ax.text(a,vval+.1,f"{v:.1f}%",ha="center",va="center",
                fontsize=7.5,color=CYAN,fontweight="bold")
    ax.legend(loc="lower right",fontsize=8)
    plt.tight_layout()
    return save_fig(fig,"metrics_radar.png")
def plot_risk_distribution(risk_stats,probs,labels,threshold):
    plt_dark()
    scores=np.array(risk_stats["score_histogram"])
    preds=(probs>=threshold).astype(int)
    fig,axes=plt.subplots(2,2,figsize=(12,9)); fig.patch.set_facecolor(BG)
    ax=axes[0,0]; ax.set_facecolor(BG2)
    bins=np.linspace(0,100,26)
    ax.hist(scores[labels==0],bins=bins,color=GREEN,alpha=.7,label="Benign (actual)")
    ax.hist(scores[labels==1],bins=bins,color=RED,alpha=.7,label="Malignant (actual)")
    for t,c in [(32,WARN),(52,ORG),(72,RED)]:
        ax.axvline(t,color=c,ls="--",lw=1.2,alpha=.8)
    ax.set_xlabel("Risk Score"); ax.set_ylabel("Count")
    ax.set_title("Risk Score Distribution",color="white",fontsize=10,fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True,axis="y",alpha=.2)
    ax=axes[0,1]; ax.set_facecolor(BG2)
    lv=risk_stats["level_counts"]
    nz={k:v for k,v in lv.items() if v>0}
    cm_map={"CRITICAL":RED,"HIGH":ORG,"MODERATE":WARN,"LOW":GREEN}
    if nz:
        wedges,texts,autos=ax.pie(list(nz.values()),labels=list(nz.keys()),
            colors=[cm_map[k] for k in nz],autopct="%1.1f%%",startangle=90,
            wedgeprops=dict(edgecolor=BG,lw=2))
        for t in texts: t.set_color(TEXT); t.set_fontsize(9)
        for a in autos: a.set_color("white"); a.set_fontsize(8.5)
    ax.set_title("Risk Level Distribution",color="white",fontsize=10,fontweight="bold")
    fp_r=risk_stats["fp_catch_rate"]
    ax.text(.5,-.06,f"FP Catch Rate: {fp_r:.1%}",transform=ax.transAxes,
            ha="center",color=WARN,fontsize=9,fontweight="bold")
    ax=axes[1,0]; ax.set_facecolor(BG2)
    c=[RED if l==1 else GREEN for l in labels]
    ax.scatter(probs,scores,c=c,alpha=.4,s=15,edgecolors="none")
    ax.axvline(threshold,color=WARN,ls="--",lw=1.5,label=f"Threshold({threshold:.2f})")
    for t,col,lbl in [(72,RED,"Critical"),(52,ORG,"High"),(32,WARN,"Moderate")]:
        ax.axhline(t,color=col,ls=":",lw=1,alpha=.7,label=lbl)
    ax.text(.02,80,"FP Zone\n(flag for review)",color=WARN,fontsize=7.5,alpha=.9)
    ax.set_xlabel("Probability"); ax.set_ylabel("Risk Score")
    ax.set_title("Risk Score vs Probability",color="white",fontsize=10,fontweight="bold")
    ax.legend(fontsize=7.5); ax.grid(True,alpha=.2); ax.set_xlim([0,1]); ax.set_ylim([0,105])
    ax=axes[1,1]; ax.set_facecolor(BG2)
    mal_mask=preds==1
    if mal_mask.sum()>0:
        mal_scores=scores[mal_mask]; mal_labels=labels[mal_mask]
        lvc={"CRITICAL":0,"HIGH":0,"MODERATE":0,"LOW":0}
        lvf={"CRITICAL":0,"HIGH":0,"MODERATE":0,"LOW":0}
        for s,l in zip(mal_scores,mal_labels):
            lv2="CRITICAL" if s>=72 else "HIGH" if s>=52 else "MODERATE" if s>=32 else "LOW"
            if l==1: lvc[lv2]+=1
            else:    lvf[lv2]+=1
        x=np.arange(4); lbls=["CRITICAL","HIGH","MODERATE","LOW"]
        corr=[lvc[l] for l in lbls]; fpos=[lvf[l] for l in lbls]
        cols=[RED,ORG,WARN,GREEN]
        ax.bar(x,corr,.4,label="True Malignant",color=cols,alpha=.8,edgecolor=BG)
        ax.bar(x,fpos,.4,bottom=corr,label="False Positive",
               color=cols,alpha=.35,edgecolor=BG,hatch="//")
        ax.set_xticks(x); ax.set_xticklabels(lbls,fontsize=8)
        ax.set_ylabel("Count")
        ax.set_title("FP Reduction by Risk Level",color="white",fontsize=10,fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True,axis="y",alpha=.2)
    plt.suptitle("Risk Stratification Analysis",color="white",fontsize=13,fontweight="bold",y=1.01)
    plt.tight_layout()
    return save_fig(fig,"risk_distribution.png")
def plot_fairness(probs,labels,itas,threshold):
    plt_dark()
    light_mask=itas>28; dark_mask=itas<=28
    results={}
    for name,mask in [("Light Skin",light_mask),("Dark Skin",dark_mask)]:
        if mask.sum()<5: continue
        pm=(probs[mask]>=threshold).astype(int)
        acc=accuracy_score(labels[mask],pm)
        try: auc=roc_auc_score(labels[mask],probs[mask])
        except: auc=0.0
        cm_=confusion_matrix(labels[mask],pm)
        if cm_.shape==(2,2):
            tn_,fp_,fn_,tp_=cm_.ravel()
            sens=tp_/(tp_+fn_) if (tp_+fn_)>0 else 0
            spec=tn_/(tn_+fp_) if (tn_+fp_)>0 else 0
        else: sens=spec=0
        results[name]=dict(acc=acc,auc=auc,sens=sens,spec=spec,n=int(mask.sum()))
    if not results: return None
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4.5))
    cats=["Accuracy","AUC","Sensitivity","Specificity"]
    x=np.arange(len(cats)); w=.32
    colors_bar=[CYAN,PUR]; names=list(results.keys())
    ax1.set_facecolor(BG2)
    for i,(nm,col) in enumerate(zip(names,colors_bar)):
        rd=results[nm]
        vb=[rd["acc"]*100,rd["auc"]*100,rd["sens"]*100,rd["spec"]*100]
        bars=ax1.bar(x+(i-.5)*w,vb,w,label=f"{nm}(n={rd['n']})",
                     color=col,alpha=.8,edgecolor=BG,lw=1.5)
        for bar in bars:
            ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.5,
                     f"{bar.get_height():.1f}",ha="center",va="bottom",
                     fontsize=7.5,fontweight="bold",color=TEXT)
    ax1.set_xticks(x); ax1.set_xticklabels(cats,fontsize=9)
    ax1.set_ylim(0,115); ax1.set_ylabel("Score(%)")
    ax1.set_title("Performance by Skin Tone",color="white",fontsize=10,fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(True,axis="y",alpha=.2)
    gap=abs(results[names[0]]["acc"]-results[names[-1]]["acc"])*100 if len(names)>=2 else 0
    ax2.set_facecolor(BG2)
    theta=np.linspace(0,np.pi,200)
    for s,e,c in [(0,.25,GREEN),(0.25,.5,WARN),(.5,.75,ORG),(.75,1.,RED)]:
        t=np.linspace(s*np.pi,e*np.pi,50)
        ax2.plot(np.cos(t),np.sin(t),color=c,lw=8,alpha=.6)
    gap_norm=min(gap/20,1.0); needle=np.pi*(1-gap_norm)
    gap_col=GREEN if gap<5 else WARN if gap<10 else ORG if gap<15 else RED
    ax2.annotate("",xy=(np.cos(needle)*.75,np.sin(needle)*.75),xytext=(0,0),
                 arrowprops=dict(arrowstyle="->",color=gap_col,lw=3.0))
    ax2.set_xlim(-1.35,1.35); ax2.set_ylim(-.2,1.3); ax2.axis("off")
    ax2.text(0,-.1,f"{gap:.1f}%",ha="center",fontsize=26,fontweight="bold",color=gap_col)
    ax2.text(0,-.3,"Fairness Gap",ha="center",fontsize=9,color=MUTED)
    ax2.set_title("Fairness Gauge",color="white",fontsize=10,fontweight="bold")
    plt.suptitle("Demographic Fairness Analysis",color="white",fontsize=12,fontweight="bold",y=1.02)
    plt.tight_layout()
    return save_fig(fig,"fairness_analysis.png")
def plot_prob_histogram(probs,labels,threshold):
    plt_dark()
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.set_facecolor(BG2)
    bins=np.linspace(0,1,41)
    ax.hist(probs[labels==0],bins=bins,color=GREEN,alpha=.75,label="Benign",edgecolor=BG)
    ax.hist(probs[labels==1],bins=bins,color=RED,alpha=.75,label="Malignant",edgecolor=BG)
    ax.axvline(threshold,color=WARN,ls="--",lw=2.5,label=f"Threshold({threshold:.2f})")
    ax.set_xlabel("Model Probability",fontsize=10); ax.set_ylabel("Count",fontsize=10)
    ax.set_title("Probability Distribution by True Label",color="white",fontsize=12,fontweight="bold")
    ax.legend(fontsize=8.5); ax.grid(True,axis="y",alpha=.2); ax.set_xlim([0,1])
    plt.tight_layout()
    return save_fig(fig,"probability_histogram.png")
def run_evaluation(model_path=None, threshold=None):
    print("\n"+"="*60)
    print("  MelanomaAI  Full Model Evaluation")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print("\n[1/6] Loading model...")
    model, model_path = load_best_model(model_path)
    print("\n[2/6] Loading validation data...")
    loader = get_val_loader()
    probs, labels, itas = run_inference(model, loader)
    print(f"  Benign:{(labels==0).sum()}  Malignant:{(labels==1).sum()}")
    print("\n[3/6] Finding optimal threshold...")
    best_thresh, thresh_results = find_threshold(probs, labels)
    if threshold is None: threshold = best_thresh
    with open(os.path.join(OUTPUT_DIR,"optimal_threshold.json"),"w") as f:
        json.dump({"optimal_threshold":float(threshold),"best_f1_threshold":float(best_thresh)},f,indent=2)
    print("\n[4/6] Computing metrics...")
    preds=(probs>=threshold).astype(int)
    cm=confusion_matrix(labels,preds)
    tn,fp,fn,tp=cm.ravel()
    acc=accuracy_score(labels,preds)
    auc=roc_auc_score(labels,probs)
    sens=tp/(tp+fn) if (tp+fn)>0 else 0
    spec=tn/(tn+fp) if (tn+fp)>0 else 0
    f1=f1_score(labels,preds,zero_division=0)
    prec=precision_score(labels,preds,zero_division=0)
    ap=average_precision_score(labels,probs)
    metrics={
        "model_path":model_path,"threshold":float(threshold),
        "evaluated_at":datetime.now().isoformat(),
        "total_samples":int(len(labels)),
        "accuracy":float(acc),"auc_roc":float(auc),
        "sensitivity":float(sens),"specificity":float(spec),
        "f1_score":float(f1),"precision":float(prec),
        "average_precision":float(ap),
        "confusion_matrix":cm.tolist(),
        "true_negatives":int(tn),"false_positives":int(fp),
        "false_negatives":int(fn),"true_positives":int(tp),
        "class_report":classification_report(labels,preds,
            target_names=["Benign","Malignant"],output_dict=True,zero_division=0),
        "threshold_analysis":thresh_results,
    }
    print(f"  Accuracy:    {acc*100:.2f}%")
    print(f"  AUC-ROC:     {auc:.4f}")
    print(f"  Sensitivity: {sens*100:.2f}%")
    print(f"  Specificity: {spec*100:.2f}%")
    print(f"  F1-Score:    {f1:.4f}")
    print(f"  Precision:   {prec*100:.2f}%")
    print("\n[5/6] Risk stratification analysis...")
    risk_stats=score_predictions(probs,itas,threshold)
    metrics["risk_analysis"]={
        "level_counts":risk_stats["level_counts"],
        "mean_score":risk_stats["mean_score"],
        "fp_caught":risk_stats["fp_caught"],
        "fp_catch_rate":risk_stats["fp_catch_rate"],
    }
    print(f"  Risk levels: {risk_stats['level_counts']}")
    print(f"  FP caught by risk scoring: {risk_stats['fp_caught']} ({risk_stats['fp_catch_rate']:.1%})")
    print(f"  Fairness Gap: {risk_stats['fairness']['prob_gap']}")
    
    with open(os.path.join(OUTPUT_DIR,"metrics.json"),"w") as f:
        json.dump(metrics,f,indent=2)
    with open(os.path.join(OUTPUT_DIR,"risk_analysis.json"),"w") as f:
        json.dump({"threshold":float(threshold),"level_counts":risk_stats["level_counts"],
                   "mean_score":risk_stats["mean_score"],"fp_caught":risk_stats["fp_caught"],
                   "fp_catch_rate":risk_stats["fp_catch_rate"]},f,indent=2)
    with open(os.path.join(OUTPUT_DIR,"fairness_metrics.json"),"w") as f:
        json.dump(risk_stats["fairness"],f,indent=2)
    print("\n[6/6] Generating graphs...")
    plot_confusion_matrix(cm)
    plot_roc(probs,labels)
    plot_precision_recall(probs,labels)
    plot_threshold_analysis(thresh_results,threshold)
    plot_radar(metrics)
    plot_risk_distribution(risk_stats,probs,labels,threshold)
    plot_fairness(probs,labels,itas,threshold)
    plot_prob_histogram(probs,labels,threshold)
    print("\n"+"="*60)
    print(f"  Accuracy:    {acc*100:.2f}%")
    print(f"  AUC-ROC:     {auc:.4f}")
    print(f"  Sensitivity: {sens*100:.2f}%")
    print(f"  FP Caught:   {risk_stats['fp_caught']} ({risk_stats['fp_catch_rate']:.1%})")
    print(f"  Graphs saved to {GRAPH_DIR}")
    print("="*60)
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",type=str,default=None)
    parser.add_argument("--threshold",type=float,default=None)
    args=parser.parse_args()
    run_evaluation(model_path=args.model,threshold=args.threshold)
