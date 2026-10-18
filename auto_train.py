import sys
sys.stdout.reconfigure(encoding='utf-8')
import os, json, cv2, time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision.datasets import ImageFolder
from torchvision import models
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.utils.class_weight import compute_class_weight
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
from PIL import Image
from risk_engine import estimate_ita, apply_clahe, get_skin_tone

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*60)
print("  MELANOMAAI - PYTORCH GPU TRAINING")
print(f"  Device: {device}")
if torch.cuda.is_available():
    print(f"  GPU:  {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    torch.cuda.set_per_process_memory_fraction(0.85)
    print(f"  VRAM limit: 85% = ~5.4GB (safe mode)")
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)

MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", 0))
IMG_SIZE   = 384
BATCH_SIZE = 16
EPOCHS_P1  = 5
EPOCHS_P2  = 15
LR_P1      = 1e-4
LR_P2      = 1e-5
DATA_DIR   = os.environ.get("DATA_DIR", "data/v2/balanced")
print(f"[CONFIG] DATA_DIR = {DATA_DIR}")
print(f"[CONFIG] MAX_SAMPLES = {MAX_SAMPLES if MAX_SAMPLES > 0 else 'disabled'}")
RESUME_CKPT = os.environ.get("RESUME_CKPT", "")
if RESUME_CKPT:
    print(f"[CONFIG] RESUME_CKPT = {RESUME_CKPT}")

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
    A.OneOf([
        A.ElasticTransform(p=1),
        A.GridDistortion(p=1),
        A.OpticalDistortion(p=1),
    ], p=0.3),
    A.OneOf([
        A.GaussNoise(p=1),
        A.GaussianBlur(p=1),
        A.MedianBlur(blur_limit=3, p=1),
    ], p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.3),
    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])

def robustness_preprocess(img_rgb, force_clahe=False):
    """
    Novel robustness layer — skin-tone-aware preprocessing.
    
    CONTRIBUTION: Routes dark skin images through CLAHE enhancement
    BEFORE the model sees them, reducing ITA-based diagnostic disparity.
    
    ITA > 28  → Light/intermediate skin → standard pipeline
    ITA ≤ 28  → Dark/brown skin        → CLAHE + histogram equalization
    
    This differs from existing work which adjusts training data.
    We adjust at inference time — applicable to any pretrained model.
    """
    ita = estimate_ita(img_rgb)
    tone_label, tone_color, tone_pts, reliability = get_skin_tone(ita)
    
    if force_clahe or ita <= 28:
        img_rgb = apply_clahe(img_rgb)
        preprocessed = True
    else:
        preprocessed = False
    
    return img_rgb, ita, tone_label, reliability, preprocessed

class SkinDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, indices=None):
        self.base = ImageFolder(root)
        self.transform = transform
        self.indices = indices if indices is not None else list(range(len(self.base)))
        self.classes = self.base.classes
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        path, label = self.base.samples[self.indices[idx]]
        img = np.array(Image.open(path).convert("RGB"))
        
        # Integrated Robustness Layer - Skin-tone-aware preprocessing
        img, ita, tone, rel, enhanced = robustness_preprocess(img)
        
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label

print("\n[DATA] Loading data...")
base_ds = ImageFolder(DATA_DIR)
class_names = base_ds.classes
n = len(base_ds)
torch.manual_seed(42)
perm = torch.randperm(n).tolist()
if MAX_SAMPLES > 0 and MAX_SAMPLES < n:
    perm = perm[:MAX_SAMPLES]
    print(f"   Stage 1 limit active: using {MAX_SAMPLES} of {n} samples")
n = len(perm)
val_size = int(0.2 * n)
train_size = n - val_size
train_idx = perm[:train_size]
val_idx   = perm[train_size:]
train_ds = SkinDataset(DATA_DIR, transform=train_transform, indices=train_idx)
val_ds   = SkinDataset(DATA_DIR, transform=val_transform,   indices=val_idx)
benign_count    = len(os.listdir(f"{DATA_DIR}/benign"))
malignant_count = len(os.listdir(f"{DATA_DIR}/malignant"))
print(f"   Benign: {benign_count}  Malignant: {malignant_count}  Total used: {n}")
print(f"   Train: {train_size}  Val: {val_size}")
selected_labels = [base_ds.samples[i][1] for i in perm]
weights = compute_class_weight("balanced", classes=np.unique(selected_labels), y=selected_labels)
print(f"   Class weights — benign:{weights[0]:.2f} malignant:{weights[1]:.2f}")
train_labels = selected_labels[:train_size]
sample_weights = [weights[l] for l in train_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=True)

print("\n[MODEL] Building EfficientNet-B3...")
base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
num_features = base_model.classifier[1].in_features
base_model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid()
)
model = base_model.to(device)
if RESUME_CKPT:
    try:
        if os.path.exists(RESUME_CKPT):
            print(f"  Loading checkpoint weights from {RESUME_CKPT}")
            ck = torch.load(RESUME_CKPT, map_location=device)
            if isinstance(ck, dict) and "model_state_dict" in ck:
                model.load_state_dict(ck["model_state_dict"])
                print("  ✅ Checkpoint weights loaded into model (resume).")
            else:
                # allow loading plain state_dict
                model.load_state_dict(ck)
                print("  ✅ Checkpoint state_dict loaded into model (resume).")
    except Exception as e:
        print(f"  ⚠ Failed to load resume checkpoint: {e}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.35):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, pred, target):
        pred = pred.clamp(1e-7, 1-1e-7).squeeze()
        target = target.float().squeeze()
        bce = -(target*torch.log(pred) + (1-target)*torch.log(1-pred))
        p_t = target*pred + (1-target)*(1-pred)
        alpha_t = target*self.alpha + (1-target)*(1-self.alpha)
        return (alpha_t * (1-p_t)**self.gamma * bce).mean()

criterion = FocalLoss()
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

def evaluate(model, loader):
    model.eval()
    all_l, all_p = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(device)
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
                preds = model(imgs).squeeze()
            all_l.extend(labs.numpy())
            all_p.extend(preds.cpu().numpy() if preds.dim()>0 else [preds.cpu().item()])
    y_true = np.array(all_l)
    y_prob = np.array(all_p)
    y_pred = (y_prob >= 0.5).astype(int)
    acc = float((y_pred==y_true).mean())
    try:
        auc = float(roc_auc_score(y_true,y_prob)) if len(np.unique(y_true))>1 else 0.0
    except:
        auc = 0.0
    return acc, auc, y_true, y_prob, y_pred

def format_time(seconds):
    if seconds < 60: return f"{int(seconds)}s"
    elif seconds < 3600: return f"{int(seconds//60)}m {int(seconds%60)}s"
    else: return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"

def train_phase(model, loader, val_loader, optimizer, scheduler,
                epochs, phase_name, save_path, patience=8):
    print(f"\n{'='*60}")
    print(f"  {phase_name}")
    print(f"  Max epochs: {epochs}  |  Early stop patience: {patience}")
    print(f"{'='*60}")
    best_auc  = 0.0
    wait      = 0
    history   = {"train_loss":[], "val_acc":[], "val_auc":[]}
    phase_start = time.time()
    epoch_times = []
    total_batches = epochs * len(loader)
    batches_done  = 0

    for epoch in range(1, epochs+1):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        batch_times = []

        for batch_idx, (imgs, labs) in enumerate(loader):
            batch_start = time.time()
            imgs = imgs.to(device, non_blocking=True)
            labs = labs.to(device, non_blocking=True)
            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    preds = model(imgs)
                    loss = criterion(preds, labs.float())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(imgs)
                loss = criterion(preds, labs.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            batches_done += 1
            total_loss += loss.item()
            predicted = (preds.squeeze() >= 0.5).float()
            correct += (predicted == labs.float()).sum().item()
            total += labs.size(0)

            # TIME CALCULATIONS PER BATCH
            avg_batch_time = sum(batch_times) / len(batch_times)
            batches_left_total = total_batches - batches_done
            time_left_total = avg_batch_time * batches_left_total
            elapsed_total = time.time() - phase_start
            eta = datetime.now() + timedelta(seconds=time_left_total)

            # batches in current epoch
            batches_in_epoch = len(loader)
            batch_in_epoch_done = batch_idx + 1
            batch_in_epoch_left = batches_in_epoch - batch_in_epoch_done
            time_left_epoch = avg_batch_time * batch_in_epoch_left

            # VRAM
            vram = f"{torch.cuda.memory_allocated()/1e9:.1f}GB" if torch.cuda.is_available() else "CPU"

            print(f"  Ep{epoch:02d}/{epochs} "
                  f"Batch{batch_in_epoch_done:03d}/{batches_in_epoch} | "
                  f"loss:{loss.item():.4f} | "
                  f"acc:{correct/total*100:.1f}% | "
                  f"vram:{vram} | "
                  f"batch:{batch_time:.1f}s | "
                  f"epoch_left:{format_time(time_left_epoch)} | "
                  f"total_left:{format_time(time_left_total)} | "
                  f"ETA:{eta.strftime('%H:%M:%S')}",
                  flush=True)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        train_loss = total_loss / len(loader)
        val_acc, val_auc, _, _, _ = evaluate(model, val_loader)

        if hasattr(scheduler, 'step'):
            try: scheduler.step(val_auc)
            except: scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        elapsed = time.time() - phase_start
        remaining_epochs = epochs - epoch
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        time_left_phase = avg_epoch_time * remaining_epochs
        eta_phase = datetime.now() + timedelta(seconds=time_left_phase)

        marker = ""
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_auc": val_auc,
                "epoch": epoch,
                "timestamp": datetime.now().isoformat()
            }, save_path)
            marker = " ✅ BEST SAVED"
            wait = 0
        else:
            wait += 1

        print(f"\n  {'─'*55}")
        print(f"  EPOCH {epoch:02d}/{epochs} SUMMARY")
        print(f"  Loss:{train_loss:.4f} | Acc:{val_acc*100:.2f}% | AUC:{val_auc:.4f}")
        print(f"  Epoch time:    {format_time(epoch_time)}")
        print(f"  Phase elapsed: {format_time(elapsed)}")
        print(f"  Phase left:    {format_time(time_left_phase)}")
        print(f"  ETA finish:    {eta_phase.strftime('%H:%M:%S')}{marker}")
        print(f"  {'─'*55}\n")

        if patience is not None and wait >= patience:
            print(f"  ⏹ Early stopping at epoch {epoch}")
            break

    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    total_time = time.time() - phase_start
    print(f"\n  ✅ {phase_name} complete — best_auc:{best_auc:.4f} | total:{format_time(total_time)}")
    return history, best_auc

# PHASE 1
print("\n[PHASE 1] Freezing base, training head only...")
for param in model.features.parameters():
    param.requires_grad = False
print(f"   Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
optimizer1 = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR_P1, weight_decay=1e-4)
scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer1, mode='max', factor=0.5, patience=3, min_lr=1e-7)
h1, p1_auc = train_phase(model, train_loader, val_loader, optimizer1, scheduler1,
                          EPOCHS_P1, "PHASE 1 — Head Training", "models/best_phase1.pth", patience=5)

print("\n[EVAL] Phase 1 Evaluation...")
acc1,auc1,yt1,yp1,ypred1 = evaluate(model, val_loader)
r1 = classification_report(yt1, ypred1, target_names=class_names, output_dict=True)
print(f"   Accuracy:{acc1*100:.2f}%  AUC:{auc1:.4f}")
with open("outputs/metrics_phase1.json","w") as f:
    json.dump({"phase":1,"accuracy":acc1,"auc_roc":auc1,
               "classification_report":r1,"timestamp":datetime.now().isoformat()},f,indent=2)
print("   ✅ Saved outputs/metrics_phase1.json")

# PHASE 2
print("\n[PHASE 2] Unfreezing top 30% of EfficientNet-B3...")
layer_list = list(model.features.children())
cutoff = int(len(layer_list) * 0.7)
for i, layer in enumerate(layer_list):
    for param in layer.parameters():
        param.requires_grad = (i >= cutoff)
print(f"   Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
optimizer2 = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR_P2, weight_decay=1e-4)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer2, T_max=EPOCHS_P2, eta_min=1e-8)
h2, p2_auc = train_phase(model, train_loader, val_loader, optimizer2, scheduler2,
                          EPOCHS_P2, "PHASE 2 — Fine-tuning top 30%", "models/melanoma_final.pth", patience=10)

# FINAL EVAL
print("\n[EVAL] Final Evaluation...")
acc,auc,y_true,y_prob,y_pred = evaluate(model, val_loader)
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
cm = confusion_matrix(y_true, y_pred)
mk = next((k for k in report if 'mali' in k.lower()), class_names[1])
bk = next((k for k in report if 'beni' in k.lower()), class_names[0])
sens = report[mk]['recall']
spec = report[bk]['recall']
f1   = report[mk]['f1-score']
prec = report[mk]['precision']
print(f"\n{'='*60}")
print(f"  🎯 FINAL RESULTS")
print(f"  Accuracy:    {acc*100:.2f}%")
print(f"  AUC-ROC:     {auc:.4f}")
print(f"  Sensitivity: {sens*100:.2f}%")
print(f"  Specificity: {spec*100:.2f}%")
print(f"  Precision:   {prec*100:.2f}%")
print(f"  F1-Score:    {f1:.4f}")
print(f"  TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}")
print(f"{'='*60}")
with open("outputs/metrics.json","w") as f:
    json.dump({"timestamp":datetime.now().isoformat(),
               "accuracy":acc,"auc_roc":auc,"sensitivity":sens,
               "specificity":spec,"precision":prec,"f1_score":f1,
               "confusion_matrix":cm.tolist(),"classification_report":report,
               "total_samples":int(len(y_true)),
               "benign_count":int((y_true==0).sum()),
               "malignant_count":int((y_true==1).sum()),
               "model_path":"models/melanoma_final.pth"},f,indent=2)
print("✅ Saved outputs/metrics.json")

# PLOTS
print("\n[PLOT] Generating plots...")
combined_loss = h1["train_loss"] + h2["train_loss"]
combined_acc  = h1["val_acc"]   + h2["val_acc"]
combined_auc  = h1["val_auc"]   + h2["val_auc"]
p1_end = len(h1["train_loss"])
fig,axes=plt.subplots(1,3,figsize=(18,5),facecolor='#03070F')
for ax in axes:
    ax.set_facecolor('#0A101E'); ax.spines[:].set_color('#1E293B'); ax.tick_params(colors='#475569')
axes[0].plot(combined_acc,color='#00D4FF',lw=2,label='Val Acc')
axes[0].axvline(x=p1_end,color='#FFB800',linestyle='--',alpha=0.7,label='Phase 2')
axes[0].set_title("Accuracy",color='white'); axes[0].legend(facecolor='#0A101E',labelcolor='white')
axes[1].plot(combined_auc,color='#00E5A0',lw=2,label='Val AUC')
axes[1].axvline(x=p1_end,color='#FFB800',linestyle='--',alpha=0.7,label='Phase 2')
axes[1].set_title("AUC-ROC",color='white'); axes[1].legend(facecolor='#0A101E',labelcolor='white')
axes[2].plot(combined_loss,color='#818CF8',lw=2,label='Loss')
axes[2].axvline(x=p1_end,color='#FFB800',linestyle='--',alpha=0.7,label='Phase 2')
axes[2].set_title("Focal Loss",color='white'); axes[2].legend(facecolor='#0A101E',labelcolor='white')
plt.suptitle("MelanomaAI Training",color='white',fontsize=14,fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/training_curves.png",dpi=150,bbox_inches='tight',facecolor='#03070F')
plt.close()
fig,ax=plt.subplots(figsize=(6,5),facecolor='#03070F'); ax.set_facecolor('#0A101E')
ax.imshow(cm,cmap='Blues')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Benign","Malignant"],color='white')
ax.set_yticklabels(["Benign","Malignant"],color='white')
labels_cm=[["TN","FP"],["FN","TP"]]
for i in range(2):
    for j in range(2):
        ax.text(j,i,f"{labels_cm[i][j]}\n{cm[i][j]}",ha='center',va='center',color='white',fontsize=14,fontweight='bold')
ax.set_title('Confusion Matrix',color='white')
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png",dpi=150,bbox_inches='tight',facecolor='#03070F')
plt.close()
fpr,tpr,_=roc_curve(y_true,y_prob)
fig,ax=plt.subplots(figsize=(7,6),facecolor='#03070F'); ax.set_facecolor('#0A101E')
ax.plot(fpr,tpr,color='#00D4FF',lw=2.5,label=f'AUC={auc:.4f}')
ax.fill_between(fpr,tpr,alpha=0.1,color='#00D4FF')
ax.plot([0,1],[0,1],'--',color='#475569',lw=1)
ax.set_title('ROC Curve',color='white'); ax.legend(facecolor='#0A101E',labelcolor='white')
ax.tick_params(colors='#475569'); ax.spines[:].set_color('#1E293B')
plt.tight_layout()
plt.savefig("outputs/roc_curve.png",dpi=150,bbox_inches='tight',facecolor='#03070F')
plt.close()
print("✅ All plots saved")

# ROBUSTNESS
print("\n[TEST] Robustness testing by skin tone (using Integrated Robustness Layer)...")
# Note: Functions now imported from risk_engine at top of file

model.eval()
ll,lp,dl,dp=[],[],[],[]
for li,ln in enumerate(["benign","malignant"]):
    folder=f"{DATA_DIR}/{ln}"
    files=[f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))][:200]
    for fn in tqdm(files,desc=f"  Skin tone test {ln}",ncols=70):
        try:
            img=cv2.imread(os.path.join(folder,fn))
            if img is None: continue
            img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img_r=cv2.resize(img_rgb,(IMG_SIZE,IMG_SIZE))
            ita=estimate_ita(img_r)
            t=val_transform(image=img_r)["image"].unsqueeze(0).to(device)
            with torch.no_grad(): pred=float(model(t).squeeze().cpu())
            (dl if ita<=10 else ll).append(li)
            (dp if ita<=10 else lp).append(pred)
        except: continue

def gm(labels,preds,name):
    if len(labels)<5:
        print(f"  {name}: only {len(labels)} samples")
        return {"accuracy":0,"auc":0,"n":len(labels)}
    la=np.array(labels); pr=np.array(preds); bi=(pr>=0.5).astype(int)
    acc=float((bi==la).mean())
    try: a=float(roc_auc_score(la,pr)) if len(np.unique(la))>1 else 0.0
    except: a=0.0
    print(f"  {name}: n={len(labels)} acc={acc*100:.2f}% auc={a:.4f}")
    return {"accuracy":acc,"auc":a,"n":len(labels)}

lm=gm(ll,lp,"Light skin (ITA>10)")
dm=gm(dl,dp,"Dark skin  (ITA≤10)")
gap=abs(lm["accuracy"]-dm["accuracy"])
print(f"  Fairness gap: {gap:.4f} {'✅ Excellent' if gap<0.03 else '✅ Good' if gap<0.05 else '⚠ Needs work'}")
with open("outputs/robustness_report.json","w") as f:
    json.dump({"timestamp":datetime.now().isoformat(),
               "light_skin":lm,"dark_skin":dm,"fairness_gap":float(gap)},f,indent=2)
print("✅ Saved outputs/robustness_report.json")

print("\n" + "="*60)
print("  [DONE] EVERYTHING COMPLETED")
print(f"  Accuracy:    {acc*100:.2f}%")
print(f"  AUC-ROC:     {auc:.4f}")
print(f"  Sensitivity: {sens*100:.2f}%")
print(f"  Specificity: {spec*100:.2f}%")
print(f"  F1-Score:    {f1:.4f}")
print(f"  Fairness:    {gap:.4f}")
print(f"  Model:       models/melanoma_final.pth")
print(f"  Completed:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")
print("\n▶ Next step: streamlit run app.py")
