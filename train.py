"""Train the MLP and Linear Regression baseline."""
from pathlib import Path
import random, urllib.request, zipfile
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from model import BikeMLP

SEED, BATCH_SIZE, MAX_EPOCHS, PATIENCE = 42, 128, 150, 15
LEARNING_RATE, DROPOUT, WEIGHT_DECAY = 1e-3, 0.20, 1e-4
ROOT = Path(__file__).resolve().parent
DATA_DIR, ARTIFACT_DIR, RESULTS_DIR = ROOT/"data", ROOT/"artifacts", ROOT/"results"
ZIP_PATH = DATA_DIR/"seoul_bike.zip"
DATA_URL = "https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip"
FEATURES = ["Hour","Temperature(°C)","Humidity(%)","Wind speed (m/s)","Visibility (10m)",
"Dew point temperature(°C)","Solar Radiation (MJ/m2)","Rainfall(mm)","Snowfall (cm)",
"Seasons","Holiday","Functioning Day","Month","DayOfWeek","DayOfYear"]
CATS = ["Seasons","Holiday","Functioning Day"]

def seed_everything():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def load_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print("Downloading UCI dataset...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH) as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csvs: raise FileNotFoundError("No CSV in UCI archive")
        name = next((n for n in csvs if "seoulbikedata" in n.lower()), csvs[0])
        with z.open(name) as f:
            try: df = pd.read_csv(f, encoding="utf-8")
            except UnicodeDecodeError:
                f.seek(0); df = pd.read_csv(f, encoding="cp949")
    df.columns = [str(c).strip() for c in df.columns]
    target = next((c for c in df.columns if c.lower()=="rented bike count"), None)
    date_col = next((c for c in df.columns if c.lower()=="date"), None)
    if target is None or date_col is None: raise ValueError(f"Unexpected columns: {df.columns.tolist()}")
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_col, target]).copy()
    df["Month"] = df[date_col].dt.month
    df["DayOfWeek"] = df[date_col].dt.dayofweek
    df["DayOfYear"] = df[date_col].dt.dayofyear
    df = df.sort_values(date_col).reset_index(drop=True)
    features = [c for c in FEATURES if c in df.columns]
    if len(features) < 5: raise ValueError(f"Too few features: {features}")
    for c in features:
        if c not in CATS: df[c] = pd.to_numeric(df[c], errors="coerce")
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=features+[target]).reset_index(drop=True)
    return df, date_col, target, features

def make_preprocessor(numeric, categorical):
    try: enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError: enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    trans = []
    if numeric: trans.append(("num", StandardScaler(), numeric))
    if categorical: trans.append(("cat", enc, categorical))
    return ColumnTransformer(trans, remainder="drop")

def loader(X, y, shuffle=False):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).reshape(-1,1))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

def loss_on(model, dl, loss_fn, device):
    model.eval(); total=0.0
    with torch.no_grad():
        for xb,yb in dl:
            xb,yb=xb.to(device),yb.to(device)
            total += loss_fn(model(xb),yb).item()*len(xb)
    return total/len(dl.dataset)

def main():
    seed_everything()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ARTIFACT_DIR.mkdir(exist_ok=True); RESULTS_DIR.mkdir(exist_ok=True)
    df,date_col,target,features=load_data()
    n=len(df); i=int(n*.70); j=int(n*.85)
    X,y=df[features].copy(),df[target].astype(float)
    Xtr,Xv,Xte=X.iloc[:i],X.iloc[i:j],X.iloc[j:]
    ytr,yv,yte=y.iloc[:i],y.iloc[i:j],y.iloc[j:]
    cats=[c for c in CATS if c in features]; nums=[c for c in features if c not in cats]
    prep=make_preprocessor(nums,cats)
    Xtr=prep.fit_transform(Xtr).astype(np.float32)
    Xv=prep.transform(Xv).astype(np.float32)
    Xte=prep.transform(Xte).astype(np.float32)
    print(f"Rows={n}; split train/val/test={len(ytr)}/{len(yv)}/{len(yte)}")
    train_dl,val_dl=loader(Xtr,ytr.to_numpy(),True),loader(Xv,yv.to_numpy())
    model=BikeMLP(Xtr.shape[1],DROPOUT).to(device)
    loss_fn=nn.MSELoss()
    opt=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    history=[]; best=float("inf"); best_state=None; waiting=0
    for epoch in range(1,MAX_EPOCHS+1):
        model.train(); total=0
        for xb,yb in train_dl:
            xb,yb=xb.to(device),yb.to(device); opt.zero_grad()
            loss=loss_fn(model(xb),yb); loss.backward(); opt.step()
            total += loss.item()*len(xb)
        tr_loss=total/len(train_dl.dataset); va_loss=loss_on(model,val_dl,loss_fn,device)
        history.append({"epoch":epoch,"train_mse":tr_loss,"val_mse":va_loss})
        if va_loss<best:
            best=va_loss; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; waiting=0
        else: waiting+=1
        if epoch==1 or epoch%10==0: print(f"Epoch {epoch}: train MSE={tr_loss:.2f}, val MSE={va_loss:.2f}")
        if waiting>=PATIENCE:
            print(f"Early stopping at epoch {epoch}"); break
    if best_state is None: raise RuntimeError("No model checkpoint saved")
    model.load_state_dict(best_state)
    baseline=LinearRegression().fit(Xtr,ytr.to_numpy())
    torch.save({"model_state_dict":model.state_dict(),"input_dim":Xtr.shape[1],"dropout":DROPOUT},ARTIFACT_DIR/"bike_mlp.pt")
    joblib.dump(prep,ARTIFACT_DIR/"preprocessor.joblib")
    joblib.dump(baseline,ARTIFACT_DIR/"linear_regression.joblib")
    joblib.dump({"features":features,"date_col":date_col,"target":target,"split_index":[i,j],
      "seed":SEED,"hyperparameters":{"batch_size":BATCH_SIZE,"max_epochs":MAX_EPOCHS,"patience":PATIENCE,
      "learning_rate":LEARNING_RATE,"dropout":DROPOUT,"weight_decay":WEIGHT_DECAY}},ARTIFACT_DIR/"metadata.joblib")
    pd.DataFrame(history).to_csv(RESULTS_DIR/"learning_curves.csv",index=False)
    print("Training complete. Run `python evaluate.py` next.")

if __name__=="__main__": main()
