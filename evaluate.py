"""Evaluate saved models on the held-out chronological test set."""
from pathlib import Path
import joblib, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model import BikeMLP
from train import load_data, ARTIFACT_DIR, RESULTS_DIR

def score(y,p):
    return {"MAE":mean_absolute_error(y,p),"RMSE":mean_squared_error(y,p)**0.5,"R2":r2_score(y,p)}

def main():
    if not (ARTIFACT_DIR/"bike_mlp.pt").exists():
        raise FileNotFoundError("Run `python train.py` first.")
    RESULTS_DIR.mkdir(exist_ok=True)
    df,date_col,target,features=load_data()
    meta=joblib.load(ARTIFACT_DIR/"metadata.joblib")
    i,j=meta["split_index"]
    X=df[features].copy(); y=df[target].astype(float)
    Xte=X.iloc[j:]; yte=y.iloc[j:].to_numpy()
    prep=joblib.load(ARTIFACT_DIR/"preprocessor.joblib")
    Xte=prep.transform(Xte).astype(np.float32)
    baseline=joblib.load(ARTIFACT_DIR/"linear_regression.joblib")
    bp=baseline.predict(Xte)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt=torch.load(ARTIFACT_DIR/"bike_mlp.pt",map_location=device)
    model=BikeMLP(ckpt["input_dim"],ckpt["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    with torch.no_grad(): mp=model(torch.tensor(Xte,dtype=torch.float32).to(device)).cpu().numpy().ravel()
    results=pd.DataFrame([{"Model":"Linear Regression (baseline)",**score(yte,bp)},
                          {"Model":"MLP (Deep Learning)",**score(yte,mp)}])
    results.to_csv(RESULTS_DIR/"test_results.csv",index=False)
    print(results.round(4).to_string(index=False))
    test=df.iloc[j:].copy().reset_index(drop=True)
    test["Actual"]=yte; test["MLP_Predicted"]=mp; test["Absolute_Error"]=np.abs(yte-mp)
    test["Baseline_Predicted"]=bp; test["Baseline_Absolute_Error"]=np.abs(yte-bp)
    cols=[c for c in [date_col,"Hour","Temperature(°C)","Humidity(%)","Seasons","Functioning Day",
       "Actual","MLP_Predicted","Absolute_Error","Baseline_Predicted","Baseline_Absolute_Error"] if c in test.columns]
    failures=test.nlargest(10,"Absolute_Error")[cols]
    failures.to_csv(RESULTS_DIR/"failure_cases.csv",index=False)
    hist_path=RESULTS_DIR/"learning_curves.csv"
    if hist_path.exists():
        h=pd.read_csv(hist_path); plt.figure(figsize=(8,5))
        plt.plot(h.epoch,h.train_mse,label="Train MSE"); plt.plot(h.epoch,h.val_mse,label="Validation MSE")
        plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("Learning Curves"); plt.legend(); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(RESULTS_DIR/"learning_curves.png",dpi=160); plt.close()
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    lo=min(yte.min(),mp.min()); hi=max(yte.max(),mp.max())
    ax[0].scatter(yte,mp,alpha=.45,s=14); ax[0].plot([lo,hi],[lo,hi],"--")
    ax[0].set_xlabel("Actual"); ax[0].set_ylabel("Predicted"); ax[0].set_title("Actual vs Predicted (MLP)")
    ax[1].hist(yte-mp,bins=35); ax[1].set_xlabel("Residual (actual - predicted)"); ax[1].set_title("Residual Distribution")
    plt.tight_layout(); plt.savefig(RESULTS_DIR/"prediction_diagnostics.png",dpi=160); plt.close()
    print(f"Evaluation files saved in {RESULTS_DIR}")

if __name__=="__main__": main()
