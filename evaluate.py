import argparse, json, os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from models import MLPBaseline, SmallCNN
CLASSES=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

def predict(model, loader, device):
    model.eval(); yt=[]; yp=[]
    with torch.no_grad():
        for x,y in loader:
            yp.extend(model(x.to(device)).argmax(1).cpu().numpy().tolist()); yt.extend(y.numpy().tolist())
    return np.array(yt),np.array(yp)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',default='./data'); p.add_argument('--checkpoint-dir',default='./outputs'); p.add_argument('--output-dir',default='./evaluation'); p.add_argument('--batch-size',type=int,default=128); a=p.parse_args()
    os.makedirs(a.output_dir,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    test=datasets.FashionMNIST(a.data_dir,train=False,download=True,transform=tf)
    loader=DataLoader(test,batch_size=a.batch_size,shuffle=False,num_workers=2); results={}
    for name,model in [('mlp',MLPBaseline()),('cnn',SmallCNN())]:
        ckpt=torch.load(os.path.join(a.checkpoint_dir,name+'.pt'),map_location=device); model.load_state_dict(ckpt['model_state_dict']); model.to(device)
        yt,yp=predict(model,loader,device); results[name]={'test_accuracy':float(accuracy_score(yt,yp)),'test_macro_f1':float(f1_score(yt,yp,average='macro'))}
        print(f"\n{name.upper()} test accuracy={results[name]['test_accuracy']:.4f} macro-F1={results[name]['test_macro_f1']:.4f}")
        print(classification_report(yt,yp,target_names=CLASSES,digits=4))
        if name=='cnn':
            cm=confusion_matrix(yt,yp); fig,ax=plt.subplots(figsize=(9,8)); ConfusionMatrixDisplay(cm,display_labels=CLASSES).plot(ax=ax,xticks_rotation=45,colorbar=False,values_format='d'); plt.tight_layout(); plt.savefig(os.path.join(a.output_dir,'cnn_confusion_matrix.png'),dpi=160); plt.close(fig)
            np.savez(os.path.join(a.output_dir,'cnn_predictions.npz'),y_true=yt,y_pred=yp,wrong_indices=np.flatnonzero(yt!=yp))
    with open(os.path.join(a.output_dir,'test_metrics.json'),'w') as f: json.dump(results,f,indent=2)
    print('\nSaved metrics and confusion matrix in',a.output_dir)
if __name__=='__main__': main()
