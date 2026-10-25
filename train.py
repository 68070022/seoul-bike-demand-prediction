import argparse, copy, json, os, random, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from models import MLPBaseline, SmallCNN

SEED = 42
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    loss_sum = correct = count = 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if training: optimizer.zero_grad(set_to_none=True)
            logits = model(x); loss = criterion(logits, y)
            if training: loss.backward(); optimizer.step()
            n = y.size(0); loss_sum += loss.item() * n
            correct += (logits.argmax(1) == y).sum().item(); count += n
    return loss_sum/count, correct/count

def fit(model, train_loader, val_loader, device, args):
    model.to(device); criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    best_loss, best_state, wait = float("inf"), copy.deepcopy(model.state_dict()), 0
    for e in range(1, args.epochs+1):
        tl, ta = epoch(model, train_loader, criterion, device, optimizer)
        vl, va = epoch(model, val_loader, criterion, device)
        for k,v in zip(hist, [tl,ta,vl,va]): hist[k].append(v)
        print(f"Epoch {e:02d}/{args.epochs} train loss={tl:.4f} acc={ta:.4f} | val loss={vl:.4f} acc={va:.4f}")
        if vl < best_loss:
            best_loss, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= args.patience: print("Early stopping"); break
    model.load_state_dict(best_state)
    return model, hist

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',default='./data'); p.add_argument('--output-dir',default='./outputs')
    p.add_argument('--epochs',type=int,default=8); p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--patience',type=int,default=3)
    args=p.parse_args(); seed_all(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:',device); os.makedirs(args.output_dir,exist_ok=True)
    tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    full=datasets.FashionMNIST(args.data_dir,train=True,download=True,transform=tf)
    tr,va=random_split(full,[50000,10000],generator=torch.Generator().manual_seed(SEED))
    trl=DataLoader(tr,batch_size=args.batch_size,shuffle=True,num_workers=2)
    val=DataLoader(va,batch_size=args.batch_size,shuffle=False,num_workers=2)
    histories={}
    for name,model in [('mlp',MLPBaseline()),('cnn',SmallCNN())]:
        print('\nTraining',name); start=time.time(); model,hist=fit(model,trl,val,device,args); histories[name]=hist
        torch.save({'model_name':name,'model_state_dict':model.state_dict(),'class_names':CLASSES,'seed':SEED,'config':vars(args),'history':hist},os.path.join(args.output_dir,name+'.pt'))
        print(f'Saved {name}.pt in {time.time()-start:.1f}s')
    with open(os.path.join(args.output_dir,'training_history.json'),'w') as f: json.dump(histories,f,indent=2)
if __name__=='__main__': main()
