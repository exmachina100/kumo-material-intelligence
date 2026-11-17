"""Train a material-family baseline. Explicit opt-in pretrained backbone download.
Dataset: training/data/{train,val,test}/{family}/*.jpg
Needs PyTorch/TorchVision/Pillow; NOT run or trained as part of repository setup.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

def loaders(root, size, batch):
    tf_train=transforms.Compose([transforms.Resize((size,size)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    tf_eval=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    sets={s:datasets.ImageFolder(root/s,transform=tf_train if s=='train' else tf_eval) for s in ['train','val','test']}
    if any(not sets[s].samples for s in sets):raise ValueError('Each split must contain labeled images')
    labels=sets['train'].classes
    if len(labels)<2:raise ValueError('At least two material classes required')
    for s in ['val','test']:
        if sets[s].classes!=labels:raise ValueError('Class names differ between splits')
    return {s:DataLoader(ds,batch_size=batch,shuffle=s=='train',num_workers=0) for s,ds in sets.items()},labels

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=Path('training/data'))
    ap.add_argument('--out',type=Path,default=Path('training/out'))
    ap.add_argument('--epochs',type=int,default=8)
    ap.add_argument('--batch-size',type=int,default=24)
    ap.add_argument('--image-size',type=int,default=160)
    ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--pretrained',action='store_true',help='Opt in to downloading pretrained weights')
    args=ap.parse_args()
    random.seed(args.seed);torch.manual_seed(args.seed)
    loads,labels=loaders(args.data,args.image_size,args.batch_size)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    weights=models.MobileNet_V3_Small_Weights.DEFAULT if args.pretrained else None
    net=models.mobilenet_v3_small(weights=weights)
    for p in net.features.parameters():p.requires_grad=False
    net.classifier[-1]=nn.Linear(net.classifier[-1].in_features,len(labels))
    net.to(device)
    loss=nn.CrossEntropyLoss()
    opt=torch.optim.AdamW(net.classifier.parameters(),lr=args.lr)
    args.out.mkdir(parents=True,exist_ok=True)
    best=-1.
    def evaluate(loader):
        net.eval();ok=total=0;preds=[];actual=[]
        with torch.no_grad():
            for x,y in loader:
                z=net(x.to(device));p=z.argmax(1).cpu();ok+=int((p==y).sum());total+=len(y);preds.extend(p.tolist());actual.extend(y.tolist())
        return ok/max(1,total),preds,actual
    for epoch in range(args.epochs):
        net.train()
        for x,y in loads['train']:
            opt.zero_grad();v=loss(net(x.to(device)),y.to(device));v.backward();opt.step()
        val,_,_=evaluate(loads['val']);print(f'epoch={epoch+1} validation_accuracy={val:.4f}')
        if val>best:
            best=val
            torch.save(net.cpu().state_dict(),args.out/'model.pt');net.to(device)
    net.load_state_dict(torch.load(args.out/'model.pt',map_location=device,weights_only=True))
    acc,pred,truth=evaluate(loads['test'])
    from sklearn.metrics import classification_report,confusion_matrix
    report={'classes':labels,'validation_best_accuracy':best,'test_accuracy':acc,'test_report':classification_report(truth,pred,labels=list(range(len(labels))),target_names=labels,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(truth,pred,labels=list(range(len(labels)))).tolist(),'pretrained':args.pretrained,'seed':args.seed}
    (args.out/'evaluation.json').write_text(json.dumps(report,indent=2))
    (args.out/'labels.json').write_text(json.dumps(labels))
    net.cpu().eval()
    example=torch.randn(1,3,args.image_size,args.image_size)
    torch.onnx.export(net,example,str(args.out/'materialnet.onnx'),input_names=['image'],output_names=['logits'],opset_version=17,dynamic_axes={'image':{0:'batch'},'logits':{0:'batch'}})
    print('Saved checkpoint, evaluation, labels, and ONNX model to',args.out)
if __name__=='__main__':main()
