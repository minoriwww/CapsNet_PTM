#Bootstrapping_allneg_continue
#from multiCNN_keras2 import MultiCNN
from DProcess import convertRawToXY
import pandas as pd
import numpy as np
import keras.models as models
import sys
sys.path.append("../")
from capsulenet import Capsnet_main


def bootStrapping_allneg_continue_keras2(trainfile,valfile=None,srate=0.8,
                                         nb_epoch1=3,nb_epoch2=30,earlystop=None,
                                         maxneg=None,model=0,codingMode=0,lam_recon=0,
                                         inputweights=None,outputweights=None,nb_classes=2,
                                         hw_res=None,hc_res=None,hc_res2=None): #inputfile:fragments (n*34);srate:selection rate for positive data;nclass:number of class models
  train_pos={} #0 S/T positive;1Y positive
  train_neg={} #0 S/T negative;1Y negative
  train_pos_s={}
  train_neg_s={}
  train_pos_ss={}
  train_neg_ss={}
  slength={}
  nclass={}
  trainX = trainfile
  for i in range(len(trainX)):
      trainX[i,0]=int(trainX[i,0])


  for i in range(2):
      train_pos[i]=trainX[np.where(trainX[:,0]==i)] #sp/tp 0 yp 1 sn/tn 2 yn 3
      train_neg[i]=trainX[np.where(trainX[:,0]==i+2)]
      train_pos[i]=pd.DataFrame(train_pos[i])
      train_neg[i]=pd.DataFrame(train_neg[i])
      train_pos_s[i]=train_pos[i].sample(train_pos[i].shape[0]); #shuffle train pos
      train_neg_s[i]=train_neg[i].sample(train_neg[i].shape[0]); #shuffle train neg
      slength[i]=int(train_pos[i].shape[0]*srate);
      nclass[i]=int(train_neg[i].shape[0]/slength[i]);

  if(valfile is not None): # use all data as val
     valX = valfile
     for i in range(len(valX)):
         valX[i,0]=int(valX[i,0])

     val_all=pd.DataFrame();
     for i in range(2):
         val_pos=valX[np.where(valX[:,0]==i)]
         val_neg=valX[np.where(valX[:,0]==i+2)]
         val_pos=pd.DataFrame(val_pos)
         val_neg=pd.DataFrame(val_neg)
         val_all=pd.concat([val_all,val_pos,val_neg])

     valX1,valY1 = convertRawToXY(val_all.as_matrix(),codingMode=codingMode) #(355340,1,33,21) after extract same size as positive (48050,1,33,21)
  else:
        val_all=pd.DataFrame()
        nclass={}
        for i in range(2):
            a=int(train_pos[i].shape[0]*0.9);
            b=train_neg[i].shape[0]-int(train_pos[i].shape[0]*0.1);
            print "train pos="+str(train_pos[i].shape[0])+str('\n');
            print "train neg="+str(train_neg[i].shape[0])+str('\n');
            print " a="+str(a)+" b="+str(b)+str('\n');
            train_pos_s[i]=train_pos[i][0:a]
            train_neg_s[i]=train_neg[i][0:b];
            print "train pos s="+str(train_pos_s[i].shape[0])+str('\n');
            print "train neg s="+str(train_neg_s[i].shape[0])+str('\n');

            val_pos=train_pos[i][(a+1):];
            print "val_pos="+str(val_pos.shape[0])+str('\n');
            val_neg=train_neg[i][b+1:];
            print "val_neg="+str(val_neg.shape[0])+str('\n');
            val_all=pd.concat([val_all,val_pos,val_neg])

            slength[i]=int(train_pos_s[i].shape[0]*srate); #transfer 0.1 to val so update slength
            nclass[i]=int(train_neg_s[i].shape[0]/slength[i])

        valX1,valY1 = convertRawToXY(val_all.as_matrix(),codingMode=codingMode)

  if(maxneg is not None):
       nclass_n=min(max([nclass[0],nclass[1]]),maxneg)

  #modelweights=None;
  for I in range(nb_epoch1):
    for i in range(2):
        train_neg_s[i]=train_neg_s[i].sample(train_neg_s[i].shape[0]); #shuffle neg sample
        train_pos_ss[i]=train_pos_s[i].sample(slength[i])

    for t in range(nclass_n):
        train_all=pd.DataFrame()
        for i in range(2):
            train_neg_ss[i]=train_neg_s[i][(slength[i]*t%nclass[i]):(slength[i]*t%nclass[i]+slength[i])];
            train_all=pd.concat([train_all,train_pos_ss[i],train_neg_ss[i]])

        classweights=None
        if(hc_res2 is not None): #negative has weight! hc_res2 is [0,2] for T
             classweights = { k:1.0 for k in range(nb_classes)} #stp 0 yp 1 stn 2 yn 3
             classweights[hc_res2[0]]=float(sum(train_all.as_matrix()[:,0]<=1))/sum(train_all.as_matrix()[:,0]==hc_res2[0])
             classweights[hc_res2[1]]=float(sum(train_all.as_matrix()[:,0]<=1))/sum(train_all.as_matrix()[:,0]==hc_res2[1])

        trainX1,trainY1 = convertRawToXY(train_all.as_matrix(),codingMode=codingMode) #(355340,1,33,21) after extract same size as positive (48050,1,33,21)
        if t==0:
            models,eval_model,manipulate_model,weight_c_model,fitHistory=Capsnet_main(trainX=trainX1,trainY=trainY1,valX=valX1,valY=valY1,nb_classes=nb_classes,nb_epoch=nb_epoch2,earlystop=earlystop,weights=inputweights,compiletimes=t,lr=0.001,batch_size=1000,lam_recon=lam_recon,routings=3,class_weight=classweights,modeltype=model)
        else:
            models,eval_model,manipulate_model,weight_c_model,fitHistory=Capsnet_main(trainX=trainX1,trainY=trainY1,valX=valX1,valY=valY1,nb_classes=nb_classes,nb_epoch=nb_epoch2,earlystop=earlystop,weights=inputweights,compiletimes=t,compilemodels=(models,eval_models,manipulate_models,weight_c_models),lr=0.001,batch_size=1000,lam_recon=lam_recon,routings=3,class_weight=classweights,modeltype=model)
        #modelweights=models.get_weights()

        print "modelweights assigned for "+str(I)+" and "+str(t)+"\n";
        if(outputweights is not None):
            models.save_weights(outputweights+ '_iteration'+str(t),overwrite=True)
        #print "learning rate="+str(models.optimizer.lr.get_value())+"\n";


  return models,eval_model,manipulate_model,weight_c_model,fitHistory
