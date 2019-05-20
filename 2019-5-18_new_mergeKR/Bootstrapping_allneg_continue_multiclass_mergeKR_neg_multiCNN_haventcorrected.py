#Bootstrapping_allneg_continue
from multiCNN_keras2 import MultiCNN
from DProcess import convertRawToXY
import pandas as pd
import numpy as np
import keras.models as models

def bootStrapping_allneg_continue_keras2(trainfile,valfile=None,srate=0.8,nb_epoch1=3,nb_epoch2=30,earlystop=None,maxneg=None,model=0,codingMode=0,frozenlayer=1,inputweights=None,outputweights=None,forkinas=False,nb_classes=2,hw_res=None,hc_res=None,hc_res2=None): #inputfile:fragments (n*34);srate:selection rate for positive data;nclass:number of class models
  train_pos={} #0S,1T,2Y
  train_neg={} #0S,1T,2Y
  train_pos_s={}
  train_neg_s={}
  train_pos_ss={}
  train_neg_ss={}
  slength={}
  nclass={}
  trainX = pd.read_table(trainfile, sep='\t', header=None).values
  for i in range(2):
      train_pos[i]=trainX[np.where(trainX[:,0]==i)] #sp 0 tp 1 yp 2 sn 3 tn 4 yn 5  p<=2 n>2
      train_neg[i]=trainX[np.where(trainX[:,0]==i+2)] #sp 0 tp 1 yp 2 sn 3 tn 4 yn 5
      train_pos[i]=pd.DataFrame(train_pos[i])
      train_neg[i]=pd.DataFrame(train_neg[i])
      train_pos_s[i]=train_pos[i].sample(train_pos[i].shape[0]); #shuffle train pos
      train_neg_s[i]=train_neg[i].sample(train_neg[i].shape[0]); #shuffle train neg
      slength[i]=int(train_pos[i].shape[0]*srate);
      nclass[i]=int(train_neg[i].shape[0]/slength[i]);
  
  if(valfile is not None): # use all data as val
     valX = pd.read_table(valfile, sep='\t', header=None).values
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
       #nclass_n=min(nclass[hc_res2[0]],maxneg); #cannot do more than maxneg times
       nclass_n=min(max(nclass.values()),maxneg)
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
        
        sampleweights=None
        
        if( hw_res is not None):
               sampleweights=np.ones(len(train_all))
               sampleweights[np.where(train_all.as_matrix()[:,0]==hw_res)]*=sum(sampleweights[np.where(train_all.as_matrix()[:,0]!=0)])/sum(sampleweights[np.where(train_all.as_matrix()[:,0]==hw_res)])
        
        classweights=None
        if(hc_res is not None):
             classweights={0:1,1:1,2:1,3:1} #0 negative, 1 S 2 T 3 Y
             classweights[hc_res]=sum(train_all.as_matrix()[:,0]!=0)/sum(train_all.as_matrix()[:,0]==hc_res)
        
        if(hc_res2 is not None): #negative has weight!
             # classweights={0:1.0,1:1.0,2:1.0,3:1.0,4:1.0,5:1.0} #sp 0 tp 1 yp 2 sn 3 tn 4 yn 5
             classweights = { k:1.0 for k in range(nb_classes)}
             classweights[hc_res2[0]]=float(sum(train_all.as_matrix()[:,0]<2))/sum(train_all.as_matrix()[:,0]==hc_res2[0])
             classweights[hc_res2[1]]=float(sum(train_all.as_matrix()[:,0]<2))/sum(train_all.as_matrix()[:,0]==hc_res2[1])
        print(train_all.as_matrix())
        trainX1,trainY1 = convertRawToXY(train_all.as_matrix(),codingMode=codingMode) #(355340,1,33,21) after extract same size as positive (48050,1,33,21)
        #models=MultiCNN(trainX1,trainY1,valX1,valY1,nb_epoch=nb_epoch2,earlystop=earlystop,model=model,frozenlayer=frozenlayer,weights=inputweights,modelweights=modelweights,forkinas=forkinas,compiletimes=t,compilemodels=models)
        print("#"*30)
        print(trainX1.shape)
        print("#"*30)
        # print(trainX1.ix[:,0])
        if t==0:
            models=MultiCNN(trainX1,trainY1,valX1,valY1,nb_epoch=nb_epoch2,earlystop=earlystop,model=model,frozenlayer=frozenlayer,weights=inputweights,sample_weight=sampleweights,nb_classes=nb_classes,class_weight=classweights,forkinas=forkinas,compiletimes=t)
        else:
            models=MultiCNN(trainX1,trainY1,valX1,valY1,nb_epoch=nb_epoch2,earlystop=earlystop,model=model,frozenlayer=frozenlayer,weights=inputweights,sample_weight=sampleweights,nb_classes=nb_classes,class_weight=classweights,forkinas=forkinas,compiletimes=t,compilemodels=models)
        
        #modelweights=models.get_weights()
        print "modelweights assigned for "+str(I)+" and "+str(t)+"\n";
        if(outputweights is not None):
            models.save_weights(outputweights+ '_iteration'+str(t),overwrite=True)
        #print "learning rate="+str(models.optimizer.lr.get_value())+"\n";
  
  
  return models;