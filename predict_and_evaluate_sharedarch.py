import sys
import pandas as pd
from keras.models import load_model,clone_model
import numpy as np
import argparse
import csv
from DProcess import convertRawToXY
from EXtractfragment_sort import extractFragforPredict
from capsulenet import Capsnet_main

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0";
import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.8

from keras.callbacks import Callback
import sys
sys.path.append("../")
from ptmdeep.DLGMethod.MLModels.NN_classifier import NN_classifier
import ptmdeep.DLGMethod.EXtractfragment_sort   as EXtractfragment
from ptmdeep.DLGMethod.BoostrappingTest import BstEnsembleTest

def evaluate():
    pred = model.predict(x_test)
    return np.mean(pred.argmax(axis=1) == y_test)



class Evaluate(Callback):

    def __init__(self):
        self.accs = []
        self.highest = 0.

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate()
        self.accs.append(acc)
        if acc >= self.highest:
            self.highest = acc
            model.save_weights('best_model.weights')

        print 'acc: %s, highest: %s' % (acc, self.highest)

# load model weights from file
def load_model_weights(inputweights,model_arch):
    all_weights = list()
    n_models = 0
    filename = inputweights +  str(n_models)
    while os.path.exists(filename):
      # define filename for this ensemble
      model_arch[0].load_weights(filename)
      # add to list of members
      all_weights.append(model_arch[0].get_weights())
      print('>loaded %s' % filename)
      n_models+=1
      filename = inputweights +  str(n_models)

    return all_weights

# evaluate a specific number of members in an ensemble
def polynomial_decay(decay_steps,init_value,end_value,power=1):
    weight_list = []
    weight = init_value
    for i in range(decay_steps):
        decayed_weight = (weight - end_value) * (1 - i / decay_steps) ** (power) + end_value
        weight = decayed_weight
        weight_list.append(decayed_weight)

    return weight_list

def predict_by_avg_members(members,model_arch,testX, id_iteration=None):
    n_models = len(members)
    increase_steps = int(n_models/2)
    decrease_steps = n_models-increase_steps
    temp_weight = polynomial_decay(increase_steps,1,0.5,power=1)
    temp_weight.reverse()
    #weights = temp_weight+polynomial_decay(decrease_steps,1,0.5,power=1)
    #weights = polynomial_decay(n_models,1,0.5,power=1)

    weights = [1.0 / n_models for i in range(1, n_models + 1)]

    # weights = list(np.ones(increase_steps)) + list(np.ones(decrease_steps) * 0.5)

    if id_iteration is not None:
        weights = [0.0 / n_models for i in range(1, n_models + 1)]
        weights[id_iteration] = 1.0

    # determine how many layers need to be averaged
    n_layers = len(members[0])

    # create an set of average model weights
    avg_model_weights = list()
    for layer in range(n_layers):
      # collect this layer from each model
      layer_weights = np.array([x[layer] for x in members])
      # weighted average of weights for this layer
      avg_layer_weights = np.average(layer_weights, axis=0, weights=weights)
      # store average layer weights
      avg_model_weights.append(avg_layer_weights)

    # set the weights in the new
    model_arch[0].set_weights(avg_model_weights)
    predict_proba = model_arch[1].predict(testX, verbose=0)[0]
    return predict_proba


def predict_by_snapshot(members,model_arch,testX):
    n_models = len(members)
    increase_steps = int(n_models/2)
    decrease_steps = n_models-increase_steps
    temp_weight = polynomial_decay(increase_steps,1,0.5,power=1)
    temp_weight.reverse()
    #weights = temp_weight+polynomial_decay(decrease_steps,1,0.5,power=1)
    #weights = [1.0 / n_models for i in range(1, n_models + 1)]
    weights = list(np.ones(increase_steps)) + list(np.ones(decrease_steps) * 0.5)
    #weights = polynomial_decay(n_models,1,0.5,power=1)
    #wei    ghts = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    predict_list=[]
    for member_weight in members:
        model_arch[0].set_weights(member_weight)
        predict_list.append(model_arch[1].predict(testX,verbose=0)[0])

    predict_list = np.asarray(predict_list)
    avg_predict_results = np.average(predict_list, axis=0, weights=weights)
    return avg_predict_results

def predict_by_singlemodel(members = 'weights_stackingMusitedeep_1547613804.54.h5', model_arch=None,testX=None):
    t = 0
    nclass=2
    # musitedeep = NN_classifier(output_dim=nclass, epochs=200, input_shape=testX.shape[1:], batch_size = 64, nb_classes=nclass)

    model_arch.model_fn = model_arch.Musitedeep
    model_arch.load_weights(members)
    pred = model_arch.predict(testX)

    # df = pd.DataFrame(pred)
    # df.to_csv(data_folder+folder+'/ensemble/probs/'+'stacking_musitedeep_codemode'+str(codemode)+'_'+weight_path+str(t))
    return pred

def write_output(outputfile,predict_proba,ids,poses,focuses):
    poses = poses + 1#start from 1
    results = np.column_stack((ids, poses, focuses, predict_proba[:, 1]))
    result = pd.DataFrame(results)
    result.to_csv(outputfile + ".txt", index=False, header=None, sep='\t',
                  quoting=csv.QUOTE_NONNUMERIC)

def evaluate(predict_proba,testY):
    from sklearn.metrics import roc_auc_score,average_precision_score
    true_label = [np.argmax(x) for x in testY]
    roc_score=roc_auc_score(true_label,predict_proba[:,1])
    pr_score=average_precision_score(true_label,predict_proba[:,1])
    return roc_score,pr_score


def oneshot_main():


    outputfile = r'./ensemble_omega_result_weight/All_K/'
    #fp = open(outputfile+"eval_by_AUC_precision_scores_polynomial_decay_increase_decrease_1_0.5_1",'w')
    #fp = open(outputfile+"eval_by_AUC_precision_scores_polynomial_decay_1_0.5_1",'w')
    #fp = open(outputfile+"eval_by_AUC_precision_scores_10fold",'w')
    fp = open(outputfile+"eval_by_AUC_precision_scores_10fold_constantweight1_0.5",'w')

    #model_arch=Capsnet_main(np.zeros([3,2*window+1,21]),[],nb_epoch=1,compiletimes=0,lr=0.001,batch_size=500,lam_recon=0,routings=3,modeltype=modeltype,nb_classes=nb_classes,predict=True)
    model_arch=Capsnet_main(np.zeros([3,2*16+1,21]),[],nb_epoch=1,compiletimes=0,lr=0.001,batch_size=500,lam_recon=0,routings=3,modeltype='nogradientstop',nb_classes=2,predict=True)

    roc_average_weight = np.zeros(10)
    roc_average_predict = np.zeros(10)
    roc_average_last_predict=np.zeros(10)

    pr_average_weight = np.zeros(10)
    pr_average_predict = np.zeros(10)
    pr_average_last_predict=np.zeros(10)


    for time in range(10):
        fp.write("############################"+str(time)+"\n")
        inputfile = '../ptmdeep/Newdata/All_K/metazoa_cross_testing_annotated_'+str(time)+'.fasta'

        # inputfile = '../CapsNet_PTM-master/all_PTM_raw_data/SUMOylation/metazoa_sequences_cross_testing_annotated_'+str(time)+'.fasta'


        #if os.path.exists(outputfile+"eval_by_AUC_precision_scores"):
        #   os.rm(outputfile+"eval_by_AUC_precision_scores")

        checkpointweights = outputfile+str(time)+'_weights'
        modelprefix = outputfile+str(time)+'_model'
        eval_type = 'all' # all evaluate by all method
                          # average_weight
                          # average_predict
                          # average_last_predict

        if modelprefix is None:
           # print ("Please specify the prefix for an existing custom model by "
           #        "-model-prefix!\n\
           # It indicates two files [-model-prefix]_HDF5model and [-model-prefix]_parameters.\n \
           # If you don't have such files, please run train_models.py to get the "
           #        "custom model first!\n")
           exit()
        else: #custom prediction
            parameters = []
            model=modelprefix+str("_HDF5model")
            parameter=modelprefix+str("_parameters")
            try:
                f=open(parameter,'r')
            except IOError:
                print('cannot open '+ parameter+" ! check if the model exists. "
                         "please run train_general.py or train_kinase.py to get the custom model first!\n")
            else:
                 f= open(parameter, 'r')
                 parameters=f.read()
                 f.close()

            nclass=int(parameters.split("\t")[0])
            window=int(parameters.split("\t")[1])
            residues=parameters.split("\t")[2]
            residues=residues.split(",")
            codemode=int(parameters.split("\t")[4])
            modeltype=str(parameters.split("\t")[5])
            nb_classes=int(parameters.split("\t")[6])
            print("nclass="+str(nclass)+"codemode="+str(codemode)+"\n")
            # if "N6" in inputfile:
            #     residues = ("K")
            # else:
            #     residues = ("R")
            # maxneg=2;
            # nclass=2 #20
            # is_pretrain = False

            # # model_list=[3,52] #mergeing model list
            # model_list=[3,52] #mergeing model list
            # #52 for original musitedeep with attention #9;#layers=2 inception original
            # nrate=1; #pos vs neg 1=1:1 0.1=1:10 0.2: 1:5  1:20=0.05

            # earlystop=30
            # earlystop_iter=1
            # codemode=42
            # window=16
            # nb_epoch2 =700

            # times = 1


        testfrag, ids, poses, focuses = extractFragforPredict(inputfile, window,'-', focus=residues)

        testX, testY = convertRawToXY(testfrag.as_matrix(), codingMode=codemode)
        if len(testX.shape) > 3:
            testX.shape = (testX.shape[0], testX.shape[2], testX.shape[3])


        predict_average_weight = np.zeros((testX.shape[0], 2))
        predict_average_predict = np.zeros((testX.shape[0], 2))
        predict_average_last_predict = np.zeros((testX.shape[0], 2))

        for bt in range(nclass): #0 648 bt=2 len(tf.trainable_variables())=1530
            #load all involving mode weights
            #sess = tf.Session()
            inputweights = checkpointweights+"_nclass"+str(bt)+"_iteration"
            model_members = load_model_weights(inputweights,model_arch)
            if eval_type == "all" or eval_type == "average_weight":
                predict_temp=predict_by_avg_members(model_members,model_arch,testX, id_iteration=1)
                predict_average_weight+=predict_temp
                auc_score,pr_score=evaluate(predict_temp,testY)
                fp.write("average_weight_results_bt"+str(bt)+"\t"+str(auc_score)+"\t"+str(pr_score)+"\n")

            if eval_type == "all" or eval_type == "average_predict":
                predict_temp = predict_by_snapshot(model_members,model_arch,testX)
                predict_average_predict+=predict_temp
                auc_score,pr_score=evaluate(predict_temp,testY)
                fp.write("average_predict_results_bt"+str(bt)+"\t"+str(auc_score)+"\t"+str(pr_score)+"\n")

            del model_members
            #sess.close()

        if eval_type == "all" or eval_type == "average_weight":
            predict_average_weight = predict_average_weight/float(nclass)
            auc_score,pr_score=evaluate(predict_average_weight,testY)
            fp.write("average_weight_results\t"+str(auc_score)+"\t"+str(pr_score)+"\n")
            roc_average_weight[time] = auc_score
            pr_average_weight[time] = pr_score
            #write_output(outputfile + "average_weight_results_fold"+str(time)+".txt",predict_average_weight,ids,poses,focuses)

        if eval_type == "all" or eval_type == "average_predict":
            predict_average_predict = predict_average_predict/float(nclass)
            auc_score,pr_score=evaluate(predict_average_predict,testY)
            fp.write("average_predict_results:\t"+str(auc_score)+"\t"+str(pr_score)+"\n")
            roc_average_predict[time] = auc_score
            pr_average_predict[time] = pr_score
            #write_output(outputfile + "average_predict_results_fold"+str(time)+".txt",predict_average_predict,ids,poses,focuses)

        if eval_type == "all" or eval_type == "average_last_predict":
            nclass_ini = 1
            for bt in range(nclass):
                model_arch[0].load_weights(model + "_class" + str(bt))
                predict_temp = model_arch[1].predict(testX)[0]
                predict_average_last_predict += predict_temp
                auc_score,pr_score=evaluate(predict_temp,testY)
                fp.write("average_last_predict_results_bt"+str(bt)+"\t"+str(auc_score)+"\t"+str(pr_score)+"\n")

            predict_average_last_predict = predict_average_last_predict / (nclass * nclass_ini)
            auc_score,pr_score=evaluate(predict_average_last_predict,testY)
            fp.write("average_last_predict_results\t"+str(auc_score)+"\t"+str(pr_score)+"\n")
            roc_average_last_predict[time]=auc_score
            pr_average_last_predict[time]=pr_score
            #write_output(outputfile + "average_last_predict_results_fold"+str(time)+".txt",predict_average_last_predict,ids,poses,focuses)
            print("Successfully predicted from custom models !\n")

    fp.write("!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    fp.write("average_weight_results\t"+",".join([str(x) for x in roc_average_weight])+"\t"+",".join([str(x) for x in pr_average_weight])+"\t"+str(np.mean(roc_average_weight))+","+str(np.std(roc_average_weight))+"\t"+str(np.mean(pr_average_weight))+","+str(np.std(pr_average_weight))+"\n")
    fp.write("average_predict_results\t"+",".join([str(x) for x in roc_average_predict])+"\t"+",".join([str(x) for x in pr_average_predict])+"\t"+str(np.mean(roc_average_predict))+","+str(np.std(roc_average_predict))+"\t"+str(np.mean(pr_average_predict))+","+str(np.std(pr_average_predict))+"\n")
    fp.write("average_last_predict_results\t"+",".join([str(x) for x in roc_average_last_predict])+"\t"+",".join([str(x) for x in pr_average_last_predict])+"\t"+str(np.mean(roc_average_last_predict))+","+str(np.std(roc_average_last_predict))+"\t"+str(np.mean(pr_average_last_predict))+","+str(np.std(pr_average_last_predict))+"\n")
    fp.close()

def stacking_main():
    outputfile = r'./ensemble_omega_result_weight/'
    #fp = open(outputfile+"eval_by_AUC_precision_scores_polynomial_decay_increase_decrease_1_0.5_1",'w')
    #fp = open(outputfile+"eval_by_AUC_precision_scores_polynomial_decay_1_0.5_1",'w')
    #fp = open(outputfile+"eval_by_AUC_precision_scores_10fold",'w')
    fp = open(outputfile+"eval_by_AUC_precision_scores_stacking_model",'w')

    #model_arch=Capsnet_main(np.zeros([3,2*window+1,21]),[],nb_epoch=1,compiletimes=0,lr=0.001,batch_size=500,lam_recon=0,routings=3,modeltype=modeltype,nb_classes=nb_classes,predict=True)
    roc_average_weight = np.zeros(10)
    roc_average_predict = np.zeros(10)
    roc_average_last_predict=np.zeros(10)

    pr_average_weight = np.zeros(10)
    pr_average_predict = np.zeros(10)
    pr_average_last_predict=np.zeros(10)


    for time in range(1):
        fp.write("############################"+str(time)+"\n")
        inputfile = '../ptmdeep/Newdata/Omega-N-methylarginine/Omega-N-methylarginine.fasta_testing_annotated_'+str(time)+'.fasta'



        checkpointweights = './ensemble_omega_result_weight/Omega-N-methylarginine_'+str(time)+'_weights'
        modelprefix = './ensemble_omega_result_weight/Omega-N-methylarginine_'+str(time)+'_model'
        eval_type = 'all' # all evaluate by all method
                          # average_weight
                          # average_predict
                          # average_last_predict

        if modelprefix is None:
           # print ("Please specify the prefix for an existing custom model by "
           #        "-model-prefix!\n\
           # It indicates two files [-model-prefix]_HDF5model and [-model-prefix]_parameters.\n \
           # If you don't have such files, please run train_models.py to get the "
           #        "custom model first!\n")
           exit()
        else: #custom prediction
            model=modelprefix+str("_HDF5model")
            parameter=modelprefix+str("_parameters")
            try:
                f=open(parameter,'r')
            except IOError:
                print('cannot open '+ parameter+" ! check if the model exists. "
                         "please run train_general.py or train_kinase.py to get the custom model first!\n")
            else:
                f= open(parameter, 'r')
                parameters=f.read()
                f.close()

            folder = "Omega-N-methylarginine"#"Omega-N-methylarginine_Dimethylated_arginine_Symmetric_dimethylarginine_Asymmetric_dimethylarginine"#"Omega-N-methylarginine"
            data_folder = "./Newdata/"#"./Newdata/" # R


            # delete="K";#"R";  #kp 0 Rp 1 Kn 2 Rn 3
            # deletenum=[0,2]


            #############################
            if "N6" in folder or "K" in folder:
                residues = ("K")
            else:
                residues = ("R")


            maxneg=2;
            nclass=2 #20
            is_pretrain = False

            # model_list=[3,52] #mergeing model list
            model_list=[3,52] #mergeing model list
            #52 for original musitedeep with attention #9;#layers=2 inception original
            nrate=1; #pos vs neg 1=1:1 0.1=1:10 0.2: 1:5  1:20=0.05

            earlystop=30
            earlystop_iter=1
            codemode=42
            window=16
            nb_epoch2 =700

            times = 1
            print("nclass="+str(nclass)+"codemode="+str(codemode)+"\n")

        ################## extract fragement ##################

        inputfile='../ptmdeep/Newdata/Omega-N-methylarginine/Omega-N-methylarginine.fasta_training_annotated_'+str(time)+'.fasta'
        outputfile='../ptmdeep/Newdata/Omega-N-methylarginine/Omega-N-methylarginine.fasta_training_annotated_'+str(time)+"_fragments"
        trainfile=outputfile;#
        EXtractfragment.extractFragforTraining(inputfile,outputfile,window,'-',residues)

        inputfile='../ptmdeep/Newdata/Omega-N-methylarginine/Omega-N-methylarginine.fasta_testing_annotated_'+str(time)+".fasta"
        outputfile='../ptmdeep/Newdata/Omega-N-methylarginine/Omega-N-methylarginine.fasta_testing_annotated_'+str(time)+"_fragments"
        testfile=outputfile
        EXtractfragment.extractFragforTraining(inputfile,outputfile,window,'-',residues)

        # fasta_validation_annotated_
        inputfile='../ptmdeep/Newdata/Omega-N-methylarginine/Omega-N-methylarginine.fasta_validation_annotated_'+str(time)+".fasta"
        outputfile='../ptmdeep/Newdata/Omega-N-methylarginine/Omega-N-methylarginine.fasta_validation_annotated_'+str(time)+"_fragments"
        valfile=outputfile
        EXtractfragment.extractFragforTraining(inputfile,outputfile,window,'-',residues)


        testfrag, ids, poses, focuses = extractFragforPredict(inputfile, window,'-', focus=residues)

        testX, testY = convertRawToXY(testfrag.as_matrix(), codingMode=codemode)
        if len(testX.shape) > 3:
            testX.shape = (testX.shape[0], testX.shape[2], testX.shape[3])


        predict_average_weight = np.zeros((testX.shape[0], 2))
        predict_average_predict = np.zeros((testX.shape[0], 2))
        predict_average_last_predict = np.zeros((testX.shape[0], 2))

        bstEnsembleTest = BstEnsembleTest()
        tempmodel = bstEnsembleTest.simple_model(trainfile,valfile=valfile,srate=1,nb_epoch1=1,nb_epoch2=1,earlystop=None,maxneg=maxneg,codingMode=codemode,predict=False, nb_classes=2)

        for bt in range(nclass): #0 648 bt=2 len(tf.trainable_variables())=1530
            #load all involving mode weights
            #sess = tf.Session()
            inputweights = checkpointweights+"_nclass"+str(bt)+"_iteration"
            model_members = "../ptmdeep/weights_stackingMusitedeep_1557474338.48.h5"

            predict_temp = predict_by_singlemodel(model_members,tempmodel,testX)
            predict_average_predict+=predict_temp
            auc_score,pr_score=evaluate(predict_temp,testY)
            fp.write("average_predict_results_bt"+str(bt)+"\t"+str(auc_score)+"\t"+str(pr_score)+"\n")
            print("#"*30)
            print("testing")
            print("#"*30)

            #sess.close()

        predict_average_predict = predict_average_predict/float(nclass)
        auc_score,pr_score=evaluate(predict_average_predict,testY)
        fp.write("average_predict_results:\t"+str(auc_score)+"\t"+str(  )+"\n")
        roc_average_predict[time] = auc_score
        pr_average_predict[time] = pr_score
        #write_output(outputfile + "average_predict_results_fold"+str(time)+".txt",predict_average_predict,ids,poses,focuses)

    fp.close()

if __name__ == '__main__':
    oneshot_main()
    # stacking_main()

#if __name__ == "__main__":
#    main()

