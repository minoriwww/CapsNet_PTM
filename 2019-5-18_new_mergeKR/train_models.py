import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0";
import tensorflow as tf
config=tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.8
import pandas as pd
import numpy as np
import argparse

sys.path.append("../")
from Bootstrapping_allneg_continue_multiclass_mergeKR_neg_capsnet import bootStrapping_allneg_continue_keras2

sys.path.append("../../")
from ptmdeep.DLGMethod.Extractfragment_multiclass_neg_mergeKR import extractFragforTraining
# from ptmdeep.DLGMethod.Bootstrapping_allneg_continue_multiclass_neg_mergeKR import bootStrapping_allneg_continue_keras2

def main():

    parser=argparse.ArgumentParser()
    parser.add_argument('-input',  dest='inputfile', type=str, help='training data in fasta format. Sites followed by "#" are positive sites for a specific PTM prediction.', required=True)
    parser.add_argument('-output-prefix',  dest='outputprefix', type=str, help='prefix of the output files (model and parameter files).', required=True)
    parser.add_argument('-residue-types',  dest='residues', type=str, help='Residue types that this model focus on. For multiple residues, seperate each with \',\'. \n\
    Note: all the residues specified by this parameter will be trained in one model.', required=True)
    parser.add_argument('-valinput',  dest='valfile', type=str, help='validation data in fasta format if any. It will randomly select 10 percent of samples from the training data as a validation data set, if no validation file is provided.', required=False,default=None)
    parser.add_argument('-nclass',  dest='nclass', type=int, help='number of classifiers to be trained for one time. [Default:5]', required=False, default=5)
    parser.add_argument('-window',  dest='window', type=int, help='window size: the number of amino acid of the left part or right part adjacent to a potential PTM site. 2*\'windo size\'+1 amino acid will be extracted for one protential fragment. [Default:16]', required=False, default=16)
    parser.add_argument('-maxneg',  dest='maxneg', type=int, help='maximum iterations for each classifier which controls the maximum copy number of the negative data which has the same size with the positive data. [Default: 30]', required=False, default=30)
    parser.add_argument('-nb_epoch',  dest='nb_epoch', type=int, help='number of epoches for one bootstrap step. It is invalidate, if earlystop is set.', required=False, default=None)
    parser.add_argument('-earlystop',  dest='earlystop', type=int, help='after the \'earlystop\' number of epochs with no improvement the training will be stopped for one bootstrap step. [Default: 20]', required=False, default=20)
    parser.add_argument('-inputweights',  dest='inputweights', type=str, help='Initial weights saved in a HDF5 file.', required=False, default=None)
    parser.add_argument('-checkpointweights',  dest='checkpointweights', type=str, help='Set the intermediate weights of every checkpoints in HDF5 files.', required=False, default=None)
    parser.add_argument('-transferlayer',  dest='transferlayer', type=int, help='Set the last \'transferlayer\' number of layers to be randomly initialized.', required=False, default=1)
    parser.add_argument('-focus_res',  dest='focus_res', type=str, help='residule this model focus on.', required=True)

    args = parser.parse_args()
    inputfile=args.inputfile;
    valfile=args.valfile;
    outputprefix=args.outputprefix;
    nclass=args.nclass;
    window=args.window;
    maxneg=args.maxneg;
    np_epoch2=args.nb_epoch;
    earlystop=args.earlystop;
    inputweights=args.inputweights;
    checkpointweights=args.checkpointweights;
    transferlayer=args.transferlayer
    residues=args.residues.split(",")
    focus_res = args.focus_res
    if "K" in focus_res:
       forcusnum = [0,2]
    elif "R" in focus_res:
       forcusnum = [1,3]

    outputmodel = outputprefix+str("_HDF5model")
    outputparameter = outputprefix+str("_parameters")

    codemode=0 #coding method
    model='nogradientstop' #use this model
    nb_classes=4 # 0 Kp 1 Rp 2 KN 3 RN

    try:
       output = open(outputparameter, 'w')
    except IOError:
       print('cannot write to ' + outputparameter+ "!\n")
       exit()
    else:
       output.write("%d\t%d\t%s\tgeneral\t%d\t%s\t%d" % (nclass,window,args.residues,codemode,model,nb_classes))

    # EXtractfragment.extractFragforTraining(inputfile,outputfile,16,'-',("K", "R")) #
    trainfrag=extractFragforTraining(inputfile,windows=window,empty_aa='-',focus=set(residues))
    if(valfile is not None):
        valfrag=extractFragforTraining(valfile,windows=window,empty_aa='-',focus= set(residues)).as_matrix()
    else:
        valfrag=None

    for bt in range(nclass):
        checkpointoutput=checkpointweights+"_nclass"+str(bt)
        models=bootStrapping_allneg_continue_keras2(trainfrag.as_matrix(),valfile=valfrag,
                                                  srate=1,nb_epoch1=1,nb_epoch2=np_epoch2,earlystop=earlystop,maxneg=maxneg,
                                                  outputweights=checkpointoutput,
                                                  inputweights=inputweights,
                                                  model=model,
                                                  codingMode=codemode,
                                                  nb_classes=nb_classes,
                                                  hc_res2 = forcusnum
                                                  )

        models[0].save_weights(outputmodel+'_class'+str(bt),overwrite=True)

if __name__ == "__main__":
    main()
