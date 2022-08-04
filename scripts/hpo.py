import pandas as pd
import torch
import numpy as np
import random
import os
from tqdm import tqdm
import itertools
import json
import argparse

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc

from vibtcr.dataset import TCRDataset, hard_split_df, get_balanced_torch_loaders
from vibtcr.mvib.mvib import MVIB
from vibtcr.mvib.mvib_trainer import TrainerMVIB

import sys
import logging
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
logger = logging.getLogger()
logger.setLevel('INFO')


metrics = [
    'AUROC',
    'Accuracy',
    'Recall',
    'Precision',
    'F1 score',
    'AUPRC'
]


def pr_auc(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    return pr_auc


def get_scores(y_true, y_prob, y_pred):
    """
    Compute a df with all classification metrics and respective scores.
    """
    scores = [
        roc_auc_score(y_true, y_prob),
        accuracy_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        f1_score(y_true, y_pred),
        pr_auc(y_true, y_prob)
    ]
    
    df = pd.DataFrame(data={'score': scores, 'metrics': metrics})
    return df


def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
   
login = os.getlogin( )
DATA_BASE = f"/home/{login}/Git/tcr-generalization/tcr-data/"
RESULTS_BASE = f"/home/{login}/Git/tcr-generalization/notebooks/notebooks.classification/results/"


def main(args):
    device = torch.device(f'cuda:{args.gpu}')

    monitor = args.monitor
    joint_posterior = "aoe"
    early_stopper_patience = 10
    lr_scheduler_param = 10
    epochs = 150
    lr = 1e-3

    # HPO
    BS = [2048, 4096, 8192]
    Z = [120, 150, 180]
    BETA = [1e-6, 1e-5]
    HEADS = [1, 3, 5]

    s = [BS, Z, BETA, HEADS]
    configs = {
        c: {
            'batch_size': c[0],
            'z_dim': c[1],
            'beta': c[2],
            'heads': c[3]
        }
        for c in list(itertools.product(*s))
    }

    scores = {}

    df = pd.read_csv(DATA_BASE+f"pre-processed/{args.dataset}")

    # only considered randomized negative samples
    if args.only_random_negs:
        df = df[df["negative.source"] != "mira"]
        df = df[df["negative.source"] != "iedb"]
        df = df[df["negative.source"] != "nettcr-2.0"]

    # we do fine-tuning on pep+CDR3b
    df = df[["antigen.epitope", "cdr3.beta", "label"]].drop_duplicates(keep=False)
    print("Samples: ", len(df))
    print("Pos: ", len(df[df.label==1]))
    print("Neg: ", len(df[df.label==1]))

    df.label = df.label.apply(lambda x: int(x))

    train_df, test_df = hard_split_df(
        df.copy(), target_col="antigen.epitope", ratio=0.2, random_state=42
    )
    
    # balance test set:
    # due to how the dataset is constructed, there might be more negatives than positives
    pos_test_df = test_df[test_df.label == 1]
    neg_test_df = test_df[test_df.label == 0]
    if len(neg_test_df) > len(pos_test_df):
        neg_test_df = neg_test_df.sample(n=len(pos_test_df), replace=False)
    test_df = pd.concat([pos_test_df, neg_test_df]).sample(frac=1, axis=1).reset_index(drop=True)
   
    for config, hyper_params in configs.items():
        set_random_seed(42)

        train_loader, val_loader, scaler = get_balanced_torch_loaders(
            train_df.copy(),
            batch_size=hyper_params['batch_size'],
            gt='label',
            pep_col='antigen.epitope',
            cdr3b_col='cdr3.beta',
            device=device,
            seed=42,
            hard_split=True,
            target_col='antigen.epitope'
        )

        model = MVIB(
            z_dim=hyper_params['z_dim'], 
            device=device, 
            joint_posterior=joint_posterior,
            heads=hyper_params['heads']).to(device)

        trainer = TrainerMVIB(
            model,
            epochs=epochs,
            lr=lr,
            beta=hyper_params['beta'],
            checkpoint_dir=".",
            mode="p+b",
            lr_scheduler_param=lr_scheduler_param
        )

        checkpoint = trainer.train(
         train_loader,
         val_loader, 
         early_stopper_patience, 
         monitor
        )
        
        model = MVIB.from_checkpoint(checkpoint, torch.device("cpu")).eval()

        ds_test = TCRDataset(
            test_df.copy(), 
            torch.device("cpu"), 
            cdr3b_col='cdr3.beta', 
            pep_col='antigen.epitope', 
            scaler=scaler,
            gt_col='label'
        )
        ds_train = train_loader.dataset
        
        # get train scores
        pred = model.classify(pep=ds_train.pep.cpu(), cdr3b=ds_train.cdr3b.cpu(), cdr3a=None)
        pred = pred.detach().numpy()
        gt = ds_train.gt.cpu().detach().numpy()
        train_auroc = roc_auc_score(gt, pred)
        train_aupr = pr_auc(gt, pred)
 
        # get test scores
        pred = model.classify(pep=ds_test.pep, cdr3b=ds_test.cdr3b, cdr3a=None)
        pred = pred.detach().numpy()
        gt = ds_test.gt.detach().numpy()
        test_auroc = roc_auc_score(gt, pred)
        test_aupr = pr_auc(gt, pred)
        
        print("Train auROC: ", train_auroc)
        print("Train auPR: ", train_aupr)
        print(f"Best val ({monitor}): ", -checkpoint['best_val_score'])
        print("Test auROC: ", test_auroc)
        print("Test auPR: ", test_aupr)

        # save to file
        scores[str(checkpoint['best_val_score'])] = {
            'config': str(config),
            'test-AUROC': test_auroc,
            'test-AUPR': test_aupr,
            'train-AUROC': train_auroc,
            'train-AUPR': train_aupr,
        }
        
        # clean-up
        del train_loader, val_loader, model, trainer, checkpoint
        torch.cuda.empty_cache()

        if args.only_random_negs:
            file_name = f'hpo.{args.dataset}.only-random-negs.txt'
        else:
            file_name = f'hpo.{args.dataset}.full.txt'

        with open(RESULTS_BASE+file_name, 'w') as file:
             file.write(json.dumps(scores, sort_keys=True, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HPO')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--monitor', type=str, default='auPR')
    parser.add_argument('--dataset', type=str, default='ds.csv')
    parser.add_argument('--only-random-negs', dest='only_random_negs', action='store_true')
    parser.set_defaults(only_random_negs=False)
    args = parser.parse_args()
    main(args)
