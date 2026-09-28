import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from tqdm import tqdm
from sklearn.metrics import accuracy_score
from datetime import datetime
import math
import matplotlib.pyplot as plt

from symbiotic_learning.loss import embed_sim, embed_summand, embed_loss

def reports_summary(reports: List[dict], epoch: int, tags: List[str]) -> None:
    """
    Prints the contents of a list of training/validation/testing reports with labels. 
    """

    label_lookup = {}
    label_lookup['personal'] = 'Average Personal Loss'
    label_lookup['symbiotic'] = 'Average Symbiotic Loss'
    label_lookup['accuracy'] = 'Accuracy'
    label_lookup['embedding'] = 'Average Embedding Loss'
    label_lookup['blame'] = 'Average Blame Loss'
    
    print(f'Summary:')
    for i, report in enumerate(reports):
        tag = tags[i]

        print(f'\t- {tag}:')
        for label in report.keys():
            if report[label] == None:
                continue
            else:
                if label != 'accuracy':
                    print(f'\t\t-- {label_lookup[label]}: {report[label]:.4}')
                else:
                    print()
                    print(f'\t\t-- {label_lookup[label]}: {report[label]:.4}')
        print()

    return

def fill_lt_reports(lt_reports: List[dict], reports: List[dict], phase: str) -> None:
    """
    Initializes or appends report data to lifetime reports.
    """
    
    for lt_report, report in zip(lt_reports, reports):
            keys = list(report.keys())
            for key in keys:
                if key not in lt_report[phase].keys():
                    lt_report[phase][key] = [report[key]]
                else:
                    lt_report[phase][key].append(report[key])
                    
    return
    
def train_one_epoch(
        train_loader: DataLoader, 
        models: List[Module], 
        opts: List[Optimizer], 
        scheds: List[LRScheduler], 
        collab_params: List[float], 
        temp: float,
        epoch: int, 
        criterion: Module, 
        uplift: int = 10, 
        eps: float = 1e-7, 
        lamb: float = 1.0) -> List[dict]:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    reports = [{} for model in models]
    personals = [[] for model in models]
    embs = [[] for model in models[:-1]]
    blames = [[] for model in models[:-1]]
    syms = [[] for model in models]
    
    for model in models:
        model.train().to(device)

    ## TRAINING LOOP
    pbar_train = tqdm(train_loader, total=len(train_loader))
    pbar_train.set_description(f'Epoch {epoch}: Training')
    for img, label in pbar_train:
        for opt in opts:
            opt.zero_grad()

        img = img.to(device)

        inds = [model.embed(img) for model in models[:-1]]
        
        ind_concat = torch.cat(inds, dim=1)
        ind_stack = torch.stack(inds)
    
        logits_list = [model(ind_concat.clone()).clone() for model in models[:-1]]
        
        L_is = []
        L_emb_is = []
        for i, personal in enumerate(personals[:-1]):
            logits = logits_list[i].clone().cpu()
            L_i = criterion(logits, label)
            L_is.append(L_i)
            personal.append(float(L_i.clone().detach()))

            src = ind_stack[i].clone()
            auxs = [ind_stack[i].clone() for i in range(len(ind_stack))]
            auxs.pop(i)
            L_emb_i = embed_loss(src, auxs).cpu()
            L_emb_is.append(L_emb_i)
            embs[i].append(float(L_emb_i.clone().detach()))
        
        L_i_tensor = torch.stack(L_is)
        L_emb_tensor = torch.stack(L_emb_is)
        
        L_syms = []
        for i in range(len(L_is)):
            param = collab_params[i]
            aux_idxs = [j!=i for j in range(len(L_is))]
            aux_L = L_i_tensor.clone()[aux_idxs]

            L_sym_i = (1-param)*L_i_tensor[i] + param*torch.sum(aux_L) + (param**2)*L_emb_tensor[i]

            L_syms.append(L_sym_i)

        if epoch >= uplift:
            upstream_input = torch.cat(logits_list, dim=1).to(device)

            final_logits = models[-1](upstream_input, ind_concat.clone()).to('cpu')

            L_F = criterion(final_logits, label)
            personals[-1].append(float(L_F.clone().detach()))

            L_up_sum = eps + torch.sum(L_i_tensor).detach()

            L_sym_F = L_F*(1+torch.exp(eps-temp*L_up_sum))
            L_syms.append(L_sym_F)

            for i, L_i in enumerate(L_i_tensor.clone().detach()):
                L_blame_i = lamb*(L_i/L_up_sum)*L_F.clone()
                blames[i].append(float(L_blame_i.clone().detach()))
                L_syms[i] = L_syms[i] + L_blame_i

        for i, L_sym_i in enumerate(L_syms):
            syms[i].append(float(L_sym_i.clone().detach()))
            if i != len(L_syms):
                L_sym_i.backward(retain_graph=True)
            else:
                L_sym_i.backward()
            
        for i in range(len(syms)-1):
            opts[i].step()
            scheds[i].step()

        if epoch >= uplift:
            opts[-1].step()
            scheds[-1].step()
        
    ## FILL REPORTS
    for i, report in enumerate(reports):
        
        if i != len(reports)-1:
            report['personal'] = np.mean(personals[i])
            report['embedding']= np.mean(embs[i])
            if epoch >= uplift:
                report['blame'] = np.mean(blames[i])
            else:
                report['blame'] = None
                
            report['symbiotic'] = np.mean(syms[i])
        else:
            if epoch >= uplift:
                report['personal'] = np.mean(personals[i])
                report['symbiotic'] = np.mean(syms[i])
            else:
                report['personal'] = None
                report['symbiotic'] = None
    
    return reports
            
def eval_one_epoch(
        eval_loader: DataLoader, 
        models: List[Module], 
        collab_params: List[float], 
        temp: float, 
        epoch: int, 
        criterion: Module, 
        uplift: int = 10, 
        eps: float = 1e-7, 
        lamb: float = 1.0, 
        phase: str = 'Validation') -> List[dict]:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    reports = [{} for model in models]
    personals = [[] for model in models]
    embs = [[] for model in models[:-1]]
    blames = [[] for model in models[:-1]]
    syms = [[] for model in models]
    preds = [[] for model in models]
    
    for model in models:
        model.eval()
        model.to(device)
        
    all_labels = []
    
    ## VALIDATION LOOP
    pbar_eval = tqdm(eval_loader, total=len(eval_loader))
    pbar_eval.set_description(f'Epoch {epoch}: {phase}')
    with torch.no_grad():
        for img, label in pbar_eval:
    
            img = img.to(device)
            all_labels += label.tolist()

            inds = [model.embed(img) for model in models[:-1]]
            
            ind_concat = torch.cat(inds, dim=1)
            ind_stack = torch.stack(inds)
            
            logits_list = [model(ind_concat.clone()).clone() for model in models[:-1]]
            
            probs_list = [F.softmax(logits.clone(),dim=-1).to('cpu') for logits in logits_list]
            preds_list = [torch.argmax(probs.clone(),dim=1).tolist() for probs in probs_list]
            
            L_is = []
            L_emb_is = []
            for i, personal in enumerate(personals[:-1]):
                logits = logits_list[i].clone().cpu()
                L_i = criterion(logits, label)
                L_is.append(L_i)
                personal.append(float(L_i.clone()))

                preds[i] += preds_list[i]
                
                src = ind_stack[i].clone()
                auxs = [ind_stack[i].clone() for i in range(len(ind_stack))]
                auxs.pop(i)
                L_emb_i = embed_loss(src, auxs).cpu()
                L_emb_is.append(L_emb_i)
                embs[i].append(float(L_emb_i.clone()))
            
            L_i_tensor = torch.stack(L_is)
            L_emb_tensor = torch.stack(L_emb_is)
            
            L_syms = []
            for i in range(len(L_is)):
                param = collab_params[i]
                aux_idxs = [j!=i for j in range(len(L_is))]
                aux_L = L_i_tensor.clone()[aux_idxs]
    
                L_sym_i = (1-param)*L_i_tensor[i] + param*torch.sum(aux_L) + (param**2)*L_emb_tensor[i]
    
                L_syms.append(L_sym_i)
    
            if epoch >= uplift:
                upstream_input = torch.cat(logits_list, dim=1).to(device)
    
                final_logits = models[-1](upstream_input, ind_concat.clone()).to('cpu')
                final_probs = F.softmax(final_logits.clone(), dim=-1)
                final_preds = torch.argmax(final_probs, dim=1).tolist()
                preds[-1] += final_preds
                
                L_F = criterion(final_logits, label)
                personals[-1].append(float(L_F.clone()))
    
                L_up_sum = eps + torch.sum(L_i_tensor)
    
                L_sym_F = L_F*(1+torch.exp(eps-temp*L_up_sum))
                L_syms.append(L_sym_F)
    
                for i, L_i in enumerate(L_i_tensor.clone()):
                    L_blame_i = lamb*(L_i/L_up_sum)*L_F.clone()
                    blames[i].append(float(L_blame_i.clone()))
                    L_syms[i] = L_syms[i] + L_blame_i
    
            for i, L_sym_i in enumerate(L_syms):
                syms[i].append(float(L_sym_i.clone()))

    ## FILL REPORTS
    for i, report in enumerate(reports):
        
        if i != len(reports)-1:
            report['personal'] = np.mean(personals[i])
            report['embedding']= np.mean(embs[i])
            if epoch >= uplift:
                report['blame'] = np.mean(blames[i])
            else:
                report['blame'] = None
                
            report['symbiotic'] = np.mean(syms[i])
            report['accuracy'] = accuracy_score(all_labels, preds[i])
        else:
            if epoch >= uplift:
                report['personal'] = np.mean(personals[i])
                report['symbiotic'] = np.mean(syms[i])
                report['accuracy'] = accuracy_score(all_labels, preds[i])
            else:
                report['personal'] = None
                report['symbiotic'] = None
                report['accuracy'] = None
    
    return reports

def train(
        epochs: int, 
        models: List[Module], 
        opts: List[Optimizer],
        scheds: List[LRScheduler], 
        data_loaders: List[DataLoader], 
        collab_params: List[float], 
        temp: float, 
        criterion: Module, 
        tags: List[str] = False, 
        uplift: int = 10, 
        eps: float = 1e-7, 
        lamb: float = 1.0, 
        save_path: str = False, 
        save_best: bool = False, 
        save_end: bool = False, 
        save_before_uplift: bool = False, 
        load_at_uplift: bool = False) -> None:

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    if tags==False:
        tags = [f'Model_{i}' for i in range(len(models)-1)] + ['Readout']

    for tag in tags:
        os.mkdir(f'{save_path}/{tag}')
        
    train_loader, val_loader, test_loader = data_loaders
    
    len_train = len(train_loader)
    len_valid = len(val_loader)
    len_test = len(test_loader)

    lt_reports = []

    for i in range(len(models)):
        lt_report_i = {
            'Validation': {},
            'Testing': {}
        }

        lt_reports.append(lt_report_i)

    if load_at_uplift:
        epoch_range = range(uplift, epochs)
    else:
        epoch_range = range(0, epochs)

    best_epoch = 0.0
    best_readout_acc = 0.0
    for epoch in epoch_range:
        
        train_reports = train_one_epoch(
            train_loader, 
            models, 
            opts, 
            scheds, 
            collab_params, 
            temp, 
            epoch, 
            criterion, 
            uplift=uplift, 
            eps=eps, 
            lamb=lamb
        )
        
        reports_summary(train_reports, epoch, tags)
        
        valid_reports = eval_one_epoch(
            val_loader, 
            models, 
            collab_params, 
            temp, 
            epoch, 
            criterion, 
            uplift=uplift, 
            eps=eps, 
            lamb=lamb, 
            phase='Validation'
        )

        if epoch >= uplift and save_best==True:
            readout_acc = valid_reports[-1]['accuracy']

            if readout_acc > best_readout_acc:
                best_readout_acc = readout_acc
                best_epoch = epoch

                best_checkpoints = []
                for i, model in enumerate(models):
                    tag = tags[i]
                    opt = opts[i]
                    sched = scheds[i]
                    
                    checkpoint = {
                        'epoch': best_epoch,
                        'model': model.state_dict(),
                        'opt': opt.state_dict(),
                        'sched': sched.state_dict(),
                        'last_step': sched.last_epoch
                    }

                    best_checkpoints.append(checkpoint)
        
        reports_summary(valid_reports, epoch, tags)
        fill_lt_reports(lt_reports, valid_reports, 'Validation')
                            
        if (epoch%5 == 0 or epoch == epochs-1) and epoch != 0:
            test_reports = eval_one_epoch(
                test_loader, 
                models, 
                collab_params, 
                temp, 
                epoch, 
                criterion, 
                uplift=uplift, 
                eps=eps, 
                lamb=lamb, 
                phase='Testing'
            )
            
            reports_summary(test_reports, epoch, tags)   
            fill_lt_reports(lt_reports, test_reports, 'Testing')
            
        if epoch == uplift-1 and save_before_uplift == True:
            for i, model in enumerate(models):
                tag = tags[i]
                opt = opts[i]
                sched = scheds[i]
                
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'opt': opt.state_dict(),
                    'sched': sched.state_dict(),
                    'last_step': sched.last_epoch
                }
                
                torch.save(checkpoint, f'{save_path}/{tag}_pre-uplift.pt')
            

    for tag, lt_report in zip(tags, lt_reports):
        for phase in lt_report.keys():
            plot_phase(
                lt_report, 
                epochs, 
                uplift, 
                save_path, 
                load_at_uplift=load_at_uplift, 
                phase=phase, 
                tag=tag
            )

    if save_end:
        for i, model in enumerate(models):
            tag = tags[i]
            opt = opts[i]
            sched = scheds[i]
            
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'opt': opt.state_dict(),
                'sched': sched.state_dict(),
                'last_step': sched.last_epoch
            }
            
            torch.save(checkpoint, f'{save_path}/{tag}.pt')

    if save_best:
        print('Best Epoch: ', best_epoch)
        print('Best Readout Accuracy: ', best_readout_acc)
        for tag, checkpoint in zip(tags, best_checkpoints):
            torch.save(checkpoint, f'{save_path}/{tag}_Best.pt')
        
    return

def plot_phase(
    lt_report: dict, 
    epochs: int, 
    uplift: int, 
    save_path: str, 
    load_at_uplift: bool = False, 
    phase: str = 'Validation',
    tag: str = 'Model_0') -> None:

    label_lookup = {}
    label_lookup['personal'] = 'Average Personal Loss'
    label_lookup['symbiotic'] = 'Average Symbiotic Loss'
    label_lookup['accuracy'] = 'Accuracy'
    label_lookup['embedding'] = 'Average Embedding Loss'
    label_lookup['blame'] = 'Average Blame Loss'
    
    if not load_at_uplift:
        if phase == 'Validation':
            full_axis = list(range(0, epochs))
            post_uplift_axis = list(range(uplift, epochs))
            
        elif phase == 'Testing':
            full_axis = list(np.arange(5,epochs,5)) + [epochs]
            post_uplift_axis = list(np.arange( max([math.ceil(uplift/5)*5, 5]),epochs,5)) + [epochs]
    else:
        if phase == 'Validation':
            full_axis = list(range(uplift, epochs))
            post_uplift_axis = list(range(uplift, epochs))
            
        elif phase == 'Testing':
            full_axis = list(np.arange(uplift,epochs,5)) + [epochs]
            post_uplift_axis = list(np.arange(math.ceil(uplift/5)*5,epochs,5)) + [epochs]
            
    full_axis = np.array(full_axis)
    post_uplift_axis = np.array(post_uplift_axis)
    
    for key in lt_report[phase].keys():
        if tag == 'Readout' or key == 'blame':
            x_axis = post_uplift_axis
        else:
            x_axis = full_axis

        data_clean = [float(x) for x in lt_report[phase][key] if x is not None]
        
        if key == 'accuracy' and tag == 'Readout':
            idx_best = np.argmax(data_clean)
            best_epoch = x_axis[idx_best]
            best_acc = data_clean[idx_best]
            label = f'{tag}: Best Accuracy=\n{best_acc:.4f} at Epoch {best_epoch}'
        else:
            label = f'{tag}: {label_lookup[key]}'

        try:
            plt.plot(x_axis, np.array(data_clean), color='black', linestyle='-', label=label)
        except:
            print(f"Error Encountered Plotting {tag}'s {phase} {key.capitalize()} Report")
            print('x axis: ', x_axis)
            print('data: ', data_clean)
            continue
        
        plt.title(f'{tag} {phase}: {label_lookup[key]} Per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel(f'{label_lookup[key]}')
        plt.grid()
        if tag != 'Readout' and uplift != 0: plt.axvline(x=uplift, linestyle='dashed', color='red', label=f'Uplift: Epoch {uplift}')
        plt.legend()
        plt.savefig(f'{save_path}/{tag}/{phase}_{key}.png')
        plt.clf()
        
    return