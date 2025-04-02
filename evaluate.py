import os
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve


def compute_auc(labels, scores):
    scores = [-item for item in scores]
    auc_score = roc_auc_score(labels, scores)
    return auc_score


def compute_tpr_at_fpr(labels, scores, fpr_target):
    scores = [-item for item in scores]
    fpr, tpr, thresholds = roc_curve(labels, scores)

    #index_at_fpr = np.argmin(np.abs(fpr - fpr_target))
    index_at_fpr = next(i for i, f in enumerate(fpr) if f >= fpr_target)
    tpr_at_fpr = tpr[index_at_fpr]
    return tpr_at_fpr


if __name__ == "__main__":
    human_dim_path = 'data/generated_dataset/human_dim_trunc_50.jsonl'
    machine_dim_path = 'data/generated_dataset/machine_dim_trunc_50.jsonl'
    #human_dim_path = 'data/GPTID/human_dim.jsonl'
    #machine_dim_path = 'data/GPTID/machine_dim.jsonl'
    df_human = pd.read_json(human_dim_path, lines=True)
    df_machine = pd.read_json(machine_dim_path, lines=True)
    domain_list = df_human['domain'].unique().tolist()
    generator_list = df_machine['generator'].unique().tolist()

    # Uncomment to evluate Wikipedia in GPTID
    #generator_list = ['davinci', 'gpt2', 'opt']
    #domain_list = ['wikipedia']

    # Uncomment to evluate WritingPrompts in GPTID
    #generator_list = ['davinci']
    #domain_list = ['writingprompt']

    # remove outliers
    df_human = df_human[df_human['dim'] < 30]
    df_machine = df_machine[df_machine['dim'] > 0]


    results_ours = []
    results_baseline = []
    for generator in generator_list:
        per_generator_results_baseline = [generator]
        per_generator_results_ours = [generator]
        df_machine_generator = df_machine[df_machine['generator'] == generator].copy()
        df_human = df_human.copy()

        labels = []
        score_baseline = []
        score_ours = []
        for i in range(df_human.shape[0]):
            try:
                score_ours.append(np.mean(df_human['dim_prompt'].iloc[i]))
                score_baseline.append(df_human['dim'].iloc[i])
                labels.append(0)
    
                score_ours.append(np.mean(df_machine_generator['dim_prompt'].iloc[i]))
                score_baseline.append(df_machine_generator['dim'].iloc[i])
                labels.append(1)
            except:
                continue

        auc = compute_auc(labels, score_baseline)
        per_generator_results_baseline.append(auc)
    
        auc = compute_auc(labels, score_ours)
        per_generator_results_ours.append(auc)

        for domain in domain_list:
            df_machine_generator_domain = df_machine_generator[df_machine_generator['domain'] == domain].copy()
            df_human_this = df_human[df_human['domain'] == domain].copy()
    
            label = []
            score_baseline = []
            score_ours = []
            for i in range(df_human_this.shape[0]):
                try:
                    score_ours.append(np.mean(df_human_this['dim_prompt'].iloc[i]))
                    score_baseline.append(df_human_this['dim'].iloc[i])
                    label.append(0)
    
                    score_ours.append(np.mean(df_machine_generator_domain['dim_prompt'].iloc[i]))
                    score_baseline.append(df_machine_generator_domain['dim'].iloc[i])
                    label.append(1)
                except:
                    continue
    
            auc = compute_auc(label, score_baseline)
            per_generator_results_baseline.append(auc)
        
            auc = compute_auc(label, score_ours)
            per_generator_results_ours.append(auc)
        
        results_baseline.append(per_generator_results_baseline)
        results_ours.append(per_generator_results_ours)

    df_results_baseline = pd.DataFrame(results_baseline, columns=['generator', 'all'] + domain_list)
    df_results_ours = pd.DataFrame(results_ours, columns=['generator', 'all'] + domain_list)
    df_results_baseline['method'] = 'PHD'
    df_results_ours['method'] = 'Short-PHD'
    df_results = pd.concat([df_results_baseline, df_results_ours], axis=0)
    df_results = df_results.sort_values(by=['method', 'generator'])
    df_results = df_results.reset_index(drop=True)
    print(df_results)
    #df_results.to_excel(f'eval_results.xlsx', index=False)



