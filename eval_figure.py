import matplotlib.pyplot as plt
import os 
import json 


if __name__=='__main__':
    model_dir = 'models'
    data_to_plot = dict([])

    for model_name in os.listdir(model_dir):
        training_json = f'{model_dir}/{model_name}/training/metrics.jsonl'

        if not os.path.exists(training_json): # Skip if the file not exist yet 
            continue

        with open(training_json, 'r') as f:
            training_report = [json.loads(line) for line in f if line.strip()]
        
        steps, train_losses, train_pples, val_pples = [], [], [], []

        starting_n = 10 # Let's skip the first few steps for the sake of clarity
        for row in training_report:
            steps.append(row['step'])
            train_losses.append(row['train_loss'])
            train_pples.append(row['train_perplexity'])
            val_pples.append(row['val_perplexity'])

        data_to_plot[model_name] = {
            'step': steps[starting_n:],
            'train_loss': train_losses[starting_n:],
            'train_ppl': train_pples[starting_n:],
            'val_ppl': val_pples[starting_n:]
        }

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    loss_lines, ppl_lines = [], []

    for i, (model_name, data) in enumerate(data_to_plot.items()):
        color = colors[i % len(colors)]
        steps = data['step']

        l1, = ax1.plot(steps, data['train_loss'], color=color, linestyle='-',
                       label=f'{model_name} — train loss')
        l2, = ax2.plot(steps, data['val_ppl'], color=color, linestyle='--',
                       label=f'{model_name} — val ppl')

        loss_lines.append(l1)
        ppl_lines.append(l2)

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Train Loss', color='black')
    ax2.set_ylabel('Val Perplexity', color='black')

    all_lines = loss_lines + ppl_lines
    ax1.legend(all_lines, [l.get_label() for l in all_lines],
               loc='upper right', fontsize=8)

    fig.suptitle('Train Loss (—) and Val Perplexity (- -) per Step', fontsize=13)
    plt.tight_layout()
    plt.savefig('figures/training_curves.png', dpi=150)
    plt.show()
