import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('gpt_training_history.csv')
df['train_loss'] = df['train_loss'].str.extract(r'tensor\(([\d.]+)\)').astype(float)
df['val_loss'] = df['val_loss'].str.extract(r'tensor\(([\d.]+)\)').astype(float)

plt.figure(figsize=(10, 6))
for exp_type in df['exp_type'].unique():
    subset = df[df['exp_type'] == exp_type]
    line, = plt.plot(subset['step'], subset['train_loss'], label=f'{exp_type} - Train Loss')
    line_color = line.get_color()
    plt.plot(subset['step'], subset['val_loss'], label=f'{exp_type} - Val Loss', color=line_color, linestyle='--')

plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('training_history_plot.png')  # Save the plot as an image file
plt.show()