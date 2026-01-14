import config
import pandas as pd
from sklearn.metrics import classification_report

if __name__== "__main__":
    csv_path = config.PROJECT_ROOT / 'results/MobileNetV2_FAKE1_REAL.csv'
    df = pd.read_csv(csv_path)
    y_true = df['label']
    y_pred = df['preds']
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))
