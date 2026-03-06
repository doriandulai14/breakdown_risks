from os import error
from numpy._core import numeric
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import tkinter as tk
import os
import seaborn as sns
import platform




def run_analysis_from_path():
    print("HELP: right click on the file -> copy as path -> paste")
    file_path = input("Please enter a valid path to a file: ")
    
    file_path = file_path.strip().replace('"', '').replace("'", "")
    
    if not os.path.exists(file_path):
        print(f"Error with file path: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
        print("success!")
    except Exception as e:
        print(f"Error: {e}")
        return

    df = df.dropna(subset=['high_breakdown_risk'])
    df = df[df['oph'] < 1000000]
    
    if 'issue_type' in df.columns:
        mapping = {'typical': 1, 'atypical': 2, 'non-related': 3, 'non-symptomatic': 4}
        df['issue_type'] = df['issue_type'].map(mapping)

    features = ['oph', 'pist_m', 'issue_type', 'bmep', 'ng_imp', 'past_dmg']
    X = df[features].fillna(0)
    y = df['high_breakdown_risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred=model.predict(X_test)
    print(classification_report(y_test, y_pred))
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
   
    print("\nOrder of importance of variables in prediction:")
    print(importances)

    print("\nModel succesfully taught.")

    # Saving the results to a txt
    with open("results.txt", "w") as r:
        r.write("---STATISTICAL SUMMARY--\n")
        r.write(df[['bmep','ng_imp','rpm_max','oph']].describe().to_string())
        r.write("\n---IMPORTANCE OF VARIABLES---")
        r.write(f"\n{importances.to_string()}")

    create_visualizations(df, importances)


def create_visualizations(df, importances):
    #default
    sns.set_theme(style="whitegrid")
    
    # 1. DIAGRAM
    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette='viridis', legend=False)
    plt.title('Which variables have the greatest impact on failure')
    plt.xlabel('Values of importance')
    plt.ylabel('Variables')
    plt.tight_layout()
    plt.show()
    
    # 2. DIAGRAM
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='high_breakdown_risk', y='ng_imp', hue='high_breakdown_risk', data=df, palette='Set2', legend=False)
    plt.title('Gas pollution (ng_imp) and the Breakdown risks relationship')
    plt.xticks([0, 1], ['Low risk (0)', 'High risk (1)'])
    plt.xlabel('Risk level')
    plt.ylabel('Contamination extent (nmol)')
    plt.show()

    # 3. DIAGRAM
    plt.figure(figsize=(12, 8))
    df_plot = df.groupby(['resting_analysis_results', 'high_breakdown_risk']).size().unstack()
    df_plot.plot(kind='bar', stacked=True, color=['green', 'red'], ax=plt.gca())
    plt.title('Impact of diagnostic reults on risk')
    plt.xticks([0, 1, 2], ['0: Normal', '1: Abnormal', '2: Critical'], rotation=0)
    plt.xlabel('Resting Analysis result')
    plt.ylabel('Number of engines')
    plt.legend(['Low risk', 'High risk'])
    plt.show()

    # 4. DIAGRAM
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='RdYlGn', fmt='.2f')
    plt.title('Correlation between variables')
    plt.tight_layout()
    plt.show()

def open_report(file_name):
    try:
        if platform.system() == 'Windows':
            # Windows
            os.startfile(file_name)
        elif platform.system() == 'Darwin':
            # macOS
            import subprocess
            subprocess.call(['open', file_name])
        else:
            # Linux
            import subprocess
            subprocess.call(['xdg-open', file_name])
        print(f"{file_name} success.")
    except Exception as e:
        print(f"Error: {e}")

run_analysis_from_path()
open_report("results.txt")









































