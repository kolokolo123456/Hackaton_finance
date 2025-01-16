#!/usr/bin/env python3
import subprocess
import os

def run_step(step_script):
    """Exécute le script Python de l'étape donnée."""
    try:
        print(f"Exécution de {step_script}...")
        result = subprocess.run(['python3', os.path.expanduser(step_script)], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Succès : {step_script} a été exécuté.")
            print(result.stdout)
        else:
            print(f"Erreur lors de l'exécution de {step_script}: {result.stderr}")
    except Exception as e:
        print(f"Erreur d'exécution de {step_script}: {str(e)}")

steps = [
    '~/Hackaton_finance/etape1.py',  
    '~/Hackaton_finance/etape2.py',  
    '~/Hackaton_finance/etape3.py',
    '~/Hackaton_finance/etape4.py',  
    '~/Hackaton_finance/etape5.py'   
]

for step in steps:
    run_step(step)

print("Toutes les étapes ont été exécutées.")
