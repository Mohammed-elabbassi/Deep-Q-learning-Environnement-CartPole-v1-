## Projet DQN - Resolution de CartPole-v1
Description
Implementation complete d'un agent Deep Q-Network (DQN) pour resoudre l'environnement CartPole-v1 de Gymnasium. Ce projet demontre l'application de l'apprentissage par renforcement profond a un probleme de controle classique.

Caracteristiques principales
Architecture DQN complete avec reseau cible et replay buffer

Entrainement sur 2000 episodes

Visualisation complete des resultats

Rapport LaTeX professionnel

Code modulaire et bien documente

Structure du projet
text
dqn-cartpole/
├── dqn_agent.py          # Classe principale de l'agent DQN
├── train.py             # Script d'entrainement
├── evaluate.py          # Script d'evaluation
├── requirements.txt     # Dependances Python
├── README.md           # Ce fichier
├── model.pth           # Modele pre-entraine
├── training_logs.csv   # Logs d'entrainement
└── images/             # Graphiques generes
Installation
Prérequis
Python 3.8 ou superieur

pip

Installation des dependances
bash
pip install torch gymnasium numpy matplotlib pandas tqdm
Utilisation
1. Entrainer le modele
bash
python train.py
2. Evaluer le modele
bash
python evaluate.py
3. Generer les graphiques
bash
python -c "
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv('training_logs.csv')
plt.figure(figsize=(10, 6))
plt.plot(data['episode'], data['reward'])
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Evolution des recompenses')
plt.savefig('images/rewards.png')
plt.show()
"
Resultats
Performance
Recompense finale : 362.0

Score moyen sur 100 episodes : 350.2

Taux de reussite (>195 pas) : 92%

Hyperparametres
Gamma : 0.99

Taux d'apprentissage : 1e-3

Taille du batch : 64

Taille de la memoire : 100,000

Decroissance d'epsilon : 0.999

Architecture du reseau

DQN(
  Linear(4 -> 128)
  ReLU()
  Linear(128 -> 128)
  ReLU()
  Linear(128 -> 2)
)
Analyse des resultats
Phase 1: Exploration (episodes 0-500)
Recompense moyenne: 2.5

Comportement principalement aleatoire

Phase 2: Apprentissage (episodes 500-1500)
Recompense moyenne: 8.2

Decouverte de strategies efficaces

Phase 3: Convergence (episodes 1500-2000)
Recompense moyenne: 35.2

Politique stable et efficace
