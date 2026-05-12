GRN\_benchmark/

├── run\_beeline\_garnet\_protocol.py    ← Main（data + evaluation + protocol）

├──  Datasets: All datasets We used to evaluate.

├── methods/                          ← 8 methods validataion

│   ├── \_\_init\_\_.py

│   ├── genie3.py

│   ├── grnboost2.py

│   ├── ilsgrn.py

│   ├── tigress.py

│   ├── deepsem.py

│   ├── deeptgi.py

│   ├── gnnlink.py

│   └── linger.py

├── slurm/                            ← Slurm script

│   ├── gp\_cpu1.sbatch                  (GENIE3 + GRNBoost2, CPU)

│   ├── gp\_cpu2.sbatch                  (TIGRESS, CPU)

│   ├── gp\_cpu3.sbatch                  (iLSGRN, CPU)

│   └── gp\_gpu.sbatch                   (DeepSEM/DeepTGI/GNNLink/LINGER, GPU)

└── results/                          ← output results

&#x20;   ├── bee\_gp\_cpu1.tsv                 (GENIE3)

&#x20;   ├── bee\_gp\_cpu2.tsv                 (TIGRESS)

&#x20;   ├── bee\_gp\_cpu3.tsv                 (iLSGRN)

&#x20;   └── bee\_gp\_gpu.tsv                  (DeepSEM/DeepTGI/GNNLink/LINGER)

