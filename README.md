# DG-NILM - A high-frequency dataset for non-intrusive load monitoring and PV-inverter BTM identification

DG-NILM-V1 is a dataset for NILM and BTM identification composed of electrical variables time-windows samples in a real-world distribution system with both electrical appliances and photovoltaic generation. Dataset version V1 measurements were obtained between December 2022 and February 2023, at POLITEC facilities, at the Federal University of Technology (UTFPR), in Pato Branco, Paraná, Brazil. Figure 1 shows the single-line diagram of our proposed experimental setup.

![Figure 1 - Single-Line diagram of our experimental setup](main/Esquematico_Geral.pdf)

Each acquisition window (AW) has 16s of sampled variables at 1000kHz, totalizing 16000 samples per AW. We separated the entire dataset into two subsets: Aggregated (with overlapped appliances) and Individual (without overlapped appliances). For each AW, we anottated the state (turned on or turned off) of a PV inverter connected to the Point of Commum Connection (PCC).



