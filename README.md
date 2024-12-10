# COSC569_Llama3Guard_Phishing

This is a University of Tennessee COSC569 Human Factors in Cybersecurity final project. We fine-tuned the Llama3 model containing safety guardrails on a phishing dataset provided by Dr. Kim. We investigate whether the model can generate HTML code that creates a decent phishing website in comparison to the original Llama3 model. This repository accompanies our final report included.


## Authors
- [Ria Patel](https://github.com/patelria007)
- [Jacob Malloy](https://github.com/JacobMalloy)
- [Lu Liu](https://github.com/Mirandalllll)

## Requirements
**Google Colaboratory Notebook**
- `requirements.txt`: installs packages from pip, which is done in the notebook

The following frameworks are used:
- PyTorch
- Hugging Face:
   - Transformers
   - PEFT
   - TRL (Transformer Reinforcement Learning)


## Dataset

The dataset for this project was sourced from Dr. Kim’s phishing website database, containing 574 phishing websites. Each entry in the dataset included various elements:
- Website IDs: Unique identifiers for each website instance
- Base Domains: The primary domains used to host the phishing sites, which allowed us to analyze domain-related patterns.
- Brand Names: The targeted brands, provide context for how phishing attacks were designed to deceive users.
- HTML Code: Both desktop and mobile versions of each phishing website’s HTML structure, enabling the model to learn design patterns and adapt across device types.


## Rogue Llama 3 Model Fine-Tuning
`src/RogueLlama3_Guard_8B_Phishing_Dataset.ipynb`: Google Colaboratory Notebook that fine-tunes Llama 3 model containing the safety guard rails. Tests it with questions to see if it will generate good phishing websites with the guardrails in place. 
