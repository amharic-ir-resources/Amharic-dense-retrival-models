# Amharic-IR-Resources
## 📌Amharic Dense Retrieval Models
This repository provides **four Amharic-specific retrieval models**, optimized for **Amharic passage retrieval**. These models outperform multilingual baselines, offering strong benchmarks for **dense retrieval** in low-resource language Amharic.

🚧 **Under Development** 🚧  
> This repository is a work in progress. We are adding features, improving documentation, and refining the code. Expect frequent updates! :)  


## 👐 Features
-  **Pretrained Amharic Retrieval Models** Includes ( RoBERTa-Base-Amharic-Embd, RoBERTa-Medium-Amharic-Embd, BERT-Medium-Amharic-Embd, and ColBERT-AM for dense retrieval.)
-  **Hugging Face model & dataset links for easy access**
-  **Training, evaluation, and inference scripts for reproducibility**
-  **Benchmarks BM25 (sparse retrieval), bi-encoder dense retrieval, and ColBERT (late interaction retrieval) for Amharic.**
-  **MS MARCO-style dataset conversion script & direct dataset links**


## 🚀 Pretrained Models
We provide **both dense embedding models and ColBERT models**.

### 🔹 **Dense Embedding Models**
| Model                          | Hugging Face Link |
|--------------------------------|------------------|
| **RoBERTa-Base-Amharic-Embd**       | [HF Model](https://huggingface.co/OUR_HF_USERNAME/RoBERTa-Base-Amharic) |
| **RoBERTa-Medium-Amharic-Embd**     | [HF Model](https://huggingface.co/OUR_HF_USERNAME/RoBERTa-Medium-Amharic) |
| **BERT-Medium-Amharic-Embd**        | [HF Model](https://huggingface.co/OUR_HF_USERNAME/BERT-Medium-Amharic) |

### 🔹 **ColBERT Retrieval Model**
| Model                          | Hugging Face Link |
|--------------------------------|------------------|
| **ColBERT-Amharic**            | [HF Model](https://huggingface.co/OUR_HF_USERNAME/ColBERT-AM) |

---
## 🗂 Datasets
We use the **Amharic News dataset** and provide a **script to convert it into MS MARCO passage retrieval dataset format**.

### 🔹 **Dataset Links**
| Dataset                         | Link |
|---------------------------------|------|
| **Amharic News Dataset (HF)**   | [Hugging Face](https://huggingface.co/datasets/OUR_HF_USERNAME/Amharic-News) |
| **MS MARCO-style Amharic Dataset** | [Google Drive](https://drive.google.com/OUR_LINK) |

### 🔹 **Create MS MARCO-Style Dataset**
You can use our script to **convert the dataset into MS MARCO format**:
```bash
python scripts/convert_to_msmarco.py --input data/amharic_news.json --output data/msmarco_amharic.json
```


## 📂 Repository Structure
- `models/` - Pre-trained retrieval models scripts  (links provided above)
- `scripts/` - To do Training, evaluation, and inference scripts for all models
- `notebooks/` - Jupyter notebooks for model comparison & visualization
- `data/` - Dataset scripts and sample data
- `results/` - Logs and benchmark results

---

## 🛠 Setup Instructions
### **1️⃣ Clone the Repository**
- Clone the Repository
```bash 
git clone https://github.com/amharic-ir-resources/Amharic-dense-retrival-models.git
cd Amharic-dense-retrival-models
```
### **2️⃣ Install Dependencies**
You can install dependencies using either Conda or pip, depending on your preference.

🔹 Option 1: Using Conda (Recommended)`
- Conda ensures better reproducibility and manages non-Python dependencies automatically.

```bash 
conda env create -f amharic_environment.yml
conda activate amharic_environment
```
🔹 Option 2: Using Pip
If you prefer pip, install dependencies with:

```bash
pip install -r requirements.txt
```

### **3️⃣ Verify Installation**
```bash
python -c "import torch, transformers, numpy; print(' All dependencies installed successfully!🎉')"
```

## 🔍 How to Train and Evaluate Models
### **4️⃣ Train a Model**
```bash
sbatch scripts/train_colbert.sh
```

### **5️⃣ Evaluate Models**
```bash
sbatch scripts/evaluate.sh
```

### **6️⃣ Inference (Retrieval)**
- Example script
```bash
python models/RoBERTa-Base-Embd/inference.py --query "ሰፊ የሆነ የዜና ምርጫ"
```


## 📊 Benchmark Results
| Model                         | MRR@10 | Recall@10 | Recall@100 |
|--------------------------------|--------|-----------|------------|
| BM25-AM                          | 0.657  | 0.774     | 0.871      |
| ColBERT-AM                     | 0.754  | 0.858     | 0.931      |
| **RoBERTa-Base-Amharic-embd**       | **0.755**  | **0.897**  | **0.971**  |
| RoBERTa-Medium-Amharic-embd         | 0.707  | 0.861     | 0.963      |


### **Multilingual vs Amharic-Specific Models**  
We compare our Amharic-specific dense retrieval models against multilingual baselines. Our models significantly outperform the multilingual counterparts in passage retrieval tasks.

To ensure a fair comparison, we evaluate the top-performing multilingual embedding models from the [Massive Text Embedding Benchmark (MTEB) Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) using the same test set as our Amharic embedding models. This provides a direct performance comparison between general-purpose multilingual models and those specifically optimized for Amharic retrieval.

Our largest embedding model, RoBERTa-Base-Amharic-Embed beats all of the multilingual embedding models on the MRR@10, NDCG@10 and Recall metrics while having 1/5th of their param count 🚀.

| Model | Params | MRR@10 | NDCG@10 | Recall@100 |
|--------------------------------|--------|--------|---------|------------|
| **Multilingual Models** |  |  |  |  |
| gte-modernbert-base | 149M | 0.019 | 0.022 | 0.065 |
| gte-multilingual-base | 305M | 0.649 | 0.684 | 0.904 |
| multilingual-e5-large-instruct | 560M | 0.713 | 0.747 | 0.946 |
| snowflake-arctic-embed-l-v2.0 | 568M | 0.719 | 0.755 | 0.957 |
| **Amharic-Specific Models (Ours)** |  |  |  |  |
| BERT-Medium-Amharic-embed | 40M | 0.657 | 0.696 | 0.945 |
| RoBERTa-Medium-Amharic-embed | 42M | 0.707 | 0.744 | 0.963 |
| RoBERTa-Base-Amharic-embed | 110M | **0.755** † | **0.790** † | **0.971** † |

---

## 📜 Citation
```bibtex
@article{our_paper,
  title={Optimized Text Embedding Models and Benchmarks for Amharic Passage Retrieval},
  author={our Name},
  year={2025},
  journal={Journal_Name 2025}
}
```

## 📄 License
This project is licensed under the Apache 2.0 License.

## 📚 Acknowledgment
This repository makes use of code from the original **ColBERT repository** by [Stanford FutureData Lab](https://github.com/stanford-futuredata/ColBERT). We appreciate their work in open-sourcing ColBERT and providing a strong foundation for efficient dense retrieval.


## 📧 Contact
For any questions or collaborations, feel free to reach out to us at **dummy-email-1@example.com** or **dummy-email-2@example.com**.