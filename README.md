# 🧪 Data Preprocessing Pipeline — Interactive Dashboard

An interactive **Streamlit** web app that walks through every stage of a professional data preprocessing pipeline applied to the **Ames Housing Dataset**.  
Designed as a visual companion to the original Jupyter Notebook, making every decision — from outlier removal to one-hot encoding — fully transparent and explorable.

---

## 📸 Preview

> Navigate through the pipeline using the sidebar — each section shows the concept, the code, and a live visualization.

| Section | What you'll find |
|---|---|
| 🏠 Overview | Pipeline summary, key metrics, imports |
| 📊 Outliers | Scatter plots, threshold rules, removed rows |
| ❓ Missing Data | Missing % chart, fill/drop/impute decisions |
| 🔢 Categorical | Dummy encoding, MS SubClass fix, column growth |
| ✅ Final Dataset | Complete step checklist, pipeline flow diagram |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/data-preprocessing-dashboard.git
cd data-preprocessing-dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run data_preprocessing_app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🗂️ Project Structure

```
data-preprocessing-dashboard/
│
├── data_preprocessing_app.py   # Main Streamlit application
├── data-preprocessing.ipynb    # Original Jupyter Notebook
├── requirements.txt            # Python dependencies
└── README.md                   # You're here
```

---

## 🔬 Pipeline Stages Covered

### 1 · Outlier Detection & Removal
- Visual inspection via scatter plots (`Overall Qual` vs `SalePrice`, `Gr Liv Area` vs `SalePrice`)
- Threshold-based removal: houses with `Gr Liv Area > 4000` and `SalePrice < $400k` dropped
- Box plots for post-removal verification

### 2 · Missing Data Handling
- Custom `missing_percent()` function to quantify and visualize NaN distribution
- **Drop rows** when < 1% missing (`Electrical`, `Garage Area`)
- **Drop columns** when > 80% missing (`Fence`, `Alley`, `Pool QC`, `Misc Feature`)
- **Fill with domain defaults** for Basement, Garage, Masonry Veneer columns (`0` or `"None"`)
- **Group-mean imputation** for `Lot Frontage` based on `Neighborhood`

### 3 · Categorical Encoding
- Convert `MS SubClass` from integer → string (prevents ordinal misinterpretation)
- One-hot encode all object columns with `pd.get_dummies(drop_first=True)`
- Final dataset: **274 features**, fully numeric, zero missing values

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Interactive web dashboard |
| [Pandas](https://pandas.pydata.org) | Data manipulation |
| [NumPy](https://numpy.org) | Numerical operations |
| [Matplotlib](https://matplotlib.org) | Custom dark-themed charts |
| [Seaborn](https://seaborn.pydata.org) | Statistical visualizations |

---

## 📦 Dataset

**Ames Housing Dataset** — A comprehensive dataset describing the sale of residential properties in Ames, Iowa (2006–2010).

- ~2,900 rows
- 80+ original features
- Target variable: `SalePrice`

> The dataset is not included in this repo. You can download it from [Kaggle](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset) or use `Housing_Price_Data.csv` if you have it locally.

---

## 💡 Key Concepts Demonstrated

- Statistical outlier definition and visual detection
- Missing data taxonomy (MCAR / MAR / MNAR)
- The fill/keep/drop decision framework
- Neighborhood-based imputation strategy
- The dummy variable trap and why `drop_first=True` matters
- Categorical vs. ordinal feature distinction

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- Dataset originally from the [Ames Housing Study](http://jse.amstat.org/v19n3/decock.pdf) by Dean De Cock
- Preprocessing methodology inspired by standard Kaggle competition pipelines
