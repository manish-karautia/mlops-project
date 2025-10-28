#  Forecasting Demand for Critical Health Commodities

## 📘 Overview

This project focuses on **developing a machine learning-based demand forecasting model** for essential health commodities across different health categories such as **HIV/AIDS, maternal and child health, tuberculosis, and COVID-19**.  
By leveraging historical order data, fiscal year funding, and program details, the project aims to **enhance inventory management**, **reduce stockouts**, and **improve procurement strategies** for critical health supplies.

---

## 🎯 Research Problem

Health systems often face challenges in maintaining an optimal stock of medical commodities due to uncertain demand patterns, funding delays, and logistical bottlenecks.  
The goal of this project is to build a **data-driven forecasting model** that can accurately predict future demand for each product category, enabling more reliable supply chain planning.

---

## 🌍 Data Source

The dataset used for this project is publicly available through the **USAID GHSC-PSM Health Commodity Delivery Dataset**:

🔗 [Dataset Link](https://catalog.data.gov/dataset/usaid-ghsc-psm-health-commodity-delivery-dataset)

It contains detailed information on global health product deliveries, program categories, funding sources, and geographic coverage.

---

## 📊 Input Features

The model utilizes multiple feature categories that influence demand:

### 1. Historical Demand Data
- Order Volume (past quantities)
- Order Frequency
- Stockout Events

### 2. Temporal Features
- Fiscal Year and Quarter
- Seasonality Trends

### 3. Health Program Details
- Health Category (HIV/AIDS, MCH, COVID-19, etc.)
- Product Category (ARV drugs, vaccines, PPE, etc.)
- Funding Source (USAID, Global Fund)
- Program Scale (population or geographic reach)

### 4. Geographic and Demographic Features
- Country or Region
- Population Metrics
- Disease Prevalence
- Health Indicators (e.g., vaccination rates)

### 5. Economic and Logistical Features
- Funding Levels
- Supplier Lead Times
- Delivery Reliability

### 6. External Factors
- Pandemic/Epidemic Events (e.g., COVID-19)
- Policy or Regulatory Changes
- Global Supply Chain Constraints

---

## 🎯 Output Targets

The model can support multiple prediction objectives:

- **Time Series Forecasting:** Predict future demand quantities for a specific time horizon.  
- **Classification (Optional):** Categorize future demand into *Low*, *Moderate*, or *High*.  
- **Inventory Optimization:** Recommend reorder quantities to minimize shortages and overstocking.

---

## 🧠 Machine Learning Pipeline

### **Step 1: Data Preprocessing**
- Handle missing values (orders, funding, etc.)
- Aggregate data by month or quarter
- Encode categorical variables
- Normalize continuous variables

### **Step 2: Feature Engineering**
- Identify trends and seasonality
- Create lag and rolling-window features
- Generate interactions (e.g., Health Category × Region)

### **Step 3: Model Selection**
- **Time Series Models:** ARIMA, SARIMA, Prophet  
- **Machine Learning Models:** Random Forest, XGBoost, LightGBM  
- **Deep Learning Models:** LSTM or Transformer-based architectures

### **Step 4: Model Evaluation**
- Metrics: `MAPE`, `MAE`, `RMSE`
- Use backtesting to compare predicted vs. actual demand

### **Step 5: Model Deployment**
- Integrate model with dashboards for visualization
- Use predictions for dynamic inventory management

---

## 🧩 Example Use Case

| **Input Variable** | **Example Value** |
|--------------------|------------------|
| Historical Demand | 10,000 units (last quarter) |
| Health Category | HIV/AIDS |
| Program Funding | \$5 million |
| Region | Sub-Saharan Africa |
| Fiscal Year | FY2025 Q1 |
| Population Served | 1 million |
| Lead Time | 30 days |
| Disease Incidence | 15% |

**→ Forecasted Demand (FY2025 Q2): 12,000 units**

---

## 🚀 Potential Outcomes

- **Enhanced Inventory Management:** Minimize both stockouts and overstocking  
- **Optimized Procurement:** Align funding utilization with demand trends  
- **Data-Driven Decisions:** Empower policymakers and health organizations  
- **Supply Chain Resilience:** Improve readiness for global health crises

---

## 🧰 Tech Stack (Suggested)

| Component | Tools / Libraries |
|------------|-------------------|
| Data Handling | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| ML Algorithms | `scikit-learn`, `xgboost`, `lightgbm` |
| Deep Learning | `TensorFlow`, `Keras`, `PyTorch` |
| Time Series | `Prophet`, `statsmodels` |
| Deployment | `Streamlit`, `Flask`, `Dash` |

---

## 🧪 Evaluation Metrics

- **Mean Absolute Percentage Error (MAPE)**
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**

---

## 📈 Future Work

- Incorporate **attention-based Transformer models** for complex temporal dependencies  
- Explore **Bayesian Optimization** for hyperparameter tuning  
- Develop a **forecast visualization dashboard** for policymakers  
- Enable **rolling forecasts** for adaptive inventory planning  

---

## 👨‍💻 Author

**Manish Karautia**  
MSc Data Science, Christ University  
Focused on supply chain visibility and demand forecasting using AI/ML  
📧 [Add your email or LinkedIn here]

---

## 🏷️ License

This project is open-source and available under the [MIT License](LICENSE).

---

## ⭐ Acknowledgments

Data provided by **USAID Global Health Supply Chain Program (GHSC-PSM)**.  
Special thanks to all open data initiatives supporting healthcare transparency and efficiency.

---
