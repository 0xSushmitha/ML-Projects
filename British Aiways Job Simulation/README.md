# ✈️ British Airways Job Simulation – Data Analysis & Machine Learning

## 📌 Project Overview

This project simulates the responsibilities of a data analyst or business analyst at British Airways. It includes **web scraping**, **natural language processing**, and **predictive modeling** to gather insights and predict customer behavior using real review data and booking patterns.

---

## 📂 Contents

- **Task 1:** Web scraping of customer reviews, sentiment analysis, topic modeling.
- **Task 2:** Predictive modeling to estimate likelihood of booking completion using structured customer booking data.

---

## 🧠 Machine Learning & NLP Concepts Used

### ✅ **Task 1: Web Scraping & Sentiment Analysis**

- **BeautifulSoup**  
  Used to scrape **1000 British Airways reviews** from Skytrax across 10 paginated pages.

- **Text Cleaning**  
  Regular expressions (`re`) used to remove tags and irrelevant patterns (e.g., “Trip Verified”).

- **Sentiment Analysis** – `TextBlob`  
  Each review is classified into **Positive**, **Negative**, or **Neutral** sentiment based on polarity.

- **Topic Modeling** – `Latent Dirichlet Allocation (LDA)`  
  Extracted **key themes from reviews** using bag-of-words and unsupervised topic modeling:
  - Topics covered included flight delays, baggage issues, customer service, and in-flight experience.

- **Word Cloud Visualization**  
  Generated a visual representation of frequently used words in reviews.

---

### ✅ **Task 2: Predictive Modeling of Customer Bookings**

- **EDA (Exploratory Data Analysis)**  
  - `.describe()`, `.info()`, value counts, and histograms for distribution analysis.
  - Visualizations: Booking completion rate, lead time patterns, flight durations.

- **Feature Engineering**
  - Mapped categorical variables like flight days into numerical format.
  - One-hot/Label encoded features: `sales_channel`, `trip_type`, `route`, etc.

- **Handling Imbalance**
  - The target variable `booking_complete` was imbalanced.
  - Used **`class_weight='balanced'`** in the model to counter this.

- **Data Preprocessing**
  - **Train/Test Split:** 80/20 ratio with stratification.
  - **StandardScaler:** Used to normalize numerical features like `purchase_lead` and `flight_duration`.

- **Modeling: Random Forest Classifier**
  - Fitted a `RandomForestClassifier` to predict whether a booking would be completed.
  - Achieved **~85% accuracy**, but with class imbalance, precision/recall metrics were critical.

- **Model Evaluation**
  - Accuracy score, classification report (precision, recall, f1-score).
  - Confusion matrix heatmap.
  - Feature importance chart highlighted top drivers of booking completion.

---

## 📁 Data Used

- `BA_reviews.csv` – Scraped dataset of 1000 customer reviews.
- `customer_booking.csv` – Structured dataset of 50,000 historical bookings.
- Both datasets were preprocessed and analyzed using Python libraries.
- Unzip the data from data.zip and place it with the notebook files

---

## 📊 Key Findings

- **Reviews:** Most customers expressed either strong satisfaction or frustration.  
  Sentiment was ~61% positive, 37% negative.

- **Topics:** LDA identified 5 main concerns (crew, food, delays, baggage, refunds).

- **Booking Behavior:**
  - Longer purchase lead times were correlated with booking completion.
  - Customers requesting extras like baggage or preferred seats showed higher likelihood to complete bookings.
  - Sales channel and booking origin had predictive value.

---

## 🛠️ Tech Stack & Libraries

```bash
Python, Jupyter Notebook
```

# Libraries
- pandas, numpy, BeautifulSoup, requests, re
- TextBlob, matplotlib, seaborn, wordcloud
- scikit-learn (LDA, Random Forest, preprocessing, model evaluation)

---
##  Author

**Name**: Sushmitha Bakthavatchalam