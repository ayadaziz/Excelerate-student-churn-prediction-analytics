# AI-Powred-Data-Insight-Virtual-Internship

A comprehensive data science project analyzing student engagement patterns and predicting churn risk using machine learning models. This repository contains the complete analysis from an AI-Powered Data Insights Virtual Internship with Excelerate.


# Executive Summary

## Overview

This project analyzes **SLU Opportunity Wise learner data** to understand engagement patterns, predict student churn, and design targeted recommendation systems. Our analysis of **8,558 records** revealed critical insights that could potentially retain **1,200+ additional students** and protect millions in revenue.

## Key Objectives

- Clean and preprocess raw learner data for reliability
- Perform Exploratory Data Analysis (EDA) to uncover behavioral and demographic trends  
- Identify churn signals and model dropout risk
- Develop an AI-driven recommendation system to enhance retention and engagement

## Critical Findings

```{r findings-table, echo=FALSE}
findings <- data.frame(
  Category = c("Churn Rate", "Peak Seasons", "Top Age Group", "Top Countries", "Model Accuracy"),
  Finding = c("70.32%", "January, July-August, Winter", "18-25 years", "US, India, Nigeria", "85%+"),
  Impact = c("Critical retention issue", "Marketing opportunities", "Target demographic", "Geographic focus", "Reliable prediction")
)

kable(findings, caption = "Key Project Findings") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed"))
```

---

# Project Journey

## Week 1: Data Preparation

### Data Overview
- **Dataset Size**: 8,558 records, 16 variables
- **Data Types**: Categorical, temporal, demographic attributes
- **Key Fields**: Signup timestamps, opportunity classifications, geographic data

### Preprocessing Achievements
- Standardized date/time formats using `pd.to_datetime()`
- Resolved missing values in critical fields
- Normalized institution names for consistency
- Removed test entries and malformed records
- **No duplicates found** - indicating strong data integrity

### Feature Engineering
Created **8+ new features** to enrich analysis:

```{r features-table, echo=FALSE}
features <- data.frame(
  Feature = c("Age", "Engagement Time", "Season", "Time Committed", 
              "Quick Applicant", "Fast Starter", "Opportunity Duration", "Engagement Score"),
  Description = c("Calculated from Date of Birth", "Time spent in system", 
                  "Application season categorization", "Total commitment hours",
                  "Applied quickly after signup", "Started program quickly", 
                  "Length of opportunity", "Composite engagement metric"),
  Purpose = c("Demographic analysis", "Behavioral patterns", "Seasonal trends", 
              "Commitment prediction", "Urgency indicator", "Efficiency measure",
              "Program analysis", "Churn prediction")
)

kable(features, caption = "Engineered Features") %>%
  kable_styling(bootstrap_options = c("striped", "hover")) %>%
  scroll_box(height = "300px")
```

## Week 2: Exploratory Data Analysis

### Visualization Strategy
- **Tools Used**: Seaborn, Matplotlib, Python pandas
- **Team Approach**: Divided visualization tasks among team members
- **Graph Types**: Box plots, count plots, correlation heatmaps, pairplots

### Key Patterns Discovered

#### Seasonal Application Trends
- **Winter Peak**: ~5,000 applications (massive spike)
- **Moderate Seasons**: Fall, Spring, Summer (800-1,500 each)
- **Hypothesis**: Reduced outdoor activities drive winter applications

#### Demographic Insights  
- **Age Distribution**: 18-25 age group dominates participation
- **Geographic Concentration**: US, India, Nigeria lead in learner count
- **Commitment Patterns**: Younger learners show longer completion times

## Week 3: Churn Analysis & Predictive Modeling

### Churn Definition
Implemented **hybrid churn identification**:

1. **Explicit Churn**: Status = ['rejected', 'dropped out', 'withdraw']
2. **Implicit Churn**: 'team allocated' + opportunity ended >1 year ago

### Model Development Pipeline

```{r model-pipeline, echo=FALSE}
pipeline <- data.frame(
  Step = 1:6,
  Process = c("Feature Selection", "One-Hot Encoding", "Multicollinearity Removal", 
              "SMOTE Balancing", "Train-Test Split", "Model Training"),
  Description = c("Domain knowledge + EDA insights", "Categorical variables", 
                  "VIF analysis", "Address 70:30 imbalance", "80:20 stratified", 
                  "Multiple algorithms tested")
)

kable(pipeline, caption = "Machine Learning Pipeline") %>%
  kable_styling(bootstrap_options = c("striped", "hover"))
```

### Model Performance Comparison

```{r model-performance, echo=FALSE}
models <- data.frame(
  Model = c("Random Forest", "LightGBM", "XGBoost", "Logistic Regression"),
  Accuracy = c("85.2%", "84.8%", "84.1%", "82.3%"),
  ROC_AUC = c("0.87", "0.89", "0.86", "0.83"),
  F1_Score = c("0.85", "0.84", "0.84", "0.82"),
  Best_For = c("Overall Performance", "Ranking Ability", "Feature Importance", "Baseline")
)

kable(models, caption = "Model Performance Metrics") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed")) %>%
  row_spec(1, bold = TRUE, color = "white", background = "#3498db") %>%
  row_spec(2, bold = TRUE, color = "white", background = "#2ecc71")
```

## Week 4: Insights & Recommendations

### Critical Churn Drivers Identified

1. **Days from Signup to Application**: Longer delays → Higher churn
2. **Time Committed**: More engagement hours → Lower churn risk  
3. **Application Frequency**: Multiple applications affect retention
4. **Profile Completeness**: Incomplete profiles → Higher dropout risk

---

# Technical Implementation

## Code Repository

### 🔗 **Access Full Implementation**
- **Google Colab Notebook**: [View Complete Analysis](https://colab.research.google.com/drive/19yv0qSYL3wiTTWYVcTskJfqFsfGWKP-m#scrollTo=HrnY8Ll58pDT)
- **GitHub Repository**: Contains all source code, data, and documentation

### Key Technical Components

```python
# Core Libraries Used
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

# Data Preprocessing Pipeline
def preprocess_data(df):
    # Date parsing with 24-hour fix
    def fix_hour_24(dt_str):
        if pd.isna(dt_str) or ' 24:' not in dt_str:
            return dt_str
        date_part, time_part = dt_str.split(' 24:')
        new_time = '00:' + time_part
        # Convert to next day
        return pd.to_datetime(date_part) + pd.Timedelta(days=1)
    
    # Apply preprocessing steps
    date_cols = ['Learner SignUp DateTime', 'Apply Date', 'Opportunity End Date']
    for col in date_cols:
        df[col] = df[col].apply(fix_hour_24)
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df
```

## Model Architecture

### Feature Engineering Process

```r
# Feature Engineering in R (Example)
create_features <- function(data) {
  data %>%
    mutate(
      age = as.numeric(difftime(Sys.Date(), date_of_birth, units = "days")) / 365.25,
      signup_to_apply_days = as.numeric(difftime(apply_date, signup_datetime, units = "days")),
      opportunity_duration = as.numeric(difftime(end_date, start_date, units = "days")),
      season = case_when(
        month(signup_datetime) %in% c(12, 1, 2) ~ "Winter",
        month(signup_datetime) %in% c(3, 4, 5) ~ "Spring", 
        month(signup_datetime) %in% c(6, 7, 8) ~ "Summer",
        month(signup_datetime) %in% c(9, 10, 11) ~ "Fall"
      ),
      age_group = case_when(
        age < 18 ~ "Under 18",
        age <= 25 ~ "18-25",
        age <= 35 ~ "26-35", 
        age <= 50 ~ "36-50",
        TRUE ~ "50+"
      )
    )
}
```

### Model Evaluation Framework

```{r model-evaluation, echo=FALSE}
evaluation <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Confusion Matrix"),
  Purpose = c("Overall correctness", "Minimize false positives", "Catch all churners", 
              "Balance precision/recall", "Ranking quality", "Detailed breakdown"),
  Business_Impact = c("Model reliability", "Resource allocation", "Risk identification",
                      "Balanced performance", "Priority scoring", "Intervention targeting")
)

kable(evaluation, caption = "Model Evaluation Framework") %>%
  kable_styling(bootstrap_options = c("striped", "hover"))
```

---

# Business Impact & Recommendations

## Quantified Impact

### Projected Business Results

```{r impact-metrics, echo=FALSE}
impact <- data.frame(
  Metric = c("Additional Students Retained", "Revenue Protection", "Process Improvement", 
             "Churn Rate Reduction", "Application Processing Time"),
  Current_State = c("N/A", "At Risk", "7+ days", "70.32%", "Manual Process"),
  Projected_Improvement = c("1,200+ students", "Millions protected", "3-4 days", 
                            "Target: <50%", "Automated alerts"),
  Timeline = c("6-12 months", "Immediate", "3 months", "12 months", "1 month")
)

kable(impact, caption = "Projected Business Impact") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed")) %>%
  column_spec(3, bold = TRUE, color = "white", background = "#27ae60")
```

## Strategic Recommendations

### 1. Immediate Actions (0-3 months)
- **Deploy Early Warning System**: Implement real-time churn risk scoring
- **Seasonal Campaign Planning**: Prepare for winter/summer application spikes
- **Profile Completion Initiatives**: Mandatory fields and guided onboarding

### 2. Medium-term Strategy (3-12 months)  
- **Personalized Engagement**: Launch recommendation system
- **Predictive Intervention**: Automated outreach for at-risk learners
- **Process Optimization**: Reduce application-to-start delays

### 3. Long-term Vision (12+ months)
- **Advanced Analytics**: Real-time behavioral modeling
- **Platform Enhancement**: Gamification and milestone tracking  
- **Strategic Expansion**: Target high-potential demographics/geographies

## Recommendation System

### Hybrid Approach Implementation

```{r recommendation-system, echo=FALSE}
rec_system <- data.frame(
  Component = c("Content-Based Filtering", "Collaborative Filtering", "Hybrid Integration"),
  Method = c("Academic background + engagement history", "User similarity patterns", 
             "Weighted combination"),
  Benefits = c("Personalized matching", "Discovery of new opportunities", 
               "Balanced recommendations"),
  Implementation = c("Profile-based algorithms", "Matrix factorization", 
                     "Ensemble weighting")
)

kable(rec_system, caption = "Recommendation System Architecture") %>%
  kable_styling(bootstrap_options = c("striped", "hover"))
```

---

# Team Leadership & Collaboration

## Project Management Approach

As **team lead** for this 4-person data science team, I coordinated:

### Team Structure & Responsibilities
- **Ayad Aziz** (Lead): Overall strategy, modeling, and reporting
- **Jatin Chotoo**: Data preprocessing and feature engineering  
- **Jacob Cronce**: Advanced visualizations and feature analysis
- **Maryam Fatima**: Comprehensive EDA and pattern identification
- **Rachel D'souza**: Statistical analysis and validation

### Leadership Insights Gained
- **Virtual Team Management**: Coordinating across time zones and skill levels
- **Technical Mentorship**: Guiding team members through complex ML concepts
- **Strategic Decision Making**: Balancing technical accuracy with business needs
- **Cross-functional Communication**: Translating data insights for stakeholders

---

# Conclusion & Future Work

## Key Achievements

This project successfully transformed a **70% churn crisis** into a **data-driven retention framework** with measurable business impact:

✅ **Technical Excellence**: 85%+ model accuracy with production-ready pipeline  
✅ **Business Value**: Potential to retain 1,200+ students annually  
✅ **Operational Impact**: Streamlined processes and proactive intervention  
✅ **Strategic Foundation**: Analytics-driven decision making framework

## Future Enhancement Opportunities

1. **Real-time Processing**: Stream processing for immediate risk detection
2. **Advanced NLP**: Analyze feedback text for sentiment-based churn prediction  
3. **A/B Testing Framework**: Systematic intervention strategy optimization
4. **Mobile Analytics**: App engagement patterns for enhanced prediction
5. **External Data Integration**: Economic indicators, seasonal trends

## Technical Skills Demonstrated

- **Data Science Pipeline**: End-to-end project execution
- **Machine Learning**: Multiple algorithms with performance optimization
- **Feature Engineering**: Creative variable creation for predictive power
- **Statistical Analysis**: Rigorous hypothesis testing and validation  
- **Business Analytics**: Translating data insights into actionable strategies
- **Team Leadership**: Managing technical talent and project delivery

---

## Acknowledgments

**Special thanks to Excelerate** for providing this transformative learning opportunity and to my incredible team members for their dedication and collaborative excellence.

This project represents more than just a churn analysis—it's a comprehensive demonstration of how data science can drive meaningful business transformation when combined with strong leadership and collaborative teamwork.

---

*For complete code implementation and detailed analysis, visit the [Google Colab Notebook](https://colab.research.google.com/drive/19yv0qSYL3wiTTWYVcTskJfqFsfGWKP-m#scrollTo=HrnY8Ll58pDT)*
