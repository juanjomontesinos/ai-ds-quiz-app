# This app requires Streamlit. If you encounter a ModuleNotFoundError,
# please ensure you've installed Streamlit via `pip install streamlit`

try:
    import streamlit as st
except ModuleNotFoundError:
    raise ModuleNotFoundError("Streamlit is not installed. Run 'pip install streamlit' to use this app.")

import random

# Questions organized by session
questions_data = {
    "Session S1 (Introduction to DS, ML, AI)": [
        ("What is the primary goal of Artificial Intelligence (AI)?", ["To process structured data only", "To mimic human intelligence and perform tasks requiring human-like reasoning", "To automate data storage", "To optimize database queries"], "To mimic human intelligence and perform tasks requiring human-like reasoning"),
        ("Which event marked the formal birth of AI as a field?", ["Alan Turing’s 1950 paper", "The 1956 Dartmouth Conference", "IBM Deep Blue defeating Garry Kasparov", "John McCarthy’s invention of LISP"], "The 1956 Dartmouth Conference"),
        ("Which of the following is NOT a traditional area of AI?", ["Speech recognition", "Linear regression", "Natural Language Processing (NLP)", "Robotics"], "Linear regression"),
        ("What does the 'I' in FAIR data principles stand for?", ["Iterative", "Integrated", "Interoperable", "Intelligent"], "Interoperable"),
        ("Who proposed the Turing Test?", ["John McCarthy", "Alan Turing", "Arthur Samuel", "Frank Rosenblatt"], "Alan Turing"),
        ("Which term refers to systems that improve autonomously over time without human intervention?", ["Reinforcement Learning", "Supervised Learning", "Machine Learning", "Deep Learning"], "Machine Learning"),
        ("What is a key component of Data Science?", ["Hardware design", "Extracting insights from unstructured data", "Network configuration", "Database indexing"], "Extracting insights from unstructured data"),
        ("Which concept is emphasized by the phrase 'Garbage in, garbage out'?", ["Model optimization", "Data quality", "Algorithm speed", "Visualization"], "Data quality")
    ],
    "Session S2 (Methodology & Algorithms)": [
        ("Which phase of CRISP-DM involves defining project objectives?", ["Data Understanding", "Business Understanding", "Modeling", "Deployment"], "Business Understanding"),
        ("What task do Data Engineers primarily focus on?", ["Building predictive models", "Cleaning and organizing datasets", "Designing data architectures and ETL pipelines", "Creating data visualizations"], "Designing data architectures and ETL pipelines"),
        ("Which step is part of handling missing values?", ["Removing all rows with missing values", "Predicting missing values using regression", "Ignoring missing values in all cases", "Replacing missing values with random noise"], "Predicting missing values using regression"),
        ("What does a box plot visualize?", ["Correlation between variables", "Data distribution, quartiles, and outliers", "Time-series trends", "Cluster boundaries"], "Data distribution, quartiles, and outliers"),
        ("Which step in the Knowledge Discovery Process involves integrating and cleaning data?", ["Model creation", "Data Preparation", "Deployment", "Evaluation"], "Data Preparation"),
        ("What is the primary purpose of exploratory data analysis (EDA)?", ["Train a predictive model", "Discover patterns and anomalies in data", "Deploy a final model", "Optimize hyperparameters"], "Discover patterns and anomalies in data"),
        ("Which technique is used to split data into training and testing sets?", ["Clustering", "Holdout method", "PCA", "Association rules"], "Holdout method"),
        ("What does ETL stand for in data integration?", ["Extract, Transform, Load", "Evaluate, Train, Learn", "Encode, Test, Label", "Explore, Transform, Link"], "Extract, Transform, Load")
    ],
    "Session S3 (Tools in Data Science)": [
        ("Which language is most recommended for aspiring data scientists?", ["Java", "Python", "MATLAB", "Scala"], "Python"),
        ("What is a key feature of Dataiku DSS?", ["It only supports Python", "It enables collaborative workflows for data processing", "It is exclusive to cloud platforms", "It lacks visualization tools"], "It enables collaborative workflows for data processing"),
        ("In R, how do you create a data frame?", ["df <- matrix(...)", "df <- data.frame(...)", "df <- list(...)", "df <- array(...)"], "df <- data.frame(...)"),
        ("Which R function calculates the mean of a vector?", ["median()", "sd()", "mean()", "var()"], "mean()"),
        ("Which operator in R checks for inequality?", ["==", "!=", ">=", "<-"], "!="),
        ("What does the `str(df)` function in R provide?", ["Summary statistics of a data frame", "Structure and data types of a data frame", "Subset of rows meeting a condition", "Visualization of data distributions"], "Structure and data types of a data frame"),
        ("Which Dataiku feature allows collaborative workflows?", ["Visual recipes", "Jupyter notebooks", "Automated model deployment", "Real-time dashboards"], "Visual recipes"),
        ("How is a variable created in R?", ["var = 5", "var <- 5", "let var = 5", "variable(5)"], "var <- 5")
    ],
    "Session S4 (Supervised Algorithms)": [
        ("Which algorithm uses Gini Index or Information Gain for splitting nodes?", ["Logistic Regression", "Decision Trees", "K-means", "SVM"], "Decision Trees"),
        ("What does logistic regression predict?", ["A continuous value", "A probability between 0 and 1", "Cluster labels", "Text embeddings"], "A probability between 0 and 1"),
        ("Which metric is used to evaluate regression models?", ["Accuracy", "Mean Squared Error (MSE)", "F1-Score", "Precision"], "Mean Squared Error (MSE)"),
        ("What is a key disadvantage of black-box models?", ["High computational cost", "Lack of interpretability", "Overfitting", "Slow training time"], "Lack of interpretability"),
        ("What is the primary purpose of Support Vector Machines (SVM)?", ["Clustering unlabeled data", "Finding a hyperplane to separate classes", "Reducing dimensionality", "Generating association rules"], "Finding a hyperplane to separate classes"),
        ("Which algorithm classifies data based on the majority class of its k-nearest neighbors?", ["Decision Trees", "k-NN", "Naive Bayes", "Logistic Regression"], "k-NN"),
        ("What assumption does Naive Bayes make about features?", ["They follow a normal distribution", "They are mutually independent", "They are highly correlated", "They are categorical"], "They are mutually independent"),
        ("What is the goal of ensemble methods like Random Forest?", ["Reduce model interpretability", "Combine predictions to improve accuracy", "Simplify data preprocessing", "Speed up training time"], "Combine predictions to improve accuracy")
    ],
    "Session S5 (Model Evaluation)": [
        ("What does the ROC curve plot?", ["Precision vs. Recall", "True Positive Rate vs. False Positive Rate", "Accuracy vs. F1-Score", "MSE vs. R-squared"], "True Positive Rate vs. False Positive Rate"),
        ("Which metric is most suitable for imbalanced datasets?", ["Accuracy", "F1-Score", "Mean Absolute Error", "R-squared"], "F1-Score"),
        ("What does AUC represent?", ["The area under the precision-recall curve", "The probability of correct random classification", "The model’s ability to avoid overfitting", "The area under the ROC curve"], "The area under the ROC curve"),
        ("In a confusion matrix, what does 'False Negative' mean?", ["Correctly predicted negative", "Incorrectly predicted negative", "Correctly predicted positive", "Incorrectly predicted positive"], "Incorrectly predicted negative"),
        ("What does cross-validation help prevent?", ["Underfitting", "Overfitting", "Feature engineering", "Data collection bias"], "Overfitting"),
        ("If Precision = 0.8 and Recall = 0.5, what is the F1-Score?", ["0.65", "0.61", "0.60", "0.55"], "0.61"),
        ("Which metric is used in a cost-sensitive learning scenario?", ["Accuracy", "ROC-AUC", "Confusion Matrix", "Cost Matrix"], "Cost Matrix"),
        ("What does Specificity measure?", ["True Positive Rate", "True Negative Rate", "False Positive Rate", "False Negative Rate"], "True Negative Rate")
    ],
    "Session S6 (Unsupervised Algorithms)": [
        ("Which algorithm requires specifying the number of clusters (k) beforehand?", ["Hierarchical Clustering", "K-means", "DBSCAN", "PCA"], "K-means"),
        ("What is the primary goal of PCA?", ["Classification", "Dimensionality reduction", "Anomaly detection", "Feature engineering"], "Dimensionality reduction"),
        ("Which technique discovers associations between items in transactional data?", ["K-nearest neighbors", "Apriori algorithm", "Linear regression", "SVM"], "Apriori algorithm"),
        ("What does a dendrogram visualize?", ["Correlation matrix", "Hierarchical clustering results", "Decision tree splits", "Feature importance"], "Hierarchical clustering results"),
        ("Which algorithm uses expectation-maximization to handle hidden variables?", ["K-means", "EM", "PCA", "Apriori"], "EM"),
        ("What is a key application of Self-Organizing Maps (SOM)?", ["Regression analysis", "Dimensionality reduction and visualization", "Time-series forecasting", "Text translation"], "Dimensionality reduction and visualization"),
        ("Which metric measures linear relationships between variables?", ["Gini Index", "Pearson Correlation", "Entropy", "Silhouette Score"], "Pearson Correlation"),
        ("What does the Apriori algorithm optimize for in market basket analysis?", ["Support and Confidence", "MSE and R-squared", "Precision and Recall", "F1-Score"], "Support and Confidence")
    ]
}

# Flatten all questions into one list for random test generation
all_questions = [(session, q[0], q[1], q[2]) for session, qs in questions_data.items() for q in qs]

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'landing'
if 'current_q' not in st.session_state:
    st.session_state.current_q = 0
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'questions' not in st.session_state:
    st.session_state.questions = []

# Navigation
if st.session_state.page == 'landing':
    st.title("📘 AI & Data Science Quiz App")
    if st.button("Start Test"):
        st.session_state.questions = random.sample(all_questions, 20)
        st.session_state.current_q = 0
        st.session_state.score = 0
        st.session_state.page = 'quiz'
        st.rerun()
    if st.button("View All Questions"):
        st.session_state.page = 'view_all'
        st.rerun()

elif st.session_state.page == 'quiz':
    q_info = st.session_state.questions[st.session_state.current_q]
    session, question, options, correct = q_info
    st.markdown(f"**{session}**")
    st.subheader(f"Question {st.session_state.current_q + 1}: {question}")

    selected = st.radio("Choose an answer:", options, key=f"q{st.session_state.current_q}")

    if st.button("Submit Answer"):
        if selected == correct:
            st.success("✅ Correct!")
            st.session_state.score += 1
        else:
            st.error("❌ Incorrect.")
            st.markdown(f"**Correct Answer:** {correct}")
        if st.session_state.current_q < 19:
            if st.button("Next Question"):
                st.session_state.current_q += 1
                st.rerun()
        else:
            if st.button("See Results"):
                st.session_state.page = 'results'
                st.rerun()

    # Add Exit Test button
    if st.button("Exit Test"):
        st.session_state.page = 'landing'
        st.session_state.questions = []
        st.session_state.current_q = 0
        st.session_state.score = 0
        st.rerun()

elif st.session_state.page == 'results':
    st.title("🎓 Quiz Completed!")
    st.metric(label="Your Score", value=f"{st.session_state.score}/20")
    if st.button("Start New Test"):
        st.session_state.page = 'landing'
        st.rerun()
    if st.button("Return to Home"):
        st.session_state.page = 'landing'
        st.rerun()

elif st.session_state.page == 'view_all':
    st.title("📚 All Questions & Correct Answers")
    for session, qs in questions_data.items():
        st.subheader(session)
        for i, (q, opts, ans) in enumerate(qs):
            st.markdown(f"**Q{i+1}:** {q}")
            st.markdown(f"✅ **Correct Answer:** {ans}")
            st.markdown("---")
    if st.button("Back to Home"):
        st.session_state.page = 'landing'
        st.rerun()
