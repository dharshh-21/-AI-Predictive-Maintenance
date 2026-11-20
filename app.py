import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# PAGE CONFIGURATION
st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

# LOAD MODEL AND DATASET
bundle = joblib.load("models/predictive_model.pkl")

model = bundle["model"]
threshold = bundle["threshold"]
feature_columns = bundle["feature_columns"]

data = pd.read_csv("data/ai4i2020.csv")

# TITLE
st.title("⚙️ AI-Based Predictive Maintenance System")

st.write(
    "Machine Failure Prediction using Random Forest Machine Learning"
)

st.info(
    "This prototype uses the AI4I 2020 simulated industrial dataset. "
    "It does not monitor live factory machines."
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a section",
    ["Home", "Machine Failure Prediction"]
)

# ---------------- HOME PAGE ----------------

if page == "Home":

    st.header("Project Overview")

    st.write("""
    This system analyzes machine operating parameters
    and predicts potential machine failures using
    a Random Forest classification model.
    """)

    # PROJECT METRICS

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Dataset Records", "10,000")
    col2.metric("Accuracy", "97.30%")
    col3.metric("Precision", "57.61%")
    col4.metric("Recall", "77.94%")

    st.divider()

    # FAILURE DISTRIBUTION

    st.subheader("1. Machine Failure Distribution")

    failure_counts = data["Machine failure"].value_counts()

    chart_data = pd.DataFrame({
        "Machine Status": ["No Failure", "Failure"],
        "Records": [
            int(failure_counts.get(0, 0)),
            int(failure_counts.get(1, 0))
        ]
    })

    st.bar_chart(
        chart_data.set_index("Machine Status")
    )

    st.write(
        "The dataset contains significantly more normal "
        "machine records than failure records."
    )

    st.divider()

    # CORRELATION HEATMAP

    st.subheader("2. Machine Parameter Correlation")

    selected_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure"
    ]

    correlation = data[selected_columns].corr()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    ax.set_title("Machine Parameter Correlation Heatmap")

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # FEATURE IMPORTANCE

    st.subheader("3. Random Forest Feature Importance")

    importance = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        importance["Feature"],
        importance["Importance"]
    )

    ax.set_title("Feature Importance")

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    # MODEL PERFORMANCE

    st.subheader("4. Model Performance")

    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Score": [
            97.30,
            57.61,
            77.94,
            66.25
        ]
    })

    st.dataframe(metrics, hide_index=True)

    st.write("Selected Classification Threshold:", round(threshold, 2))

    st.divider()

    # CONFUSION MATRIX

    st.subheader("5. Confusion Matrix")

    cm = [
        [1893, 39],
        [15, 53]
    ]

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Failure", "Failure"],
        yticklabels=["No Failure", "Failure"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Random Forest Confusion Matrix")

    st.pyplot(fig)
    plt.close(fig)


# ---------------- PREDICTION PAGE ----------------

elif page == "Machine Failure Prediction":

    st.header("Machine Failure Prediction")

    st.write("Enter the machine operating parameters.")

    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )

    col1, col2 = st.columns(2)

    with col1:

        air_temp = st.number_input(
            "Air Temperature (K)",
            250.0, 350.0, 300.0
        )

        process_temp = st.number_input(
            "Process Temperature (K)",
            250.0, 400.0, 310.0
        )

        rotational_speed = st.number_input(
            "Rotational Speed (rpm)",
            0, 5000, 1500
        )

    with col2:

        torque = st.number_input(
            "Torque (Nm)",
            0.0, 150.0, 40.0
        )

        tool_wear = st.number_input(
            "Tool Wear (min)",
            0, 300, 100
        )

    if st.button("Predict Machine Failure"):

        input_data = pd.DataFrame([{
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "Type": machine_type
        }])

        input_data = pd.get_dummies(
            input_data,
            columns=["Type"],
            dtype=int
        )

        input_data = input_data.reindex(
            columns=feature_columns,
            fill_value=0
        )

        probability = model.predict_proba(input_data)[0][1]

        prediction = int(probability >= threshold)

        st.divider()

        st.subheader("Prediction Result")

        st.metric(
            "Model Failure Score",
            f"{probability * 100:.2f}%"
        )

        if prediction == 1:

            st.error("⚠️ Potential Machine Failure Detected")

        else:

            st.success("✅ No Machine Failure Predicted")

        st.caption(
            f"Classification threshold: {threshold:.2f}. "
            "The displayed score is not a calibrated real-world "
            "failure probability."
        )