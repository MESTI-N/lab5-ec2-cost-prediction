
# Part 1 - Step 1: Setup and Install Dependencies

import streamlit as st
import pandas as pd
import plotly.express as px


# Part 2 - Step 1: Setup and Install Dependencies

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Streamlit Page Setup

st.set_page_config(
    page_title="EC2 Cost Dashboard",
    page_icon="☁️",
    layout="wide"
)

st.title("EC2 Cost Analysis and Prediction Dashboard")
st.write("Analyze Amazon EC2 instance costs and predict On-Demand pricing.")


# Part 1 - Step 2: Load and Explore the Dataset

@st.cache_data
def load_data():
    data = pd.read_csv("ec2dataset.csv")
    return data


data = load_data()


# Part 1 - Step 3: Clean the Data

cost_columns = [
    "On Demand",
    "Linux Reserved cost",
    "Linux Spot Minimum cost",
    "Windows On Demand cost",
    "Windows Reserved cost"
]

for column in cost_columns:
    data[column] = pd.to_numeric(
        data[column]
        .astype(str)
        .str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce"
    )


# Part 2 - Step 2: Load and Clean the Data

# Use the cleaned cost data from Part 1


# Part 2 - Step 3: Feature Engineering

# Clean memory
data["Memory_GiB"] = pd.to_numeric(
    data["Instance Memory"]
    .astype(str)
    .str.extract(r"([\d.]+)")[0],
    errors="coerce"
)

# Clean vCPUs
data["vCPU_Count"] = pd.to_numeric(
    data["vCPUs"]
    .astype(str)
    .str.extract(r"(\d+)")[0],
    errors="coerce"
)


# Part 2 - Step 4: Handle Missing Data

model_data = data.dropna(
    subset=[
        "On Demand",
        "Memory_GiB",
        "vCPU_Count"
    ]
).copy()


# Part 2 - Step 5: Split the Data

X = model_data[["Memory_GiB", "vCPU_Count"]]
y = model_data["On Demand"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Part 2 - Step 6: Train a Linear Regression Model

model = LinearRegression()
model.fit(X_train, y_train)


# Part 2 - Step 7: Evaluate the Model

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)


# Create Dashboard Tabs

tab1, tab2, tab3 = st.tabs([
    "Dataset Overview",
    "EC2 Cost Analysis",
    "Cost Prediction"
])


# TAB 1 - Dataset Overview

with tab1:

    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Instances", len(data))
    col2.metric("Total Columns", len(data.columns))
    col3.metric("Missing Values", int(data.isna().sum().sum()))

    st.subheader("Dataset Preview")
    st.dataframe(data.head(20), use_container_width=True)

    st.subheader("Dataset Information")

    info = pd.DataFrame({
        "Column": data.columns,
        "Data Type": data.dtypes.astype(str).values,
        "Non-Null Count": data.notna().sum().values,
        "Missing Values": data.isna().sum().values
    })

    st.dataframe(info, use_container_width=True)

    st.subheader("Dataset Shape")
    st.write(f"Rows: {data.shape[0]}")
    st.write(f"Columns: {data.shape[1]}")


# TAB 2 - PART 1: EC2 Cost Analysis

with tab2:

    st.header("Part 1 - Analyze EC2 Instance Costs")


    # Part 1 - Step 4: Perform Summary Analysis

    st.subheader("Step 4 - Cost Summary")

    cost_summary = data[cost_columns].describe()

    st.dataframe(
        cost_summary,
        use_container_width=True
    )


    # Part 1 - Step 5: Visualize the Data

    st.subheader("Step 5 - Cost Comparison")

    cost_long = data[cost_columns].melt(
        var_name="Pricing Type",
        value_name="Hourly Cost"
    )

    fig1 = px.box(
        cost_long,
        x="Pricing Type",
        y="Hourly Cost",
        title="EC2 Hourly Cost Comparison"
    )

    st.plotly_chart(fig1, use_container_width=True)


    # Part 1 - Step 6: Identify Outliers

    st.subheader("Step 6 - Cost Outliers")

    q1 = data["On Demand"].quantile(0.25)
    q3 = data["On Demand"].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = data[
        (data["On Demand"] < lower_bound) |
        (data["On Demand"] > upper_bound)
    ]

    st.write(f"Number of Outliers: {len(outliers)}")

    st.dataframe(
        outliers[
            ["Name", "API Name", "On Demand"]
        ],
        use_container_width=True
    )


    # Part 1 - Step 7: Conclusion and Further Analysis

    st.subheader("Step 7 - Cost Comparison")

    cost_comparison = data[
        ["Name", "API Name", "On Demand", "Linux Reserved cost"]
    ].dropna()

    cost_comparison = cost_comparison.sort_values(
        "On Demand"
    )

    st.write("10 Lowest-Cost Instances")

    st.dataframe(
        cost_comparison.head(10),
        use_container_width=True
    )

    # Calculate savings
    cost_comparison["Savings"] = (
        cost_comparison["On Demand"] -
        cost_comparison["Linux Reserved cost"]
    )

    st.subheader("On-Demand vs Reserved Costs")

    fig2 = px.scatter(
        cost_comparison,
        x="On Demand",
        y="Linux Reserved cost",
        hover_name="API Name",
        title="On-Demand vs Reserved Hourly Costs"
    )

    st.plotly_chart(fig2, use_container_width=True)


    # Part 1 - Step 8: Filter and Compare Instance Families

    st.subheader("Step 8 - Instance Family Comparison")

    # Get instance family from API Name
    data["Family"] = (
        data["API Name"]
        .astype(str)
        .str.split(".")
        .str[0]
        .str.upper()
    )

    families = sorted(
        data["Family"].dropna().unique().tolist()
    )

    selected_families = st.multiselect(
        "Select Instance Families",
        families,
        default=[
            family for family in ["T2", "T3"]
            if family in families
        ]
    )

    family_data = data[
        data["Family"].isin(selected_families)
    ]

    if not family_data.empty:

        st.subheader("Selected Family Data")

        st.dataframe(
            family_data,
            use_container_width=True
        )

        st.subheader("Family Cost Summary")

        family_summary = family_data.groupby(
            "Family"
        )[cost_columns].mean()

        st.dataframe(
            family_summary,
            use_container_width=True
        )

        # Compare cost distribution
        fig3 = px.box(
            family_data,
            x="Family",
            y="On Demand",
            title="On-Demand Cost by Instance Family"
        )

        st.plotly_chart(fig3, use_container_width=True)

        # Compare On-Demand and Reserved
        st.subheader("Family Pricing Comparison")

        family_comparison = family_data[
            ["Family", "On Demand", "Linux Reserved cost"]
        ].melt(
            id_vars="Family",
            var_name="Pricing Type",
            value_name="Hourly Cost"
        )

        fig4 = px.box(
            family_comparison,
            x="Family",
            y="Hourly Cost",
            color="Pricing Type",
            title="On-Demand vs Reserved by Family"
        )

        st.plotly_chart(fig4, use_container_width=True)

    else:
        st.info("Select an instance family to view results.")


# TAB 3 - PART 2: Cost Prediction

with tab3:

    st.header("Part 2 - Predicting EC2 Instance Costs")

    st.write(
        "This model predicts Linux On-Demand hourly cost "
        "using instance memory and vCPUs."
    )


    # Part 2 - Step 3: Feature Engineering

    st.subheader("Step 3 - Features Used")

    st.write("The model uses these two features:")

    st.write("- Instance Memory (GiB)")
    st.write("- Number of vCPUs")

    st.dataframe(
        model_data[
            ["Name", "Memory_GiB", "vCPU_Count", "On Demand"]
        ].head(10),
        use_container_width=True
    )


    # Part 2 - Step 4: Handle Missing Data

    st.subheader("Step 4 - Cleaned Dataset")

    col1, col2 = st.columns(2)

    col1.metric("Original Rows", len(data))
    col2.metric("Cleaned Rows", len(model_data))


    # Part 2 - Step 5: Split the Data

    st.subheader("Step 5 - Training and Testing Data")

    col1, col2 = st.columns(2)

    col1.metric("Training Rows", len(X_train))
    col2.metric("Testing Rows", len(X_test))

    st.write("Training: 80% | Testing: 20%")


    # Part 2 - Step 6: Train a Linear Regression Model

    st.subheader("Step 6 - Linear Regression Model")

    st.write("Model: Linear Regression")

    st.write(f"Intercept: {model.intercept_:.4f}")

    coefficients = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    })

    st.dataframe(
        coefficients,
        use_container_width=True
    )


    # Part 2 - Step 7: Evaluate the Model

    st.subheader("Step 7 - Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("MAE", f"${mae:.4f}")
    col2.metric("MSE", f"{mse:.4f}")
    col3.metric("RMSE", f"${rmse:.4f}")
    col4.metric("R² Score", f"{r2:.4f}")

    st.info(
        "Lower MAE, MSE, and RMSE values mean smaller "
        "prediction errors. R² measures model fit."
    )


    # Part 2 - Step 8: Visualize the Results

    st.subheader("Step 8 - Actual vs Predicted Costs")

    results = pd.DataFrame({
        "Actual Cost": y_test.to_numpy(),
        "Predicted Cost": y_pred
    })

    fig5 = px.scatter(
        results,
        x="Actual Cost",
        y="Predicted Cost",
        title="Actual vs Predicted EC2 Costs"
    )

    # Perfect prediction line
    min_cost = min(
        results["Actual Cost"].min(),
        results["Predicted Cost"].min()
    )

    max_cost = max(
        results["Actual Cost"].max(),
        results["Predicted Cost"].max()
    )

    fig5.add_shape(
        type="line",
        x0=min_cost,
        y0=min_cost,
        x1=max_cost,
        y1=max_cost,
        line=dict(color="red", dash="dash")
    )

    st.plotly_chart(fig5, use_container_width=True)

    st.caption(
        "The red dashed line represents perfect predictions."
    )


    # Part 2 - Step 9: Make Predictions

    st.subheader("Step 9 - Predict EC2 Cost")

    col1, col2 = st.columns(2)

    with col1:
        memory = st.number_input(
            "Instance Memory (GiB)",
            min_value=0.5,
            max_value=float(model_data["Memory_GiB"].max()),
            value=64.0,
            step=0.5
        )

    with col2:
        vcpus = st.number_input(
            "Number of vCPUs",
            min_value=1,
            max_value=int(model_data["vCPU_Count"].max()),
            value=8,
            step=1
        )

    if st.button("Predict Cost"):

        new_instance = pd.DataFrame({
            "Memory_GiB": [memory],
            "vCPU_Count": [vcpus]
        })

        prediction = model.predict(new_instance)[0]

        st.success(
            f"Predicted On-Demand Cost: "
            f"${prediction:.4f} per hour"
        )

        # Linear regression can predict negative costs
        if prediction < 0:
            st.warning(
                "The model predicted a negative cost. "
                "This is a limitation of linear regression."
            )

        st.caption(
            "This is a model estimate, not an official "
            "AWS pricing quote."
        )
