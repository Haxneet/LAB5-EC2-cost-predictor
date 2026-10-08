# Lab 5: Predicting EC2 Cost
# Streamlit dashboard that predicts the Linux On Demand hourly cost of an EC2 instance
# from its memory (GiB) and vCPUs, using linear regression.
# Run locally with:  streamlit run app.py

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="EC2 Cost Predictor", layout="wide")

COST_COLUMNS = ["On Demand", "Linux Reserved cost", "Linux Spot Minimum cost",
                "Windows On Demand cost", "Windows Reserved cost"]
FEATURES = ["Instance Memory", "vCPUs"]


# ---------------------------------------------------------------
# Load and clean the data (same steps as the activity)
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("ec2dataset.csv")

    # "$0.0116 hourly" -> 0.0116   ("unavailable" becomes NaN)
    for column in COST_COLUMNS:
        data[column] = pd.to_numeric(data[column].str.replace("[$, hourly]", "", regex=True),
                                     errors="coerce")

    # "0.5 GiB" -> 0.5   and   "2 vCPUs for a 1h 12m burst" -> 2
    data["Instance Memory"] = pd.to_numeric(
        data["Instance Memory"].str.replace(" GiB", "").str.replace(",", ""), errors="coerce")
    data["vCPUs"] = pd.to_numeric(data["vCPUs"].str.extract(r"(\d+)", expand=False))

    return data.dropna(subset=["On Demand"] + FEATURES)


# ---------------------------------------------------------------
# Train the model
# ---------------------------------------------------------------
@st.cache_resource
def train_model(exclude_outliers):
    data = load_data()

    # Optional: drop very expensive instances (IQR method) so they don't distort the model
    if exclude_outliers:
        q1, q3 = data["On Demand"].quantile([0.25, 0.75])
        data = data[data["On Demand"] <= q3 + 1.5 * (q3 - q1)]

    X = data[FEATURES]
    y = data["On Demand"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
        "R2": r2_score(y_test, y_pred),
    }
    test_results = X_test.assign(Name=data.loc[X_test.index, "Name"],
                                 Actual=y_test, Predicted=y_pred)
    return model, metrics, test_results, len(data)


data = load_data()

# ---------------------------------------------------------------
# Sidebar: inputs
# ---------------------------------------------------------------
st.sidebar.header("Instance configuration")
memory = st.sidebar.number_input("Memory (GiB)", min_value=0.5, max_value=4096.0,
                                 value=4.0, step=0.5)
vcpus = st.sidebar.number_input("vCPUs", min_value=1, max_value=448, value=2, step=1)

st.sidebar.header("Model")
exclude_outliers = st.sidebar.checkbox("Exclude outlier instances (IQR)", value=True,
                                       help="Removes very expensive instances, such as 24 TB "
                                            "memory machines, before training.")

model, metrics, test_results, n_rows = train_model(exclude_outliers)

new_instance = pd.DataFrame([[memory, vcpus]], columns=FEATURES)
predicted = float(model.predict(new_instance)[0])

# ---------------------------------------------------------------
# Header and prediction
# ---------------------------------------------------------------
st.title("EC2 Cost Predictor")
st.caption("Predicts the Linux On Demand hourly cost of an Amazon EC2 instance from its "
           "memory and vCPUs, using linear regression.")

c1, c2, c3 = st.columns(3)
c1.metric("Predicted cost per hour", f"${max(predicted, 0):.4f}")
c2.metric("Predicted cost per month (730 hrs)", f"${max(predicted, 0) * 730:,.2f}")
c3.metric("Configuration", f"{memory:g} GiB, {vcpus} vCPUs")

if predicted < 0:
    st.warning(f"The model predicts a negative cost (${predicted:.4f}), so it is shown as $0. "
               "Linear regression is pulled toward the very expensive instances. "
               "Tick 'Exclude outlier instances' in the sidebar for a more realistic estimate.")

tab1, tab2, tab3, tab4 = st.tabs(["Similar instances", "Model performance",
                                  "Cost analysis", "Data"])

# ---------------------------------------------------------------
# Tab 1: real instances closest to the chosen configuration
# ---------------------------------------------------------------
with tab1:
    st.subheader("Real instances closest to this configuration")
    st.write("Use these to check the prediction against actual AWS prices.")
    distance = ((data["Instance Memory"] - memory) / memory).abs() + \
               ((data["vCPUs"] - vcpus) / vcpus).abs()
    similar = data.assign(Distance=distance).nsmallest(10, "Distance")
    st.dataframe(similar[["Name", "API Name", "Instance Memory", "vCPUs", "On Demand",
                          "Linux Reserved cost", "Linux Spot Minimum cost"]],
                 hide_index=True, use_container_width=True)

# ---------------------------------------------------------------
# Tab 2: model performance
# ---------------------------------------------------------------
with tab2:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R-squared", f"{metrics['R2']:.3f}")
    m2.metric("MAE", f"${metrics['MAE']:.4f}")
    m3.metric("RMSE", f"${metrics['RMSE']:.4f}")
    m4.metric("Instances used", n_rows)

    st.markdown(f"**Model:** cost = {model.intercept_:.4f} "
                f"+ {model.coef_[0]:.5f} x memory (GiB) "
                f"+ {model.coef_[1]:.5f} x vCPUs")

    fig = px.scatter(test_results, x="Actual", y="Predicted", hover_name="Name",
                     hover_data=FEATURES, opacity=0.7,
                     title="Actual vs Predicted On Demand Cost (test set)",
                     labels={"Actual": "Actual cost (USD/hr)",
                             "Predicted": "Predicted cost (USD/hr)"})
    lo = min(test_results["Actual"].min(), test_results["Predicted"].min())
    hi = max(test_results["Actual"].max(), test_results["Predicted"].max())
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                  line=dict(color="red", dash="dash"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Points on the red line are perfect predictions.")

# ---------------------------------------------------------------
# Tab 3: cost analysis (Part 1 of the activity)
# ---------------------------------------------------------------
with tab3:
    long = data.melt(value_vars=COST_COLUMNS, var_name="Pricing option",
                     value_name="Cost (USD/hr)").dropna()
    st.plotly_chart(px.box(long, x="Pricing option", y="Cost (USD/hr)", log_y=True,
                           title="Cost by pricing option (log scale)"),
                    use_container_width=True)

    reserved = (1 - data["Linux Reserved cost"] / data["On Demand"]).mean() * 100
    spot = (1 - data["Linux Spot Minimum cost"] / data["On Demand"]).mean() * 100
    s1, s2 = st.columns(2)
    s1.metric("Average Reserved saving vs On Demand", f"{reserved:.0f}%")
    s2.metric("Average Spot saving vs On Demand", f"{spot:.0f}%")

    st.subheader("Compare instance families")
    families = sorted(data["Name"].str.split().str[0].unique())
    chosen = st.multiselect("Families", families, default=["T2", "T3"])
    fam = data[data["Name"].str.split().str[0].isin(chosen)]
    if fam.empty:
        st.info("Pick at least one family.")
    else:
        fam = fam.assign(Family=fam["Name"].str.split().str[0])
        st.plotly_chart(px.bar(fam.sort_values("On Demand"), x="Name",
                               y=["On Demand", "Linux Reserved cost"], barmode="group",
                               title="On Demand vs Reserved cost",
                               labels={"value": "Cost (USD/hr)", "variable": ""}),
                        use_container_width=True)

# ---------------------------------------------------------------
# Tab 4: raw data
# ---------------------------------------------------------------
with tab4:
    st.write(f"{len(data)} instances after cleaning.")
    st.dataframe(data, hide_index=True, use_container_width=True)
