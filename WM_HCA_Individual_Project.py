import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import hydralit_components as hc
import plotly.graph_objects as go
import plotly.express as px
import graphviz as graph

# import numpy as np
# import time
# import seaborn as sns
# import requests
# import inspect
# from streamlit_lottie import st_lottie
# from ydata_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report
# from numerize import numerize
# import joblib
# from itertools import chain
# import folium
# import statsmodels.api as sm


# Set Page Icon,Title, and Layout
st.set_page_config(
    layout="wide",
    page_icon="pics/smok_animation.jpg",
    page_title="Smoking Trend Analysis",
)


# Navigation Bar Design
menu_data = [
    {"label": "Home", "icon": "bi bi-house"},
    {"label": "Risk Factors", "icon": "bi bi-clipboard-data"},
    {"label": "Smoking Trends", "icon": "bi bi-graph-up-arrow"},
    {"label": "Gender Dispartiy & Income", "icon": "bi bi-file-person"},
    {"label": "Recommendations", "icon": "fa fa-brain"},
]


# Set the Navigation Bar
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    sticky_mode="sticky",
    sticky_nav=False,
    hide_streamlit_markers=False,
    override_theme={
        "txc_inactive": "white",
        "menu_background": "#800000",  # Bordeuax color
        "txc_active": "#800000",  # Bordeuax color
        "option_active": "white",
    },
)


def risk_factors():
    deathsby = pd.read_csv("number-of-deaths-by-risk-factor.csv")

    # renaming first row
    # Read the data into a DataFrame
    data = deathsby

    # Extract the risk factors from column names
    risk_factors = [
        title.split("Risk:")[1].split(" - Sex:")[0].strip()
        for title in data.columns[3:]
    ]

    # Update the column names with risk factors
    data.columns = data.columns[:3].tolist() + risk_factors

    # Display the modified DataFrame
    data.head()

    # Set the max_rows parameter to display fewer options in the dropdown
    pd.set_option("display.max_rows", 10)

    # Read the data into a DataFrame
    # Calculate the sum of deaths by category for each year
    sum_deaths = data.groupby(["Year", "Entity"]).sum().reset_index()

    # Get the unique entities in the data
    entities = sum_deaths["Entity"].unique()

    # Create a columns layout for slider and dropdown
    col1, col2 = st.columns([3, 1])

    # header
    col1.markdown(
        """
        <div style="background-color: #800020; padding: 10px">
            <h2 style="color: white; font-weight: bold">What are the major risk factors causing deaths in different parts of the world?</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Create a dropdown for selecting the entity
    default_entity = "World"
    entity_dropdown = col1.selectbox(
        "Entity:",
        ["World", "Middle East & North Africa (WB)", "Lebanon"] + entities.tolist(),
        index=0,
        key="entity_dropdown",
    )

    # Convert the min and max year values to integers
    min_year = int(sum_deaths["Year"].min())
    max_year = int(sum_deaths["Year"].max())

    # Create a slider for selecting the year
    year_slider = col1.slider(
        "Year:", min_value=min_year, max_value=max_year, value=2019
    )

    col2.image("pics/Smok tray 2.jpg")

    # Define a function to update the plot based on the selected year and entity
    def update_plot(year, entity):
        deaths_year_entity = sum_deaths[
            (sum_deaths["Year"] == year) & (sum_deaths["Entity"] == entity)
        ].squeeze()
        deaths_year_entity = deaths_year_entity.drop(["Year", "Entity"])

        # Sort the values in ascending order
        deaths_year_entity_sorted = deaths_year_entity.sort_values(ascending=True)

        # Get the top 5 categories
        top_categories = deaths_year_entity_sorted.tail(5)
        top_categories_names = top_categories.index.tolist()

        plt.figure(figsize=(12, 6))
        bars = plt.barh(
            deaths_year_entity_sorted.index,
            deaths_year_entity_sorted.values,
            color="skyblue",
        )
        plt.xlabel("Number of Deaths")
        plt.ylabel("Category")
        plt.title("Number of Deaths by Cause for {}".format(entity))

        # Add value labels to each bar
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width,
                bar.get_y() + bar.get_height() / 2,
                f"{width:,}",
                ha="left",
                va="center",
            )

        # Split the screen between the graph and the top risk factors
        col_graph, col_top = st.columns([3, 1])

        # Display the graph in the left column
        with col_graph:
            st.pyplot(plt)

        # Clear previous top risk factors
        col_top.empty()

        # Display top categories in the right column
        with col_top:
            st.markdown(
                "<h3 style='font-size: 50px;'>Top Risk Factors:</h3>",
                unsafe_allow_html=True,
            )
            for i, category in enumerate(reversed(top_categories_names), 1):
                st.markdown(
                    f'<p style="font-size: 35px;">{i}. {category}</p>',
                    unsafe_allow_html=True,
                )

    # Update the plot based on the selected year and entity
    update_plot(year_slider, entity_dropdown)


def table_risk_comparison():
    st.header("")

    st.markdown(
        """
        <div style="background-color: #800020; padding: 10px">
            <h2 style="color: white; font-weight: bold">Can we see similar trends between the World, Middle East & North Africa and Lebanon?</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Define the data for the table
    data = {
        "World": [
            "High systolic blood pressure",
            "Smoking",
            "Air pollution",
            "High fasting plasma glucose",
            "High body-mass index",
            "Outdoor air pollution – OWID",
        ],
        "Middle East and North Africa": [
            "High systolic blood pressure",
            "High body-mass index",
            "High fasting plasma glucose",
            "Air pollution",
            "Outdoor air pollution – OWID",
            "Smoking",
        ],
        "Lebanon": [
            "High systolic blood pressure",
            "Smoking",
            "High fasting plasma glucose",
            "High body-mass index",
            "Outdoor air pollution – OWID",
            "Air pollution",
        ],
    }

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Reset the index starting from 1
    df.index += 1

    # Define a function to style the cells based on the value
    def style_cell(cell):
        style = "background-color: #FFCCCC" if cell == "Smoking" else ""
        return style

    # Apply the cell styling
    styled_df = (
        df.style.applymap(style_cell)
        .set_properties(**{"text-align": "left"})
        .set_caption("Top 6 Risk Factors")
    )

    # Define the CSS properties for the table
    table_style = [
        {
            "selector": "table",
            "props": [("font-size", "40px"), ("background-color", "#F8F8F8")],
        },
        {"selector": "th", "props": [("font-size", "40px"), ("font-weight", "bold")]},
        {"selector": "td", "props": [("font-size", "30px")]},
    ]

    # Set the table styles
    styled_df.set_table_styles(table_style)

    # Display the styled DataFrame
    st.table(styled_df)


def table_top10_death(col):
    ### top 10 by percentage deaths

    # Load the dataset
    data = pd.read_csv("updated_combined_data.csv")

    # Filter the data for the year 2019
    data_2019 = data[data["Year"] == 2019]

    # Sort the data by percentage in descending order and select the top 10 countries
    top_10_countries = data_2019.nlargest(10, "(Percent)")

    # Reset the index and add 1 to start counting from 1
    top_10_countries.reset_index(drop=True, inplace=True)
    top_10_countries.index += 1

    # Define the CSS properties for the table
    table_style = [
        {
            "selector": "table",
            "props": [("font-size", "40px"), ("background-color", "#F8F8F8")],
        },
        {"selector": "th", "props": [("font-size", "40px"), ("font-weight", "bold")]},
        {"selector": "td", "props": [("font-size", "40px"), ("padding", "10px")]},
    ]

    # Apply the cell styling
    styled_df = top_10_countries.style.set_table_styles(table_style)

    # Display the styled DataFrame
    col.write(styled_df, unsafe_allow_html=True)
    pass


def map_deathby_smok(col):
    ## working map  size variation and color

    # Load the dataset
    mapdata = pd.read_csv("updated_combined_data.csv")

    # Create a dropdown for country selection
    country_options = ["All Countries"] + mapdata["Entity"].unique().tolist()
    selected_country = st.selectbox("Select a country", country_options)

    # Create a range slider for year selection
    year_range = st.slider("Select year range", 1990, 2019, (1990, 2019))

    # Create radio buttons for the two options
    option = st.radio(
        "Select an option", ["Smoking per 100,000 people", "Smoking by %"]
    )

    # Filter the dataset based on user selection
    if selected_country == "All Countries":
        filtered_data = mapdata[
            (mapdata["Year"] >= year_range[0]) & (mapdata["Year"] <= year_range[1])
        ]
    else:
        filtered_data = mapdata[
            (mapdata["Entity"] == selected_country)
            & (mapdata["Year"] >= year_range[0])
            & (mapdata["Year"] <= year_range[1])
        ]

    # Select the appropriate column based on the chosen option
    if option == "Smoking per 100,000 people":
        column_name = "(Rate per 100,000)"
    else:
        column_name = "(Percent)"

    # Scale down the marker sizes
    scaled_marker_sizes = filtered_data[column_name] / 10

    # Create a 2D map using Plotly Express
    fig = px.scatter_geo(
        filtered_data,
        lat="lat",
        lon="long",
        color=column_name,
        hover_data=["Entity", column_name],
        projection="equirectangular",
        size=scaled_marker_sizes,
    )

    # Remove the marker line width
    fig.update_traces(marker=dict(line=dict(width=0)))

    # Set the map title
    fig.update_layout(title_text="Interactive Map")

    # Set the size of the map
    fig.update_layout(width=1600, height=1200)

    # Adjust the size and position of the legend
    fig.update_layout(
        legend=dict(title="", yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    # Display the map using Streamlit
    st.plotly_chart(fig)
    pass


def gender():
    # Load the dataset
    gender_data = pd.read_csv("gender.csv")

    # Create a range slider for year selection
    year_range = st.slider("Select year range", 2000, 2020, (2000, 2020))

    # Filter the dataset based on selected years
    filtered_data = gender_data[
        (gender_data["Year"] >= year_range[0]) & (gender_data["Year"] <= year_range[1])
    ]

    # Remove rows with NaN values in the 'Population (historical estimates)' column
    filtered_data = filtered_data.dropna(subset=["Population (historical estimates)"])

    # Function to filter countries by region
    def filter_countries(data, region):
        if region == "mena":
            mena_countries = [
                "Bahrain",
                "Cyprus",
                "Egypt",
                "Iran",
                "Iraq",
                "Israel",
                "Jordan",
                "Kuwait",
                "Lebanon",
                "Libya",
                "Morocco",
                "Oman",
                "Palestine",
                "Qatar",
                "Saudi Arabia",
                "Syria",
                "Tunisia",
                "Turkey",
                "United Arab Emirates",
                "Yemen",
            ]
            return data[data["Entity"].isin(mena_countries)]
        else:
            return data

    # Create a button for MENA filtering
    if st.button("MENA"):
        filtered_data = filter_countries(filtered_data, "mena")

    # Exclude the entity "World" from the filtered data
    filtered_data = filtered_data[filtered_data["Entity"] != "World"]

    # Aggregate the data by taking the average smoking percentages and population for each country
    agg_data = filtered_data.groupby(["Entity"])[["Population (historical estimates)", "Prevalence of current tobacco use, females (% of female adults)", "Prevalence of current tobacco use, males (% of male adults)"]].mean().reset_index()


    # Calculate the scaling factor for the size of the circles
    circle_scaling_factor = 0.00005

    # Scale the size of the circles based on the population
    agg_data["Circle Size"] = (
        agg_data["Population (historical estimates)"] * circle_scaling_factor
    )

    # Create a scatter plot using Matplotlib
    plt.figure(figsize=(12, 9))

    plt.scatter(
        agg_data["Prevalence of current tobacco use, females (% of female adults)"],
        agg_data["Prevalence of current tobacco use, males (% of male adults)"],
        s=agg_data["Circle Size"],
        alpha=0.7,
    )

    # Add labels to the data points
    for i in range(len(agg_data)):
        plt.text(
            agg_data.iloc[i][
                "Prevalence of current tobacco use, females (% of female adults)"
            ],
            agg_data.iloc[i][
                "Prevalence of current tobacco use, males (% of male adults)"
            ],
            agg_data.iloc[i]["Entity"],
            fontsize=10,
            ha="center",
            va="center",
        )

    # Set the plot title and axes labels
    plt.title("Difference in Smoking between Men and Women", fontsize=18)
    plt.xlabel("Female Smoking Percentage", fontsize=18)
    plt.ylabel("Male Smoking Percentage", fontsize=18)

    # Customize the plot appearance
    plt.grid(True)
    plt.tight_layout()

    # Display the plot using Streamlit
    st.pyplot(plt)


def barplot_female_smokers():
    # Load the dataset
    data = pd.read_csv("gender.csv")

    # Filter the data for MENA region countries and world
    mena_countries = [
        "Bahrain",
        "Cyprus",
        "Egypt",
        "Iran",
        "Iraq",
        "Israel",
        "Jordan",
        "Kuwait",
        "Lebanon",
        "Libya",
        "Morocco",
        "Oman",
        "Palestine",
        "Qatar",
        "Saudi Arabia",
        "Syria",
        "Tunisia",
        "Turkey",
        "United Arab Emirates",
        "Yemen",
        "World",
    ]

    data_mena = data[data["Entity"].isin(mena_countries)]

    # Filter the data for the year 2019
    data_2019 = data_mena[data_mena["Year"] == 2019]

    # Create a bar plot using Plotly Express
    fig = px.bar(
        data_2019,
        x="Entity",
        y="Prevalence of current tobacco use, females (% of female adults)",
        title="Percentage of Female Smokers in 2019 - MENA Region and World",
        labels={
            "Entity": "Country",
            "Prevalence of current tobacco use, females (% of female adults)": "Percentage",
        },
    )

    # Customize the layout
    fig.update_layout(
        xaxis_tickangle=-45,
        legend=dict(font=dict(size=16)),
        font=dict(size=18),
        width=1200,  # Adjust the width of the graph
        height=1000,  # Adjust the height of the graph
    )

    # Adjust the size of x and y labels
    fig.update_xaxes(tickfont=dict(size=28))
    fig.update_yaxes(tickfont=dict(size=28))

    # Remove color legends
    fig.update_traces(showlegend=False)

    # Display the bar plot using Streamlit
    st.plotly_chart(fig)


def line_male_smokers():
    data = pd.read_csv("gender.csv")

    # Filter the data for MENA region countries and world
    mena_countries = [
        "Bahrain",
        "Cyprus",
        "Egypt",
        "Iran",
        "Iraq",
        "Israel",
        "Jordan",
        "Kuwait",
        "Lebanon",
        "Libya",
        "Morocco",
        "Oman",
        "Palestine",
        "Qatar",
        "Saudi Arabia",
        "Syria",
        "Tunisia",
        "Turkey",
        "United Arab Emirates",
        "Yemen",
        "World",
    ]

    data_mena = data[data["Entity"].isin(mena_countries)]

    # Filter the data for the years 2000 to 2020
    data_years = data_mena[(data_mena["Year"] >= 2000) & (data_mena["Year"] <= 2020)]

    # Create a line chart using Plotly Express
    fig = px.line(
        data_years,
        x="Year",
        y="Prevalence of current tobacco use, males (% of male adults)",
        color="Entity",
        title="Percentage of Male Smokers Over Time - MENA Region and World",
        labels={
            "Year": "Year",
            "Prevalence of current tobacco use, males (% of male adults)": "Percentage",
        },
    )

    # Customize the layout
    fig.update_layout(xaxis=dict(tickmode="linear", dtick=5))

    # Add buttons to filter data based on increasing or decreasing trend
    button_increase = st.button("Increasing")
    button_decrease = st.button("Decreasing")

    if button_increase:
        increased_countries = (
            data_years.groupby("Entity")[
                "Prevalence of current tobacco use, males (% of male adults)"
            ].last()
            > data_years.groupby("Entity")[
                "Prevalence of current tobacco use, males (% of male adults)"
            ].first()
        )
        increased_countries = increased_countries[increased_countries].index.tolist()
        fig.data = [trace for trace in fig.data if trace.name in increased_countries]

    if button_decrease:
        decreased_countries = (
            data_years.groupby("Entity")[
                "Prevalence of current tobacco use, males (% of male adults)"
            ].last()
            < data_years.groupby("Entity")[
                "Prevalence of current tobacco use, males (% of male adults)"
            ].first()
        )
        decreased_countries = decreased_countries[decreased_countries].index.tolist()
        fig.data = [trace for trace in fig.data if trace.name in decreased_countries]

    # Customize the layout
    fig.update_layout(
        xaxis_tickangle=-45,
        legend=dict(font=dict(size=18)),
        font=dict(size=22),
        width=1200,  # Adjust the width of the graph
        height=800,  # Adjust the height of the graph
    )

    # Adjust the size of x and y labels
    fig.update_xaxes(tickfont=dict(size=22))
    fig.update_yaxes(tickfont=dict(size=22))

    # Display the line chart using Streamlit
    st.plotly_chart(fig)


def line_female_smokers():
    data = pd.read_csv("gender.csv")

    # Filter the data for MENA region countries and world
    mena_countries = [
        "Bahrain",
        "Cyprus",
        "Egypt",
        "Iran",
        "Iraq",
        "Israel",
        "Jordan",
        "Kuwait",
        "Lebanon",
        "Libya",
        "Morocco",
        "Oman",
        "Palestine",
        "Qatar",
        "Saudi Arabia",
        "Syria",
        "Tunisia",
        "Turkey",
        "United Arab Emirates",
        "Yemen",
        "World",
    ]

    data_mena = data[data["Entity"].isin(mena_countries)]

    # Filter the data for the years 2000 to 2020
    data_years = data_mena[(data_mena["Year"] >= 2000) & (data_mena["Year"] <= 2020)]

    # Create a line chart using Plotly Express
    fig = px.line(
        data_years,
        x="Year",
        y="Prevalence of current tobacco use, females (% of female adults)",
        color="Entity",
        title="Percentage of Female Smokers Over Time - MENA Region and World",
        labels={
            "Year": "Year",
            "Prevalence of current tobacco use, females (% of female adults)": "Percentage",
        },
    )

    # Customize the layout
    fig.update_layout(xaxis=dict(tickmode="linear", dtick=5))

    # Add buttons to filter data based on increasing or decreasing trend
    button_increase = st.button("Increasing (Female)")
    button_decrease = st.button("Decreasing (Female)")

    if button_increase:
        increased_countries = (
            data_years.groupby("Entity")[
                "Prevalence of current tobacco use, females (% of female adults)"
            ].last()
            > data_years.groupby("Entity")[
                "Prevalence of current tobacco use, females (% of female adults)"
            ].first()
        )
        increased_countries = increased_countries[increased_countries].index.tolist()
        fig.data = [trace for trace in fig.data if trace.name in increased_countries]

    if button_decrease:
        decreased_countries = (
            data_years.groupby("Entity")[
                "Prevalence of current tobacco use, females (% of female adults)"
            ].last()
            < data_years.groupby("Entity")[
                "Prevalence of current tobacco use, females (% of female adults)"
            ].first()
        )
        decreased_countries = decreased_countries[decreased_countries].index.tolist()
        fig.data = [trace for trace in fig.data if trace.name in decreased_countries]

    # Customize the layout
    fig.update_layout(
        xaxis_tickangle=-45,
        legend=dict(font=dict(size=18)),
        font=dict(size=22),
        width=1200,  # Adjust the width of the graph
        height=800,  # Adjust the height of the graph
    )

    # Adjust the size of x and y labels
    fig.update_xaxes(tickfont=dict(size=22))
    fig.update_yaxes(tickfont=dict(size=22))

    # Display the line chart using Streamlit
    st.plotly_chart(fig)


def income():
    st.header(
        "Did smoking trends change over time between different income levels around the world?"
    )

    deathsby = pd.read_csv("number-of-deaths-by-risk-factor.csv")

    # Filter the data by income categories and deaths by smoking
    income_levels = [
        "World Bank Low Income",
        "World Bank Lower Middle Income",
        "World Bank Upper Middle Income",
        "World Bank High Income",
    ]
    deaths_by_smoking = deathsby[
        deathsby["Entity"].isin(income_levels)
        & (
            deathsby[
                "Deaths - Cause: All causes - Risk: Smoking - Sex: Both - Age: All Ages (Number)"
            ]
            > 0
        )
    ]

    # Create a line plot using Plotly Express
    fig = px.line(
        deaths_by_smoking,
        x="Year",
        y="Deaths - Cause: All causes - Risk: Smoking - Sex: Both - Age: All Ages (Number)",
        color="Entity",
        labels={
            "Year": "Year",
            "Deaths - Cause: All causes - Risk: Smoking - Sex: Both - Age: All Ages (Number)": "Number of Deaths",
        },
        title="Number of Deaths by Smoking in Different Income Categories",
    )

    # Customize the layout
    fig.update_layout(
        xaxis_title_font=dict(size=25),
        yaxis_title_font=dict(size=25),
        legend=dict(font=dict(size=25)),
        font=dict(size=25),
        width=1400,  # Adjust the width of the graph
        height=700,  # Adjust the height of the graph
    )

    # Display the line plot using Streamlit
    st.plotly_chart(fig)


def line_mena_smok_deaths():
    ## line chart
    # Load the dataset
    data = pd.read_csv("updated_combined_data.csv")

    # Filter the data for world and Middle East and North Africa countries
    world_data = data[data["Entity"] == "World"]
    mena_countries_data = data[
        data["Entity"].isin(
            [
                "Bahrain",
                "Egypt",
                "Iraq",
                "Jordan",
                "Kuwait",
                "Lebanon",
                "Libya",
                "Morocco",
                "Oman",
                "Palestine",
                "Qatar",
                "Saudi Arabia",
                "Syria",
                "Tunisia",
                "United Arab Emirates",
                "Yemen",
            ]
        )
    ]

    # Group the data by year and calculate the total deaths
    world_total_deaths = world_data.groupby("Year")["(Percent)"].sum().reset_index()
    mena_countries_total_deaths = mena_countries_data.pivot_table(
        values="(Percent)", index="Year", columns="Entity", aggfunc="sum"
    ).reset_index()

    # Create a line chart for world and Middle East and North Africa countries
    fig = go.Figure()

    # Add the line trace for world data
    fig.add_trace(
        go.Scatter(
            x=world_total_deaths["Year"],
            y=world_total_deaths["(Percent)"],
            name="World",
            line=dict(color="blue"),
        )
    )

    # Add the line traces for Middle East and North Africa countries
    for country in mena_countries_total_deaths.columns[1:]:
        fig.add_trace(
            go.Scatter(
                x=mena_countries_total_deaths["Year"],
                y=mena_countries_total_deaths[country],
                name=country,
            )
        )

    # Customize the layout
    fig.update_layout(
        title=dict(
            text="Total Deaths - Middle East and North Africa Countries",
            font=dict(size=22),  # Adjust the font size of the title
        ),
        xaxis_title="Year",
        yaxis_title="Total Deaths (%)",
        legend_title_text="Entity",
        width=2000,  # Adjust the width of the chart
        height=1000,  # Adjust the height of the chart
        legend=dict(font=dict(size=22)),  # Adjust the font size of the legend
        xaxis=dict(
            title=dict(font=dict(size=22)),  # Adjust the font size of the x-axis label
            tickfont=dict(size=22),  # Adjust the font size of the x-axis tick labels
        ),
        yaxis=dict(
            title=dict(font=dict(size=22)),  # Adjust the font size of the y-axis label
            tickfont=dict(size=22),  # Adjust the font size of the y-axis tick labels
        ),
    )

    # Add buttons to filter the lines based on trend
    button_increasing = st.button("Increasing")
    button_decreasing = st.button("Decreasing")

    if button_increasing:
        filtered_countries = mena_countries_total_deaths.columns[1:][
            mena_countries_total_deaths.iloc[-1, 1:]
            > mena_countries_total_deaths.iloc[0, 1:]
        ]
        fig.data = [trace for trace in fig.data if trace.name in filtered_countries]

    if button_decreasing:
        filtered_countries = mena_countries_total_deaths.columns[1:][
            mena_countries_total_deaths.iloc[-1, 1:]
            < mena_countries_total_deaths.iloc[0, 1:]
        ]
        fig.data = [trace for trace in fig.data if trace.name in filtered_countries]

    # Display the line chart using Streamlit
    st.plotly_chart(fig)


def table_top10_death(col):
    ### top 10 by percentage deaths

    # Load the dataset
    data = pd.read_csv("updated_combined_data.csv")

    # Filter the data for the year 2019
    data_2019 = data[data["Year"] == 2019]

    # Sort the data by percentage in descending order and select the top 10 countries
    top_10_countries = data_2019.nlargest(10, "(Percent)")

    # Reset the index and add 1 to start counting from 1
    top_10_countries.reset_index(drop=True, inplace=True)
    top_10_countries.index += 1

    # Define the CSS style for the table
    table_style = """
        <style>
        table {
            font-size: 40px;  /* Adjust the font size of the table */
            width: 800px;  /* Adjust the width of the table */
        }
        th {
            font-weight: bold;
        }
        </style>
    """

    # Display the top 10 countries in a table with custom style
    col.markdown(table_style, unsafe_allow_html=True)
    col.table(top_10_countries[["Entity", "(Percent)"]])


def affordability():
    # Load the dataset
    data = pd.read_csv("affordability-cigarettes.csv")

    # Filter data for MENA countries
    mena_countries = [
        "Bahrain",
        "Cyprus",
        "Egypt",
        "Iran",
        "Iraq",
        "Israel",
        "Jordan",
        "Kuwait",
        "Lebanon",
        "Libya",
        "Morocco",
        "Oman",
        "Palestine",
        "Qatar",
        "Saudi Arabia",
        "Syria",
        "Tunisia",
        "Turkey",
        "United Arab Emirates",
        "Yemen",
    ]
    mena_data = data[data["Entity"].isin(mena_countries)]

    # Filter data for the selected year
    selected_year = st.slider("Select Year", min_value=2010, max_value=2020, value=2020)
    selected_data = mena_data[mena_data["Year"] == selected_year]

    # Create the bar plot using Matplotlib
    fig, ax = plt.subplots()
    ax.bar(
        selected_data["Entity"],
        selected_data[
            "Indicator:Affordability - percentage of GDP per capita required to purchase 2000 cigarettes of the most sold brand"
        ],
    )
    ax.set_title(
        f"% of GDP per capita to purchase 2000 cigarettes of popular brand ({selected_year})"
    )
    ax.set_xlabel("Country")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=90)  # Rotate x-axis labels for better readability

    # Display the chart using Streamlit
    st.pyplot(fig)


def affordability_table():
    ranks = [
        "1. Iraq",
        "2. Qatar",
        "3. Kuwait",
        "4. United Arab Emirates",
        "5. Lebanon",
    ]

    info_text = "Top 5 countries in mena region in terms of cigarettes affordability:"
    font_size = 30
    color_normal = "black"
    color_bold = "black"

    st.markdown(
        "<p style='font-size:{}px; color:{};'>{}</p>".format(
            font_size, color_normal, info_text
        ),
        unsafe_allow_html=True,
    )

    for ranks in ranks:
        st.markdown(
            "<p style='font-size:{}px; font-weight:bold; color:{};'>{}</p>".format(
                font_size, color_bold, ranks
            ),
            unsafe_allow_html=True,
        )


def recommend():
    # Create a list of recommendations
    recommendations = [
        {
            "title": "1. Raising taxes on nicotine products ",
            "description": 'Raising taxes on nicotine products has been proven effective in reducing smoking rates. For instance, a 10% price increase in cigarettes can lead to a "4%" decrease in smoking prevalence. ',
        },
        {
            "title": "2. Banning smoking in public places ",
            "description": "Implementing smoking bans in public places is another effective measure to reduce exposure to secondhand smoke. Studies have shown significant reductions in secondhand smoke exposure when smoking is prohibited in workplaces, restaurants, and other public settings. ",
        },
        {
            "title": "3. Smoking cessation programs ",
            "description": "Investing in smoking cessation programs and supporting research efforts can aid individuals in quitting smoking and develop more effective strategies, with governments and organizations playing a vital role in funding these initiatives.",
        },
        {
            "title": "4. Campaigns to quit smoking",
            "description": "Conducting public education campaigns to raise awareness about the dangers of smoking is crucial in motivating people to quit. This can be achieved through various channels such as public service announcements and educational programs in schools. ",
        },
        {
            "title": "5. Prohibition of nicotine advertisement",
            "description": "Implementing a comprehensive prohibition on tobacco advertising, promotion, and sponsorship has proven highly effective in reducing tobacco use, particularly among youth populations.",
        },
        {
            "title": "6. Banning sales & consumption for under 18",
            "description": "Banning sales and consumption of nicotine products for those under 18 years old: an important measure in reducing underage smoking rates. By implementing strict regulations and penalties, these policies aim to restrict access to tobacco products and create a tobacco-free environment for young people, protecting their health and well-being.",
        },
    ]

    font_size = 35  # Set the desired font size

    # Create CSS styling for the title
    title_style = (
        "background-color: maroon; color: white; font-weight: bold; padding: 8px;"
    )

    # Set the layout for the titles
    cols = st.columns(len(recommendations))

    # Display the titles horizontally
    for col, recommendation in zip(cols, recommendations):
        with col:
            st.markdown(
                "<h4 style='font-size:{}px;{}'>{}</h4>".format(
                    font_size, title_style, recommendation["title"]
                ),
                unsafe_allow_html=True,
            )

            # Create an expander for the description
            with st.expander("Description", expanded=False):
                # Display the description
                st.markdown(
                    "<p style='font-size:{}px'>{}</p>".format(
                        font_size, recommendation["description"]
                    ),
                    unsafe_allow_html=True,
                )


def create_mind_map():
    # Define the nodes and connections in the mind map
    # nodes = {
    #     'Root': ['Node 1', 'Node 2'],
    #     'Node 1': ['Subnode 1', 'Subnode 2'],
    #     'Node 2': ['Subnode 3', 'Subnode 4']
    # }

    nodes = {
        "Deaths attributed to smoking": [
            "  cancer       ",
            "heart disease",
            "   stroke        ",
            "kidney disease",
            " lung disease ",
            " miscarriage ",
        ],
        # 'cancer': ['Subnode 1', 'Subnode 2'],
        # 'heart disease': ['Subnode 3', 'Subnode 4']
    }

    # Create the graphviz dot string for the mind map
    dot = graph.Digraph()
    dot.attr(rankdir="LR")  # Set the direction of the graph (from left to right)

    # Define the style for nodes and edges
    node_style = (
        'style="filled", fillcolor="#800000", fontcolor="white", fontweight="bold"'
    )
    edge_style = 'style="solid"'

    # Add the nodes and connections to the graph with the defined style
    for parent, children in nodes.items():
        dot.node(parent, **parse_attributes(node_style))
        for child in children:
            dot.node(child, **parse_attributes(node_style))
            dot.edge(parent, child, **parse_attributes(edge_style))

    # Render the mind map using graphviz_chart in Streamlit
    st.graphviz_chart(dot.source)


# Helper function to parse attributes from string to dictionary
def parse_attributes(attribute_string):
    attributes = {}
    for attr in attribute_string.split(","):
        key, value = attr.strip().split("=")
        attributes[key.strip()] = value.strip().strip('"')
    return attributes


def million_premature_deaths():
    info_text = ". 8 MILLION+ PREMATURE DEATHS YEARLY"
    font_size = 30
    color = "#800000"  # Bordeaux color
    st.markdown(
        "<p style='font-size:{}px; animation: highlight 2s infinite; color:{}; font-weight: bold;'>{}</p>".format(
            font_size, color, info_text
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        @keyframes highlight {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def premature_deaths_century():
    info_text = ". 100 MILLION+ DEATHS IN 20TH CENTURY"
    font_size = 30
    color = "#800000"  # Bordeaux color
    st.markdown(
        "<p style='font-size:{}px; animation: highlight 2s infinite; color:{}; font-weight: bold;'>{}</p>".format(
            font_size, color, info_text
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        @keyframes highlight {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def define_smoking():
    info_text = "Term Definition: Smoking, as defined by the WHO, refers to the use of tobacco products in the form of smoking, sucking, chewing, or snuffing, excluding electronic cigarettes."
    font_size = 30
    color = "#800000"  # Bordeaux color

    st.markdown(
        "<p style='font-size:{}px; animation: highlight 2s infinite; color:{}; font-weight: bold;'>{}</p>".format(
            font_size, color, info_text
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        @keyframes highlight {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def one_in_seven_deaths():
    info_text = ". 1 IN 7 GLOBAL DEATHS IS ATTRIBUTED TO SMOKING"
    font_size = 30
    color = "#800000"  # Bordeaux color

    st.markdown(
        "<p style='font-size:{}px; animation: highlight 2s infinite; color:{}; font-weight: bold;'>{}</p>".format(
            font_size, color, info_text
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        @keyframes highlight {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def cancer_deaths():
    info_text = ". MORE THAN 1 IN 5 CANCER DEATHS ATTRIBUTED TO SMOKING"
    font_size = 30
    color = "#800000"  # Bordeaux color

    st.markdown(
        "<p style='font-size:{}px; animation: highlight 2s infinite; color:{}; font-weight: bold;'>{}</p>".format(
            font_size, color, info_text
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        @keyframes highlight {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Globally more than one in five cancer deaths (22% in 2016) are attributed to smoking


def display_study_questions():
    questions = [
        "1. What are the major risk factors causing deaths in different parts of the world?",
        "2. Can we see similar trends between the World, Middle East & North Africa, and Lebanon?",
        "3. What are the shares of death attributed to smoking in different parts of the world?",
        "4. How did the share of deaths attributed to smoking change: World vs MENA vs Lebanon?",
        "5. What is the trend of gender disparity in smoking over time?",
        "6. Is the percentage of smoking among men and women diminishing over time in the MENA region?",
        "7. How can we locally minimize the use and impact of tobacco and nicotine products?",
    ]

    info_text = "In this study, we will investigate different smoking trends in the World, Lebanon, and the MENA region by answering the following seven questions:"
    font_size = 30
    color_normal = "black"
    color_bold = "black"

    st.markdown(
        "<p style='font-size:{}px; color:{};'>{}</p>".format(
            font_size, color_normal, info_text
        ),
        unsafe_allow_html=True,
    )

    for question in questions:
        st.markdown(
            "<p style='font-size:{}px; font-weight:bold; color:{};'>{}</p>".format(
                font_size, color_bold, question
            ),
            unsafe_allow_html=True,
        )


# Define the correct password
correct_password = "123"

# Initialize a variable to track if the password is entered correctly
password_entered = False

# Check if the password is already entered correctly
if "password_entered" in st.session_state:
    password_entered = st.session_state.password_entered

# If password is not entered correctly, display the password input
if not password_entered:
    # Get the user input for the password
    st.title("Enter the Password")
    st.header("Kindly adjust the screen zoom to 50% for better experience")
    password_input = st.text_input("", type="password")
    # password_input = st.text_input("Enter the password", type="password")
    # Check if the password is correct
    if password_input == correct_password:
        password_entered = True
        st.session_state.password_entered = True
    elif password_input != "":
        st.error("Invalid password. Please try again.")

# Only show the app content if the password is entered correctly
if password_entered:
    # Home Page
    if menu_id == "Home":
        col1, col2 = st.columns([1, 1])  # Create two columns
        with col1:
            st.markdown(
                """
            <div style="background-color: #800020; padding: 10px">
                <h2 style="color: white; font-weight: bold">Analysis of Smoking Trends: World, Mena, & Lebanon</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.header("By Wissam Malaeb")
            # Adjust font size using HTML and CSS
            st.markdown(
                """
            <style>
                .custom-text {
                    font-size: 30px;
                }
            </style>

            <div class="custom-text">
                Smoking has been a significant global health issue for decades, and its impact remains a cause for concern. According to two regularly updated studies on the global death toll from the use of tobacco published by the World Health Organization (WHO) and the Institute for Health Metrics and Evaluation (IHME), more than 8 million people die prematurely each year due to smoking-related causes. Additionally, the 20th century saw approximately 100 million premature deaths linked to smoking with the greater portion of these deaths occurring in wealthier countries, highlighting its devastating effects on global health.
            """,
                unsafe_allow_html=True,
            )

            # col11, col22 = st.columns([1,1])
            # with col11:]

            st.header(" ")
            define_smoking()
            st.image("pics/death attrributed to smoking.PNG", width=1100)

        with col2:
            st.image("pics/poeple smoking pic.jpg")
            st.markdown(
                """
            <style>
                .custom-text {
                    font-size: 30px;
                }
            </style>

            <div class="custom-text">
                This study aims to analyze smoking trends from 1990 to 2020, focusing on the prevalence of smoking worldwide with a specific emphasis on the MENA region and Lebanon. It also explores the variations in smoking patterns across different geographic locations, considering cultural factors, gender differences, and income levels. By examining these trends, we can gain valuable insights into the challenges posed by smoking and develop targeted strategies to address them effectively.
                Ultimately, this study seeks to provide actionable recommendations for reducing smoking rates and mitigating its impact within Lebanon. By implementing measures to discourage smoking and promote healthier lifestyles, we can strive towards a smoke-free future and improve the overall well-being of individuals and communities.
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.header(" ")
            million_premature_deaths()
            # with col22:
            premature_deaths_century()
            one_in_seven_deaths()
            cancer_deaths()
            st.header(" ")
            display_study_questions()

    # EDA page
    if menu_id == "Risk Factors":
        risk_factors()
        table_risk_comparison()

    # Dashboard Page
    if menu_id == "Smoking Trends":
        # Create columns layout
        col1, col2 = st.columns([3, 1])

        # Call the map_deathby_smok function in col1
        with col1:
            st.markdown(
                """
            <div style="background-color: #800020; padding: 10px">
                <h2 style="color: white; font-weight: bold">What are shares of death attributed to smoking in different parts of the world?</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )
            map_deathby_smok(col1)

        # Call the table_top10_death function in col2
        with col2:
            # st.image("pics/smoking stand.png")
            st.image("pics/marlboro.jpg")
            st.markdown(" ")
            st.markdown(" ")
            st.markdown(" ")
            st.markdown(" ")

            table_top10_death(col2)

        # map_deathby_smok();
        # table_top10_death();
        st.markdown(
            """
            <div style="background-color: #800020; padding: 10px">
                <h2 style="color: white; font-weight: bold">How did the share of deaths attributed to smoking change: World vs Mena vs Lebanon?</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        line_mena_smok_deaths()

    # Profiling Page
    if menu_id == "Gender Dispartiy & Income":
        col1, col2 = st.columns([1, 1])  # Create two columns

        with col1:
            st.markdown(
                """
            <div style="background-color: #800020; padding: 10px">
                <h2 style="color: white; font-weight: bold">What is the trend of gender disparity in smoking over time?</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )
            gender()

        with col2:
            st.header(" ")
            st.header(" ")
            st.header(" ")
            st.header(" ")
            st.header(" ")

            barplot_female_smokers()
        st.markdown(
            """
                    <div style="background-color: #800020; padding: 10px">
                        <h2 style="color: white; font-weight: bold">Is the percentage of smoking among men and women diminishing over time in the Mena region?</h2>
                    </div>
                    """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([1, 1])  # Create two columns

        with col1:
            # st.markdown(
            #         """
            #         <div style="background-color: #800020; padding: 10px">
            #             <h2 style="color: white; font-weight: bold">test?</h2>
            #         </div>
            #         """,
            #         unsafe_allow_html=True
            # )
            line_male_smokers()

        with col2:
            line_female_smokers()

        #    How affordable are cigarettes in Mena regions and Lebanon?
        st.markdown(
            """
                    <div style="background-color: #800020; padding: 10px">
                        <h2 style="color: white; font-weight: bold">How affordable are cigarettes in Mena regions and Lebanon?</h2>
                    </div>
                    """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([1, 1])  # Create two columns
        with col1:
            affordability()

        with col2:
            st.header(" ")
            st.header(" ")
            st.header(" ")
            st.header(" ")
            st.header(" ")
            affordability_table()
        st.markdown(
            """
                <div style="background-color: #800020; padding: 10px">
                    <h2 style="color: white; font-weight: bold">Does income level affect the trends in deaths attributed to smoking worldwide?</h2>
                </div>
                """,
            unsafe_allow_html=True,
        )
        income()

    # Machine Learning Application
    if menu_id == "Recommendations":
        col1, col2 = st.columns([3, 1])  # Create two columns

        with col1:
            st.markdown(
                """
                        <div style="background-color: #800020; padding: 10px">
                            <h2 style="color: white; font-weight: bold">How can we locally minimize the use and impact of tobacco and nicotine products?</h2>
                        </div>
                        """,
                unsafe_allow_html=True,
            )
        with col2:
            st.image("pics/man breaking cigarette.jpg")
        recommend()
