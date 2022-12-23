'''
sections of app

load data
    loading data from csv file and introduce app
    plot interactive column heatmap of resale transactions using pydeck

clean and filter data for visualisation
    interactive slider for user to select period to visualise

plot visualisation
    various plots to understand data using seaborn

predictions using lgb model
    interactive form for user to enter features for lgb model to make a prediction

explaining model
    explain impact of features on resale price using shap
    
end notes

'''


# imports
import pandas as pd
import streamlit as st
import os
import datetime as dt
from dateutil.relativedelta import relativedelta
import pickle

import matplotlib.pyplot as plt
import pydeck as pdk
import plotly.express as px

import lightgbm as lgb
import shap

import resale_flat_prices_library as LIB



# set tab name
st.set_page_config(page_title='Resale HDB Prices', page_icon='🏠')
# hide menu button
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)
# title of app
st.title('Exploring Resale Prices of Public Housing in Singapore')
st.write('\n')
st.write('\n')



### load data section ###

# cache data
@st.cache
# load data from csv
def load_data_from_csv(file_path):
    # read data
    data = pd.read_csv(file_path)
    # convert to year_month to datetime format
    data['year_month'] = pd.to_datetime(data['year_month'])
    # convert latitude and longitude to numeric
    data[['latitude', 'longitude']] = data[['latitude', 'longitude']].apply(pd.to_numeric)
    # return 
    return data

# define data directory
data_folder = 'data'
# cache data
@st.cache
# combine data
def combine_data(data_folder):
    # create empty dataframe to concat data later
    data_full = pd.DataFrame()
    # loop through each file in data_folder and concat into one df
    for file_name in os.listdir(data_folder):
        # load data
        temp_data = load_data_from_csv(os.path.join(data_folder, file_name))
        # concat data
        data_full = pd.concat([data_full, temp_data])
    # reset index to prevent duplicated indexes and drop created index column
    data_full = data_full.reset_index().drop('index', axis=1)
    return data_full
# combine data
data_full = combine_data(data_folder)

# get date range of full data set
min_data_date = data_full['year_month'].min().strftime('%B %Y')
max_data_date = data_full['year_month'].max().strftime('%B %Y')
total_row_count = '{:,}'.format(len(data_full))

# introduction to app
st.write('''
    Hi there! This app explores public housing or HDB resale prices in Singapore, it covers data visualisations,
     training and tuning a machine learning model to predict resale prices, and the explanation of the model\'s predictions. 
     The source code can be found on [Github](https://github.com/Greco1899/streamlit_resale_flat_prices) 
     and you can connect with me on [LinkedIn](https://www.linkedin.com/in/russellchanws).
    ''')
st.write('''
    The data has been extracted from [Data.gov.sg](https://data.gov.sg/dataset/resale-flat-prices) 
    and used in accordance with [Singapore Open Data License](https://data.gov.sg/open-data-licence).
    ''')
st.write(f'There are a total of {total_row_count} recorded resale flat transactions from {min_data_date} to {max_data_date}.')
st.write('\n')
st.write('\n')



### pydeck map using latitude and longitude section ###

# cache data
@st.cache
# filter data for map due to memory limit in streamlit
def filter_data_for_map(data, years):
    # get number of years to filter
    map_date = data['year_month'].max() - relativedelta(years=years)
    # filter data for map by years
    data_for_map = data.loc[(data['year_month'] >= map_date)]
    # return
    return data_for_map

# filter data for map due to memory limit in streamlit
# define max number of years
max_heatmap_years = 1
data_for_map = filter_data_for_map(data_full, max_heatmap_years)

# describe map
st.write(f'''
    Let\'s start by having a look at where flats have been transacted in the past {max_heatmap_years} year. 
    The taller the pillars, the more transactions have taken place.
    ''')
st.write('''
    The spikes are likely to be flats where the [Minimum Occupation Period](https://www.hdb.gov.sg/residential/selling-a-flat/eligibility) 
    has just passed and are eligible to be sold on the resale market.
    ''')

# create pydeck map in streamlit
def pydeck_map(data_for_map):
    # create map from pydeck
    layer=[
        pdk.Layer(
            'HexagonLayer',
            data=data_for_map,
            get_position='[longitude, latitude]',
            radius=40,
            elevation_scale=3,
            elevation_range=[0,500],
            pickable=True,
            extruded=True,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=data_for_map,
            get_position='[longitude, latitude]',
            get_color='[200, 30, 0, 160]',
            get_radius=40,
        )
    ]

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(latitude=1.355, longitude=103.81, zoom=10, pitch=40),
        layers=layer
        ))

# create pydeck map in streamlit
pydeck_map(data_for_map)
st.write('\n')
st.write('\n')



### clean and filter data visualisation section ###

# slider to select period of years to visualise

# describe visualisation section
st.write(f'Next up is a deeper dive into the Resale Flat Prices data.')
# default minimum and maximum year from data
period_date_max = data_full['year_month'].max().year
period_date_min = data_full['year_month'].min().year
# define slider, default value max as latest data year and min as x years before
# define years before
default_years_before = 3
visualisation_period = st.slider('Select a period would you like to visualise.', min_value=period_date_min, max_value=period_date_max, value=(period_date_max-default_years_before,period_date_max))
# cache data
@st.cache
# filter data based on period
def filter_visualisation_data(data, visualisation_period):
    data = data.loc[(data['year_month'] >= str(visualisation_period[0])+'-01-01') & (data['year_month'] <= str(visualisation_period[1])+'-12-01')]
    return data
data = filter_visualisation_data(data_full, visualisation_period)

# get descriptive statistics of resale_price for selected period

# cache data
@st.cache
# describe resale_price column
def describe_data_column(data, column):
    data = data[column].describe()
    return data
period_describe = describe_data_column(data, 'resale_price')

# mean
period_mean = '{:,}'.format(round(period_describe['mean']))
# median
period_median = '{:,}'.format(round(period_describe['50%']))
# max
period_max = '{:,}'.format(round(period_describe['max']))
# get data for most expensive flat, if there is more than one, select latest transaction
most_expensive = data.loc[(data['resale_price'] == period_describe['max'])].reset_index(drop=True).sort_values('year_month', ascending=False).iloc[0]

# print descriptive statistics
st.write(f'In the period from {visualisation_period[0]} to {visualisation_period[1]}:')
st.write(f'The average resale flat price is \${period_mean} and the median is \${period_median}.')
st.write(f"""
    The most expensive flat sold for **${period_max}**! 
    The flat was transacted in {most_expensive['year_month'].strftime('%B %Y')} at 
    {most_expensive['block'].title()+' '+most_expensive['street_name'].title()}, 
    it was a {most_expensive['flat_type'].title()} with a {round(most_expensive['floor_area_sqm'])} square meter floor area.
    """)
st.write('\n')
st.write('\n')


### visualisation section ###

# set plot attributes
plot_width = 1000
plot_height = 700
plot_title_x = 0.5

### histogram of town ###

# order by value count of town
town_order = data['town'].value_counts().index
# describe plot
st.write(f"This chart shows the number of transactions by Town. The most transactions occured in {town_order[0].title()}.")

fig = px.histogram(
    data, 
    y='town',
    text_auto=True,
    template='simple_white',
    height=plot_height,
    ).update_yaxes(categoryorder='total ascending')

fig = fig.update_layout(
    title='Count of Transactions by Town',
    title_x=plot_title_x,
    xaxis_title='Count',
    yaxis_title='Town',
    )
st.plotly_chart(fig, use_container_width=True)
st.write('\n')

### histogram of flat type ###

# describe plot
st.write(f"""
    And this chart shows the number of transactions by Flat Type. 
    The most often transacted Flat Type are {data['flat_type'].value_counts().index[0].title()} flats.
    """)

# order by flat_type alphabetically
flat_type_order = sorted(list(data['flat_type'].unique()))

fig = px.histogram(
    data,
    x='flat_type',
    category_orders=dict(flat_type=flat_type_order),
    text_auto=True,
    template='simple_white',
    height=plot_height,
    )
    
fig = fig.update_layout(
    title='Count of Transactions by Flat Type',
    title_x=plot_title_x,
    xaxis_title='Flat Type',
    yaxis_title='Count'
    )
st.plotly_chart(fig, use_container_width=True)
st.write('\n')

### scatterplot of resale_price and floor_area_sqm ###

# describe plot
st.write(f"""
    Now this is a scatterplot of Resale Prices compared to Floor Area. 
    No surprise here that we can see a general trend that a flat with more rooms and space cost more.
    """)
st.write(f"However, we can also see that similar flats with the same floor area and flat type can vary significantly in price!")

fig = px.scatter(
    data,
    x='floor_area_sqm',
    y='resale_price',
    color='flat_type',
    opacity=0.4,
    category_orders=dict(flat_type=flat_type_order),
    template='simple_white',
    height=plot_height,
    )
    
fig = fig.update_layout(
    title='Resale Flat Price vs Floor Area',
    title_x=plot_title_x,
    xaxis_title='Floor Area (SQM)',
    yaxis_title='Resale Price',
    legend_title='Flat Type'
    )
st.plotly_chart(fig, use_container_width=True)
st.write('\n')

### boxplot of flat type ###

# describe plot
st.write("""
    These are boxplots of Resale Prices by different Flat Types. 
    Here's a quick refresher on how to [read boxplots](https://miro.medium.com/max/2400/1*2c21SkzJMf3frPXPAR_gZA.png) if you need it.
    """)

fig = px.box(
    data,
    x='resale_price',
    y='flat_type',
    category_orders=dict(flat_type=flat_type_order),
    template='simple_white',
    height=plot_height,
    )

fig = fig.update_layout(
    title='Resale Price by Flat Type',
    title_x=plot_title_x,
    xaxis_title='Resale Price',
    yaxis_title='Flat Type',
    )
st.plotly_chart(fig, use_container_width=True)
st.write('\n')

### boxplot of town ###

# group by town and get median resale price
town_median_resale_price = data.groupby(['town']).agg({'resale_price':'median'}).reset_index().sort_values('resale_price', ascending=False)
# get town with highest median 
most_expensive_town = town_median_resale_price['town'].iloc[0].title()
# get town with lowest median
least_expensive_town = town_median_resale_price['town'].iloc[-1].title()
# compare and get difference between highest median and lowest median
median_price_difference = round(town_median_resale_price['resale_price'].iloc[0] / town_median_resale_price['resale_price'].iloc[-1], 1)
# order by descending median resale_price
town_order = list(town_median_resale_price['town'])

# describe plot
st.write(f"Here are more pretty boxplots of Resale Prices by Town. The most expensive area to buy a flat is in {most_expensive_town}.")
st.write(f"""
    Apart from the floor area and flat type, the location of the flat can also influence the price. 
    The median price of a flat in {most_expensive_town} is **{median_price_difference}** times higher than {least_expensive_town}!
    """)

fig = px.box(
    data,
    x='resale_price',
    y='town',
    category_orders={'town':town_order},
    template='simple_white',
    height=plot_height,
    )

fig = fig.update_layout(
    title='Resale Price by Town',
    title_x=plot_title_x,
    xaxis_title='Resale Price',
    yaxis_title='Town',
    )
st.plotly_chart(fig, use_container_width=True)
st.write('\n')


### prediction section ###

st.write('\n')
st.write('# Predicting Resale Flat Price')
st.write('''
    Enter some basic information (or just try it out with the default values) of the flat you want to 
    sell or buy for the model to predict it\'s price.
    ''')

# form to store users input
with st.form(key='input_form'):

    # ask and store users input
    input_postal_code = st.text_input(label='Postal Code', value='440033')
    input_floor_area_sqm = st.number_input(label='Floor Area in square meters (1 square meter approximately 10 square feet) ', min_value=1, max_value=500, value=70, step=10)
    input_floor = st.number_input(label='Floor', min_value=1, max_value=100, value=10, step=2)
    input_lease_commence_year = st.number_input(label='Lease Commence Year', min_value=1900, max_value=dt.date.today().year, value=1975, step=1)

    # get latitude and longitude from postal code
    coordinates = LIB.get_coordinates_from_address(input_postal_code+' Singapore', st.secrets['geocode_api_key'])
    # calculate remaining lease years from lease commencement date
    input_remaining_lease_years = dt.date.today().year - input_lease_commence_year

    # format user inputs into df for xgb prediction
    input_data = pd.DataFrame({
        'latitude':[coordinates[0]],
        'longitude':[coordinates[1]],
        'floor_area_sqm':[input_floor_area_sqm],
        'floor':[input_floor],
        'remaining_lease_years':[input_remaining_lease_years]
    })

    # submit form button
    st.write('First, load the inputs to the machine learning model to prepare for a prediction:')
    submit = st.form_submit_button(label='Load')

# load model 
lgb_model = pickle.load(open('lgb_model.pkl', 'rb'))

# describe predict button
st.write('Second, take a guess at the price before running the model 😊.')

# add predict button
if st.button('Predict'):
    # predict input_data using model
    prediction = lgb_model.predict(input_data)[0]
    # format prediction with thousands separator and round to two decimal places
    prediction = '{:,}'.format(round(prediction))
    # print prediction
    st.success(f'''
        The predicted resale price of a flat at postal code {input_postal_code}, 
        with a floor area of {input_floor_area_sqm} square meters, on the {input_floor} floor, 
        and with a lease that commenced in {input_lease_commence_year} is **${prediction}**!
        ''')

st.write('After submitting the prediction, we will move on to the next section and look at how we can explain our model!')
st.write('\n')
st.write('\n')



### explain model section ###

# cache data
# @st.cache
# filter and split data into train and test set
def get_train_test_data(data_full):
    train_data, test_data = LIB.filter_and_split_data(clean_data=data_full, test_months=3, train_years=1)
    return train_data, test_data
train_data, test_data = get_train_test_data(data_full)

# set X and y using train data
X_train = train_data.drop('resale_price', axis=1)
y_train = train_data['resale_price']
# set X_test, y_test using test data
X_test = test_data.drop('resale_price', axis=1)
y_test = test_data['resale_price']

# define number of samples to use
number_of_samples = 1000
# filter X_train by number of samples
X_train_filtered = X_train[-number_of_samples:]

st.write('# Explaining the Model')
st.write(f'''
    Now we will explore how does each feature impacts the predicted resale price using 
    [SHAP (SHapley Additive exPlainations)](https://github.com/slundberg/shap#readme) with {number_of_samples} samples from the data.
    ''')
st.write('\n')

# cache data
@st.cache
def shap_explainer(model, X_train):
    # explain the model's predictions using SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)
    return shap_values

# with st.spinner('Generating Feature Explaination...'):
shap_values = shap_explainer(lgb_model, X_train_filtered)

### barplot of mean shap values ###

# create df to store features and shap values
mean_shap_values = pd.DataFrame({
    'features':X_train_filtered.columns.tolist(), 
    'shap_values':abs(shap_values.values).mean(axis=0).tolist()
    })
# sort by importance
mean_shap_values = mean_shap_values.sort_values('shap_values', ascending=False)
# describe plot
st.write(f'Here we have the mean SHAP values of each of the features, and most important feature is "_{mean_shap_values["features"].iloc[0]}_".')

fig = px.bar(
    mean_shap_values,
    x='shap_values',
    y='features',
    template='simple_white',
    height=plot_height,
    ).update_yaxes(categoryorder='total ascending')

fig = fig.update_layout(
    title='Mean SHAP Values and Importance of Features',
    title_x=plot_title_x,
    xaxis_title='Mean SHAP Value',
    yaxis_title='Features',
    )
st.plotly_chart(fig, use_container_width=True)
st.write('\n')

### swarmplot of shap values of each data point ###

# set plot attributes
plot_figsize = (15,10)
plot_title_fontsize = 18
plot_axis_fontsize = 15

# describe plot
st.write(f'The below plot summarises all {"{:,}".format(number_of_samples)} data points in the training data set and shows the positive or negative impact on the outcome of the model.')
st.write('We can see that for a high "_floor_area_sqm_" (coloured pink) has a high positive impact on resale price, while a low "_floor_area_sqm_" (coloured blue) has a high negative impact on the resale price.')
st.write('''
    Taking "_latitude_" as another example, we can observe that a flat with a low "_latitude_" (located further south) 
    has a bigger positive impact on it\'s resale price than a flat with a higher "_latitude_" (located further north).
    ''')

# set plot and figure size
fig, ax = plt.subplots(figsize=plot_figsize)

# plot
shap.plots.beeswarm(shap_values, show=False)
# get current figure
fig = plt.gcf()
fig.set_figwidth(15)
fig.set_figheight(10)

# formatting
# set title
ax.set_title('SHAP Values and Impact on Model Prediction', fontsize=plot_title_fontsize)
ax.set_xlabel('SHAP Value', fontsize=plot_axis_fontsize)
ax.set_ylabel('Features', fontsize=plot_axis_fontsize)

# show plot
st.pyplot(fig)
st.write('\n')

### waterfall plot of shap values of one data point ###

# define sample row to inspect in detail
sample_row = 42
# get average predicted shap value or base shap value
averge_predict_value = round(shap_values.base_values[sample_row], 3)
# get predicted shap value
sample_predict_value = round(averge_predict_value + shap_values.values[sample_row].sum(), 3)
# create dataframe of sample with features and respective shap values
waterfall_sample = pd.DataFrame({
    'features':list(X_train_filtered),
    'feature_values':X_train_filtered.iloc[sample_row],
    'shap_values':shap_values.values[sample_row].tolist(),
    'shap_impact':abs(shap_values.values[sample_row]).tolist()
})
# sort waterfall_sample by shap_values in descending order
waterfall_sample = waterfall_sample.sort_values('shap_impact', ascending=False)
# get feature and feature_value of high shap_value
highest_feature = waterfall_sample["features"].loc[waterfall_sample["shap_values"].idxmax()]
highest_feature_value = round(waterfall_sample["feature_values"].loc[waterfall_sample["shap_values"].idxmax()], 3)
# get feature and feature_value of lowest shap_value
lowest_feature = waterfall_sample["features"].loc[waterfall_sample["shap_values"].idxmin()]
lowest_feature_value = round(waterfall_sample["feature_values"].loc[waterfall_sample["shap_values"].idxmin()])

# describe plot
st.write('''
    Apart from having an explainatory overview of how each feature impacts the model, 
    we can also focus in on 1 sample using a waterfall plot below to show how each of the 
    features impact or move the resale price of the sample.
    ''')
st.write('There\'s quite a lot going on here but we\'ll break it down one step at a time.')
st.write(f'''
    On the Y axis, we can see the features and respective values of the sample with the most impactful feature, 
    "_{waterfall_sample["features"].iloc[0]}_" at the top and the least impactful feature, 
    "_{waterfall_sample["features"].iloc[-1]}_" at the bottom.
    ''')
st.write(f'''
    Next up on the X axis, we see the value {averge_predict_value}, that is the average predicted resale price of our training data. 
    And on the top of the chart we can see the value {sample_predict_value} which is the model\'s predicted resale price of this sample.
    ''')
st.write(f'''
    And finally in the main chart area, we can inspect how each of the features \'forces\' 
    the resale price from {averge_predict_value} towards {sample_predict_value}. 
    A "_{highest_feature}_" of {highest_feature_value} increases the resale price the most 
    while a "_{lowest_feature}_" of {lowest_feature_value} reduces the resale price the most.
    ''')

# set plot and figure size
fig, ax = plt.subplots(figsize=plot_figsize)

# plot
shap.plots.waterfall(shap_values[sample_row], show=False)
# get current figure
fig = plt.gcf()
fig.set_figwidth(15)
fig.set_figheight(10)

# formatting
# set title
ax.set_title('Impact of Features on Resale Price', fontsize=plot_title_fontsize)
ax.set_xlabel('SHAP Value', fontsize=plot_axis_fontsize)
ax.set_ylabel('Features', fontsize=plot_axis_fontsize)

# show plot
st.pyplot(fig)
st.write('\n')

### waterfall plot of shap values of one data point ###

# describe plot
st.write('Here a look at another example using your earlier inputs!')

# find shap values of earlier input data used for prediction
explainer = shap.Explainer(lgb_model)
shap_values_input = explainer(input_data)

# set plot and figure size
fig, ax = plt.subplots(figsize=plot_figsize)

# plot
shap.plots.waterfall(shap_values_input[0], show=False)
# get current figure
fig = plt.gcf()
fig.set_figwidth(15)
fig.set_figheight(10)

# formatting
# set title
ax.set_title('Impact of Features on Resale Price', fontsize=plot_title_fontsize)
ax.set_xlabel('SHAP Value', fontsize=plot_axis_fontsize)
ax.set_ylabel('Features', fontsize=plot_axis_fontsize)

# show plot
st.pyplot(fig)
st.write('\n')
st.write('\n')
st.write('\n')



### end notes ###

st.write('_That\'s all folks!_ Thanks for viewing 😎.')
st.write('''
    I\'m always looking for feedback so do reach out and connect with me on [LinkedIn](https://www.linkedin.com/in/russellchanws) 
    and the source code can be found on [Github](https://github.com/Greco1899/streamlit_resale_flat_prices).
    ''')
st.write('\n')
st.write('\n')
st.write('\n')
st.image('https://images.unsplash.com/photo-1617275249641-322ed29f098e?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=2671&q=80')
st.caption('Photo by [Jiachen Lin](https://unsplash.com/@jiachenlin) on [Unsplash](https://unsplash.com).')


