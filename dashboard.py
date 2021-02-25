import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import NaN
import datetime
import time
import seaborn as sns

# ------------------------------------------------
# conda activate datamining
# streamlit run dashboard.py
st.title('DASHBOARD - Digital Economy')
# ------------------------------------------------

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache(allow_output_mutation=True)
def load_engagement_data():
    data = pd.read_csv('Engagement Rate.csv')
    return data


@st.cache(allow_output_mutation=True)
def load_follower_data():
    data = pd.read_csv('Followers count.csv')
    return data


@st.cache(allow_output_mutation=True)
def load_response_data():
    data = pd.read_csv('Response Time.csv')
    return data


@st.cache(allow_output_mutation=True)
def load_ssov_pr():
    data = pd.read_csv('SSOV & PR.csv')
    return data


# Assignment Selection
menu = ['Assignment 1', 'Assignment 2']
# st.sidebar.markdown('Izzah\t\t1171101738 \nGlenn\t\t1171101736 \nNiroshaan\t1171101816 \nSyazwan\t\t1171101803')

st.sidebar.header('Social Media Computing')
st.sidebar.subheader('Main Menu')
page = st.sidebar.selectbox("Select Page Menu", menu)

st.sidebar.subheader('Group Member')
st.sidebar.info('Izzah 1171101738 - GrabMY')
st.sidebar.warning('Glenn 1171101736 - LazadaMY')
st.sidebar.error('Niroshaan 1171101816 - ShopeeMY')
st.sidebar.success('Syazwan 1171101803 - watsonsmy')

engagement_rate_df = load_engagement_data()
followers_count_df = load_follower_data()
response_time_df = load_response_data()
ssov_pr_df = load_ssov_pr()

brands = ['GrabMY', 'LazadaMY', 'ShopeeMY', 'watsonsmy']
palette = ['dodgerblue', 'gold', 'lightcoral', 'mediumturquoise']
# palette = ['blue', 'orange', 'green', 'red']
font_size = 16

if page == 'Assignment 1':
    st.title('Assignment 1')

    st.header('1. Growth of followers')
    # =================================
    followers_count_df = followers_count_df.sort_values(['screen_name', 'date'], ascending=[True, True])
    followers_count_df['daily_new_followers'] = followers_count_df['followers_count'].diff()
    followers_count_df["daily_new_followers"] = followers_count_df["daily_new_followers"].replace(NaN, 0)
    followers_count_df["daily_new_followers"] = followers_count_df["daily_new_followers"].replace(-13163.0, 0)
    followers_count_df["daily_new_followers"] = followers_count_df["daily_new_followers"].replace(-19208, 0)
    followers_count_df["daily_new_followers"] = followers_count_df["daily_new_followers"].replace(285732, 0)
    followers_count_df['growth_rate'] = (followers_count_df.daily_new_followers /
                                         followers_count_df.followers_count.shift(1)) * 100
    followers_count_df["growth_rate"] = followers_count_df["growth_rate"].replace(NaN, 0)
    followers_count_df["growth_rate"] = round(followers_count_df["growth_rate"], 4)

    arrGrab = followers_count_df.loc[followers_count_df['screen_name'] == 'GrabMY', 'growth_rate'].array
    arrLazada = followers_count_df.loc[followers_count_df['screen_name'] == 'LazadaMY', 'growth_rate'].array
    arrShopee = followers_count_df.loc[followers_count_df['screen_name'] == 'ShopeeMY', 'growth_rate'].array
    arrWatsons = followers_count_df.loc[followers_count_df['screen_name'] == 'watsonsmy', 'growth_rate'].array

    data_growth = {
        'date': followers_count_df['date'].unique(),
        'GrabMY': arrGrab,
        'LazadaMY': arrLazada,
        'ShopeeMY': arrShopee,
        'watsonsmy': arrWatsons
    }

    followers_count_df.set_index('date', inplace=True)
    fig, ax = plt.subplots()
    plt.figure(figsize=(16, 10))
    followers_count_df.groupby('screen_name')['growth_rate'].plot(legend=True, marker='o')
    plt.xticks(rotation='horizontal')
    plt.title("Growth rate of followers for Grab, Lazada, Shopee and Watsons", fontsize=21)
    plt.xticks(rotation=360)
    plt.xlabel("Brand", fontsize=font_size)
    plt.ylabel("Growth Rate (%)", fontsize=font_size)
    # st.pyplot()

    df_growth = pd.DataFrame(data_growth)
    df_growth = df_growth.rename(columns={'date': 'index'}).set_index('index')
    st.line_chart(df_growth[1:])

    if st.checkbox('Show growth rate data'):
        st.subheader('Growth Rate Data')
        st.table(df_growth.transpose())

    st.header('2. Response Time')
    # ===========================
    response_time_df['time'] = pd.to_datetime(response_time_df['tweet_created_at'])
    response_time_df['dates'] = response_time_df['time'].dt.date

    response_time_df['duration'] = 0
    rows = len(response_time_df['response_time'])

    for i in range(0, len(response_time_df['response_time'])):
        x = time.strptime(response_time_df['response_time'][i].split(',')[0], '%H:%M:%S')
        response_time_df['duration'][i] = datetime.timedelta(hours=x.tm_hour, minutes=x.tm_min,
                                                             seconds=x.tm_sec).total_seconds()

    date_arr = sorted(response_time_df['dates'].unique())

    response_time_df = response_time_df.sort_values(['query', 'dates'], ascending=[True, True])
    response_time_df.set_index('dates', inplace=True)
    response_time_df = response_time_df.groupby(['dates', 'query'])['duration'].mean().unstack()

    # st.table(response_time_df)
    response_time_df["@grabmy"] = response_time_df["@grabmy"].replace(NaN, response_time_df["@grabmy"].median())
    response_time_df["@lazadamy"] = response_time_df["@lazadamy"].replace(NaN, response_time_df["@lazadamy"].median())
    response_time_df["@shopeemy"] = response_time_df["@shopeemy"].replace(NaN, response_time_df["@shopeemy"].median())
    response_time_df["@watsonsmy"] = response_time_df["@watsonsmy"].replace(NaN, response_time_df["@watsonsmy"].median())

    # st.table(response_time_df)
    # st.write(response_time_df["@watsonsmy"].array)

    arrGrab = response_time_df["@grabmy"].array
    arrLazada = response_time_df["@lazadamy"].array
    arrShopee = response_time_df["@shopeemy"].array
    arrWatsons = response_time_df["@watsonsmy"].array

    data_response = {
        'dates': date_arr,
        '@grabmy': arrGrab.astype(int),
        '@lazadamy': arrLazada.astype(int),
        '@shopeemy': arrShopee.astype(int),
        '@watsonsmy': arrWatsons.astype(int)
    }

    # st.table(data_response)
    #
    # st.write(len(response_time_df['dates'].unique()))
    # st.write(len(arrGrab))
    # st.write(len(arrLazada))
    # st.write(len(arrShopee))
    # st.write(len(arrWatsons))

    df_response = pd.DataFrame(data_response)
    df_response_new = pd.DataFrame(df_response[:], columns=['@grabmy', '@lazadamy', '@shopeemy', '@watsonsmy'])
    st.area_chart(df_response_new)
    df_response = df_response.rename(columns={'dates': 'index'}).set_index('index')
    # st.line_chart(df_response)
    # st.area_chart(df_response)

    if st.checkbox('Show response time data'):
        st.subheader('Response Time Data')
        st.table(df_response.transpose())

    fig, ax = plt.subplots(figsize=(15, 7))
    boxplot_rt = plt.boxplot(response_time_df, patch_artist=True)

    for patch, color in zip(boxplot_rt['boxes'], palette):
        patch.set_facecolor(color)

    for median in boxplot_rt['medians']:
        median.set(color='w',
                   linewidth=2.5)
    x = [1, 2, 3, 4]
    plt.xticks(x, brands, rotation=360)
    plt.xlabel("Brand", fontsize=font_size)
    plt.ylabel("Time (s)", fontsize=font_size)
    plt.title("Distribution of response time within 21 days period for all brands", fontsize=21)
    st.pyplot()

    st.header('3. Engagement Rate')
    # ==============================
    eng_col1, eng_col2 = st.beta_columns(2)
    eng_col1.markdown('### Total Retweets :repeat:')
    eng_col2.markdown('### Total Favourites :heart:')
    # eng_col2.markdown('### Total Favourites :heart: :bird: :blue_heart: :globe_with_meridians:')

    fig, ax = plt.subplots(figsize=(15, 7))
    plot_er = engagement_rate_df.groupby(['username'])['retweet_count'].sum().plot(kind='bar', ax=ax, color=palette)
    plt.xticks(rotation=360)
    plt.xlabel("Brand", fontsize=font_size)
    plt.ylabel("Total retweets", fontsize=font_size)
    plt.title("Total retweets within 21days period for all brands", fontsize=21)
    eng_col1.pyplot()

    fig, ax = plt.subplots(figsize=(15, 7))
    plot_er1 = engagement_rate_df.groupby(['username'])['favorite_count'].sum().plot(kind='bar', ax=ax, color=palette)
    plt.xticks(rotation=360)
    plt.xlabel("Brand", fontsize=font_size)
    plt.ylabel("Total favorites", fontsize=font_size)
    plt.title("Total favorites within 21 days period for all brands", fontsize=21)
    eng_col2.pyplot()

    arr_rt = engagement_rate_df.groupby(['username'])['retweet_count'].sum().array
    arr_fav = engagement_rate_df.groupby(['username'])['favorite_count'].sum().array

    data_rt = {
        'Brand': brands,
        'Retweet count': arr_rt
    }

    data_fav = {
        'Brand': brands,
        'Favorite count': arr_fav
    }

    data_rt_fav = {
        'Brand': brands,
        'Retweet count': arr_rt,
        'Favorite count': arr_fav
    }

    df_rt = pd.DataFrame(data_rt)
    df_fav = pd.DataFrame(data_fav)
    df_rt_fav = pd.DataFrame(data_rt_fav)

    df_rt = df_rt.set_index('Brand')
    df_fav = df_fav.set_index('Brand')
    df_rt_fav = df_rt_fav.set_index('Brand')

    # eng_col1.bar_chart(df_rt)
    # eng_col2.bar_chart(df_fav)

    st.table(df_rt_fav.transpose())

    # total both rt and fav
    engagement_rate_df['engagement_actions'] = engagement_rate_df['retweet_count'] + engagement_rate_df[
        'favorite_count']
    total_fav_rt = engagement_rate_df.groupby(['username'])['engagement_actions'].sum().array
    followers_brand = followers_count_df['followers_count']['2021-02-23'].array
    # engagement_rate_percentage = (total_fav_rt / followers_brand) * 100
    # for i in range(len(engagement_rate_percentage)):
    #     engagement_rate_percentage[i] = round(engagement_rate_percentage[i], 3)

    engagement_rate_df.loc[engagement_rate_df.username == 'GrabMY', 'followers'] = followers_brand[0]
    engagement_rate_df.loc[engagement_rate_df.username == 'LazadaMY', 'followers'] = followers_brand[1]
    engagement_rate_df.loc[engagement_rate_df.username == 'ShopeeMY', 'followers'] = followers_brand[2]
    engagement_rate_df.loc[engagement_rate_df.username == 'watsonsmy', 'followers'] = followers_brand[3]

    engagement_rate_df["engagement_rate"] = 0
    engagement_rate_df["engagement_rate"] = (engagement_rate_df["engagement_actions"] /
                                             engagement_rate_df["followers"]) * 100

    # st.table(engagement_rate_df)
    plot_engagement = engagement_rate_df.groupby(['username'])['engagement_rate'].mean()
    engagement_percent = [plot_engagement[0], plot_engagement[1], plot_engagement[2], plot_engagement[3]]

    plot_er3 = pd.DataFrame({'screen_name': brands, 'engagement_rate(%)': engagement_percent},
                            columns=['screen_name', 'engagement_rate(%)'])
    plt.figure(figsize=(8, 6))
    plots = sns.barplot(x="screen_name", y="engagement_rate(%)", data=plot_er3, palette=palette)

    for bar in plots.patches:
        plots.annotate(format(bar.get_height(), '.3f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=12, xytext=(0, 8),
                       textcoords='offset points')

    # change here
    plt.title("Avg Engagement rate within 21 days (03/02/2021 - 21/02/2021)", fontsize=21)
    plt.xticks(rotation=360)
    plt.xlabel("Brand", fontsize=font_size)
    plt.ylabel("Engagement Rate (%)", fontsize=font_size)
    st.pyplot()

    st.header('4. Social Share of Voice (SSOV)')
    # ==========================================
    brands_mentions = ssov_pr_df['query'].value_counts().sort_index().array
    total_mentions = brands_mentions.sum()
    ssov_brands = (brands_mentions / total_mentions) * 100
    for i in range(len(ssov_brands)):
        ssov_brands[i] = round(ssov_brands[i], 2)

    plot_ssov = pd.DataFrame({'screen_name': brands, 'ssov(%)': ssov_brands}, columns=['screen_name', 'ssov(%)'])
    fig, ax = plt.subplots(figsize=(6, 6))
    _, _, autotexts = plt.pie(plot_ssov['ssov(%)'], labels=plot_ssov['screen_name'], colors=palette, autopct='%1.2f%%')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_size(font_size)
    plt.title("Social Share of Voice for all brands", fontsize=font_size)
    st.pyplot()

    st.text('text')
    st.warning('warning')
    st.success('success')
    st.error('error')
    st.info('info')
    st.code('code')
    st.markdown('markdown :heart: :bird: :blue_heart: :globe_with_meridians:')
    st.echo('echo')
    # st.help('help')

    st.header('5. Potential Reach')
    # ==========================================
    ssov_pr_df = ssov_pr_df.drop_duplicates(subset=['username'])
    audience = ssov_pr_df.groupby(['query'])['user_follower_count'].sum()
    brands_mentions_no_duplicate = ssov_pr_df['query'].value_counts().sort_index().array
    tr = brands_mentions_no_duplicate * audience
    pr = tr * 0.035
    pr = pr.astype(int)

    fig, ax = plt.subplots(figsize=(15, 7))
    pr.plot(kind='bar', color=palette)
    plt.title("Potential Reach (Audience) within 21 days period for all brands.", fontsize=21)
    plt.xticks(rotation=360)
    plt.xlabel("Brand", fontsize=15)
    plt.ylabel("Audience (in ten million)", fontsize=15)
    st.pyplot()

    data_pr = {
        'Brand': brands,
        'Retweet count': pr.array
    }
    df_pr = pd.DataFrame(data_pr)
    df_pr = df_pr.set_index('Brand')
    st.table(df_pr.transpose())

# ----------------------------------------------------------------------------------------------------------------
if page == 'Assignment 2':
    st.title('Assignment 2')
