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
st.title('DASHBOARD')
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
st.sidebar.header('Social Media Computing')
st.sidebar.subheader('Group Member')
st.sidebar.text('Izzah\t\t1171101738 \nGlenn\t\t1171101736 \nNiroshaan\t1171102016 \nSyazwan\t\t1171102003')

st.sidebar.subheader('Main Menu')
page = st.sidebar.selectbox("Select Page Menu", menu)

engagement_rate_df = load_engagement_data()
followers_count_df = load_follower_data()
response_time_df = load_response_data()
ssov_pr_df = load_ssov_pr()

brands = ['GrabMY', 'LazadaMY', 'ShopeeMY', 'watsonsmy']
palette = ['blue', 'orange', 'green', 'red']
font_size = 18

if page == 'Assignment 1':
    st.title('Assignment 1')

    st.header('1. Growth of followers')
    # =================================
    followers_count_df = followers_count_df.sort_values(['screen_name', 'date'], ascending=[True, True])
    followers_count_df['daily_new_followers'] = followers_count_df['followers_count'].diff()
    followers_count_df["daily_new_followers"] = followers_count_df["daily_new_followers"].replace(NaN, 0)
    followers_count_df["daily_new_followers"] = followers_count_df["daily_new_followers"].replace(-12719.0, 0)
    followers_count_df["daily_new_followers"] = followers_count_df["daily_new_followers"].replace(-19197, 0)
    followers_count_df["daily_new_followers"] = followers_count_df["daily_new_followers"].replace(286156, 0)
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
    st.pyplot()

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

    response_time_df = response_time_df.sort_values(['query', 'dates'], ascending=[True, True])

    arrGrab = response_time_df[response_time_df['query'] == '@grabmy'].groupby(['dates'])['duration'].mean().array
    arrLazada = response_time_df[response_time_df['query'] == '@lazadamy'].groupby(['dates'])['duration'].mean().array
    arrShopee = response_time_df[response_time_df['query'] == '@shopeemy'].groupby(['dates'])['duration'].mean().array
    arrWatsons = response_time_df[response_time_df['query'] == '@watsonsmy'].groupby(['dates'])['duration'].mean().array

    data_response = {
        'dates': response_time_df['dates'].unique(),
        '@grabmy': arrGrab.astype(int),
        '@lazadamy': arrLazada.astype(int),
        '@shopeemy': arrShopee.astype(int),
        '@watsonsmy': arrWatsons.astype(int)
    }

    df_response = pd.DataFrame(data_response)
    df_response_new = pd.DataFrame(df_response[:8], columns=['@grabmy', '@lazadamy', '@shopeemy', '@watsonsmy'])
    # df_response_new = df_response_new.set_index('dates')
    st.area_chart(df_response_new)
    df_response = df_response.rename(columns={'dates': 'index'}).set_index('index')
    # st.line_chart(df_response)

    if st.checkbox('Show response time data'):
        st.subheader('Response Time Data')
        st.table(df_response.transpose())

    fig, ax = plt.subplots(figsize=(15, 7))
    dataframe = response_time_df.groupby(['dates', 'query'])['duration'].mean().unstack()

    # ERROR  - cannot perform reduce with flexible type
    # boxplot_rt = plt.boxplot(dataframe, patch_artist=True)
    #
    # for patch, color in zip(boxplot_rt['boxes'], palette):
    #     patch.set_facecolor(color)
    #
    # for median in boxplot_rt['medians']:
    #     median.set(color='w',
    #                linewidth=2.5)
    # x = [1, 2, 3, 4]
    # plt.xticks(x, brands, rotation=360)
    # plt.xlabel("Brand", fontsize=font_size)
    # plt.ylabel("Time (s)", fontsize=font_size)
    # plt.title("Distribution of response time within 14 days period for all brands", fontsize=21)
    # st.pyplot()

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
    plt.title("Total retweets within 14 days period for all brands", fontsize=21)
    eng_col1.pyplot()

    fig, ax = plt.subplots(figsize=(15, 7))
    plot_er1 = engagement_rate_df.groupby(['username'])['favorite_count'].sum().plot(kind='bar', ax=ax, color=palette)
    plt.xticks(rotation=360)
    plt.xlabel("Brand", fontsize=font_size)
    plt.ylabel("Total favorites", fontsize=font_size)
    plt.title("Total favorites within 14 days period for all brands", fontsize=21)
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
    followers_brand = followers_count_df['followers_count']['2021-02-10'].array
    engagement_rate_percentage = (total_fav_rt / followers_brand) * 100
    for i in range(len(engagement_rate_percentage)):
        engagement_rate_percentage[i] = round(engagement_rate_percentage[i], 3)

    plot_er3 = pd.DataFrame({'screen_name': brands, 'engagement_rate(%)': engagement_rate_percentage},
                            columns=['screen_name', 'engagement_rate(%)'])
    plt.figure(figsize=(8, 6))
    plots = sns.barplot(x="screen_name", y="engagement_rate(%)", data=plot_er3, palette=palette)

    for bar in plots.patches:
        plots.annotate(format(bar.get_height(), '.3f'),
                       (bar.get_x() + bar.get_width() / 2,
                        bar.get_height()), ha='center', va='center',
                       size=12, xytext=(0, 8),
                       textcoords='offset points')

    plt.title("Engagement rate within 14 days (03/02/2021 - 16/02/2021)", fontsize=21)
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
    plt.title("Potential Reach (Audience) within 14 days period for all brands.", fontsize=21)
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
