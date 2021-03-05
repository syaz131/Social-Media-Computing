import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import community
from numpy import NaN
import datetime
import time
import seaborn as sns
import csv
import networkx as nx
from operator import itemgetter

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


@st.cache(allow_output_mutation=True)
def load_grabmy_node():
    with open('Master_Nodelist-GrabMY.csv', newline='', encoding="utf-8") as fr:
        noderead = csv.reader(fr)
        nodeRow = [n for n in noderead][1:]
    return nodeRow


@st.cache(allow_output_mutation=True)
def load_grabmy_edge():
    with open('Master_Edgelist-GrabMY.csv', newline='', encoding="utf-8") as fr:
        edgeread = csv.reader(fr)
        edgeRow = [n for n in edgeread][1:]
    return edgeRow


@st.cache(allow_output_mutation=True)
def load_lazadamy_node():
    with open('Master_Nodelist-LazadaMY.csv', newline='', encoding="utf-8") as fr:
        noderead = csv.reader(fr)
        nodeRow = [n for n in noderead][1:]
    return nodeRow


@st.cache(allow_output_mutation=True)
def load_lazadamy_edge():
    with open('Master_Edgelist-LazadaMY.csv', newline='', encoding="utf-8") as fr:
        edgeread = csv.reader(fr)
        edgeRow = [n for n in edgeread][1:]
    return edgeRow


@st.cache(allow_output_mutation=True)
def load_shopeemy_node():
    with open('Master_Nodelist-ShopeeMY.csv', newline='', encoding="utf-8") as fr:
        noderead = csv.reader(fr)
        nodeRow = [n for n in noderead][1:]
    return nodeRow


@st.cache(allow_output_mutation=True)
def load_shopeemy_edge():
    with open('Master_Edgelist-ShopeeMY.csv', newline='', encoding="utf-8") as fr:
        edgeread = csv.reader(fr)
        edgeRow = [n for n in edgeread][1:]
    return edgeRow


@st.cache(allow_output_mutation=True)
def load_watsonsmy_node():
    with open('Master_Nodelist-watsonsmy.csv', newline='', encoding="utf-8") as fr:
        noderead = csv.reader(fr)
        nodeRow = [n for n in noderead][1:]
    return nodeRow


@st.cache(allow_output_mutation=True)
def load_watsonsmy_edge():
    with open('Master_Edgelist-watsonsmy.csv', newline='', encoding="utf-8") as fr:
        edgeread = csv.reader(fr)
        edgeRow = [n for n in edgeread][1:]
    return edgeRow


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

    st.write('The brand that encountered the highest growth rate of followers in Twitter is ShopeeMY during the '
             'period of 21 days where data were collected. ShopeeMY achieved a peak of daily growth rate of 0.55% '
             'which comparatively 2 times higher than the peak of any other of the brands. Throughout the three weeks '
             'time, ShopeeMY managed to maintain the daily highest growth rate of followers on the majority of the '
             'days.')
    st.write('On the other hand, LazadaMY recorded the lowest growth rate of followers in their Twitter account. '
             'During the 21 days period, this brand recorded a negative growth rate (loss of followers) for 5 days.')
    st.write('One of the patterns that our team discovered is that most of the brands recorded a positive growth rate '
             'during the festive seasons of Chinese New Year together with Valentine Days.')

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
    response_time_df["@watsonsmy"] = response_time_df["@watsonsmy"].replace(NaN,
                                                                            response_time_df["@watsonsmy"].median())

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
    # st.area_chart(df_response_new)
    df_response = df_response.rename(columns={'dates': 'index'}).set_index('index')
    st.line_chart(df_response)
    # st.area_chart(df_response)

    if st.checkbox('Show response time data'):
        st.subheader('Response Time Data')
        st.table(df_response.transpose())

    # fig, ax = plt.subplots(figsize=(15, 7))
    # boxplot_rt = plt.boxplot(response_time_df, patch_artist=True)
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
    # plt.title("Distribution of response time within 21 days period for all brands", fontsize=21)
    # st.pyplot()

    st.write('Looking at the distribution of the boxplot and the trend of the line graph for the response time for '
             'all brands, LazadaMY has the lowest median response time in comparison. This means that LazadaMY is the '
             'fastest in responding to a tweet that they were mentioned.On the other hand, watsonsmy recorded the '
             'highest median response time thus is the slowest in giving a response.')
    st.write('The length of the box for ShopeeMy is the smallest showing that the brand is consistent in giving '
             'response at a short period of time after recording the lowest median response time. GrabMY has the '
             'biggest box length showing that the response time for this brand is inconsistent during the 21 days '
             'period of time.')
    st.write('All of the brands recorded several outliers during the 21 periods of time. With that being said, '
             'all the brands took a relatively longer time compared to usual average response time. At one point, '
             'watsonsmy recorded an extreme outlier on one of the day due to some reasons that causes inactivity for '
             'its brand account.')

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
    plt.title("Average engagement rate(per post) within 21 days", fontsize=21)
    plt.xticks(rotation=360)
    plt.xlabel("Brand", fontsize=font_size)
    plt.ylabel("Engagement Rate (%)", fontsize=font_size)
    st.pyplot()

    st.write('During the 21 days period, ShoppeMY achieved the highest engagement rate with 0.896% while GrabMY '
             'achieved the lowest engagement rate with 0.004%. ShopeeMY recorded the highest engagement rate due to '
             'several reasons. First, ShopeeMY twitter account is actively posting or "tweeting" on a daily basis '
             'compared to the other brands.')
    st.write('Some of their tweets required interaction such as likes or retweets from '
             'accounts that are following them. For example, such tweets are posts regarding useful selling products '
             'and posts with a sense of humour. In addition, the language that the account used is non-formal with a '
             'mix of few languages to engage more with its followers. On the other hand, GrabMY achieved the lowest '
             'engagement rate due to the main factor that the account only posted once during the 21 days period.')

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

    st.write('ShopeeMY recorded the highest percentage of social share of voice and coming close in second is '
             'watsonsmy. This mean that these two brands are among the most mentioned brands in the digital economy '
             'domain in comparison with LazadaMY and GrabMY. There are several reasons that contribute to this '
             'outcome. First, these two brands constantly promote their products in Twitter that catch the attention '
             'of users hence being mentioned more compared to others.')

    # st.text('text')
    # st.warning('warning')
    # st.success('success')
    # st.error('error')
    # st.info('info')
    # st.code('code')
    # st.markdown('markdown :heart: :bird: :blue_heart: :globe_with_meridians:')
    # st.echo('echo')
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
        'Potential Reach': pr.array
    }
    df_pr = pd.DataFrame(data_pr)
    df_pr = df_pr.set_index('Brand')
    st.table(df_pr.transpose())
    st.write('GrabMY recorded the highest potential reach compared to the other brands in the domain. The reason for '
             'this outcome is due to the user mentioned this brand has a huge number of followers. One of the '
             'possibilities is that the users that mentioned GrabMY are well-known individuals such as influencers '
             'and celebrities.')

    # st.text_area('insert input')

# ----------------------------------------------------------------------------------------------------------------
if page == 'Assignment 2':
    st.title('Assignment 2')

    # grab Malaysia
    nodesGrab = load_grabmy_node()
    nodeGrab_names = [n[0] for n in nodesGrab]
    edgesGrab = load_grabmy_edge()
    G_grab = nx.Graph()
    G_grab.add_nodes_from(nodeGrab_names)
    G_grab.add_edges_from(edgesGrab)

    location_dict = {}
    created_at_dict = {}
    followers_count_dict = {}
    friends_count_dict = {}
    relationship_dict = {}

    for user in nodesGrab:
        location_dict[user[0]] = user[1]
        created_at_dict[user[0]] = user[2]
        followers_count_dict[user[0]] = user[3]
        friends_count_dict[user[0]] = user[4]
        relationship_dict[user[0]] = user[5]

    nx.set_node_attributes(G_grab, name='location', values=location_dict)
    nx.set_node_attributes(G_grab, name='created_at', values=created_at_dict)
    nx.set_node_attributes(G_grab, name='followers_count', values=followers_count_dict)
    nx.set_node_attributes(G_grab, name='friends_count', values=friends_count_dict)
    nx.set_node_attributes(G_grab, name='relationship', values=relationship_dict)

    # lazada Malaysia
    nodesLazada = load_lazadamy_node()
    nodeLazada_names = [n[0] for n in nodesLazada]
    edgesLazada = load_lazadamy_edge()
    G_lazada = nx.Graph()
    G_lazada.add_nodes_from(nodeLazada_names)
    G_lazada.add_edges_from(edgesLazada)

    location_dict = {}
    created_at_dict = {}
    followers_count_dict = {}
    friends_count_dict = {}
    relationship_dict = {}

    for user in nodesLazada:
        location_dict[user[0]] = user[1]
        created_at_dict[user[0]] = user[2]
        followers_count_dict[user[0]] = user[3]
        friends_count_dict[user[0]] = user[4]
        relationship_dict[user[0]] = user[5]

    nx.set_node_attributes(G_lazada, name='location', values=location_dict)
    nx.set_node_attributes(G_lazada, name='created_at', values=created_at_dict)
    nx.set_node_attributes(G_lazada, name='followers_count', values=followers_count_dict)
    nx.set_node_attributes(G_lazada, name='friends_count', values=friends_count_dict)
    nx.set_node_attributes(G_lazada, name='relationship', values=relationship_dict)

    # shopee Malaysia
    nodesShopee = load_shopeemy_node()
    nodeShopee_names = [n[0] for n in nodesShopee]
    edgesShopee = load_shopeemy_edge()
    G_shopee = nx.Graph()
    G_shopee.add_nodes_from(nodeShopee_names)
    G_shopee.add_edges_from(edgesShopee)

    location_dict = {}
    created_at_dict = {}
    followers_count_dict = {}
    friends_count_dict = {}
    relationship_dict = {}

    for user in nodesShopee:
        location_dict[user[0]] = user[1]
        created_at_dict[user[0]] = user[2]
        followers_count_dict[user[0]] = user[3]
        friends_count_dict[user[0]] = user[4]
        relationship_dict[user[0]] = user[5]

    nx.set_node_attributes(G_shopee, name='location', values=location_dict)
    nx.set_node_attributes(G_shopee, name='created_at', values=created_at_dict)
    nx.set_node_attributes(G_shopee, name='followers_count', values=followers_count_dict)
    nx.set_node_attributes(G_shopee, name='friends_count', values=friends_count_dict)
    nx.set_node_attributes(G_shopee, name='relationship', values=relationship_dict)

    # watsons Malaysia
    nodesWatsons = load_watsonsmy_node()
    nodeWatsons_names = [n[0] for n in nodesWatsons]
    edgesWatsons = load_watsonsmy_edge()
    G_watsons = nx.Graph()
    G_watsons.add_nodes_from(nodeWatsons_names)
    G_watsons.add_edges_from(edgesWatsons)

    location_dict = {}
    created_at_dict = {}
    followers_count_dict = {}
    friends_count_dict = {}
    relationship_dict = {}

    for user in nodesWatsons:
        location_dict[user[0]] = user[1]
        created_at_dict[user[0]] = user[2]
        followers_count_dict[user[0]] = user[3]
        friends_count_dict[user[0]] = user[4]
        relationship_dict[user[0]] = user[5]

    nx.set_node_attributes(G_watsons, name='location', values=location_dict)
    nx.set_node_attributes(G_watsons, name='created_at', values=created_at_dict)
    nx.set_node_attributes(G_watsons, name='followers_count', values=followers_count_dict)
    nx.set_node_attributes(G_watsons, name='friends_count', values=friends_count_dict)
    nx.set_node_attributes(G_watsons, name='relationship', values=relationship_dict)

    # ========================================= Graph Data ============================

    val_map_shopee = {'friends': 1.0, 'followers': 0.75}
    val_map_watsons = {'friends': 0.9, 'followers': 0.3}
    val_map_grab = {'friends': 0.9, 'followers': 0.35}
    val_map = {'friends': 1.0, 'followers': 0.0}
    font_map = {'friends': 500, 'followers': 100}

    # grab ==============
    posGrab = nx.spring_layout(G_grab, scale=100)
    values_grab = [val_map_grab.get(node[1], 0.25) for node in G_grab.nodes.data('relationship')]
    values_nodes_grab = [font_map.get(node[1], 500) for node in G_grab.nodes.data('relationship')]

    labels_grab = {}
    labels_grab['GrabMY'] = 'GrabMY'
    for node in G_grab.nodes.data('relationship'):
        if node[1] == 'friends':
            labels_grab[node[0]] = node[0]

    between_dict_grab = nx.betweenness_centrality(G_grab)
    eigen_dict_grab = nx.eigenvector_centrality(G_grab, max_iter=600)
    deg_dict_grab = nx.degree_centrality(G_grab)
    nx.set_node_attributes(G_grab, name='betweenness', values=between_dict_grab)
    nx.set_node_attributes(G_grab, name='eigenvector', values=eigen_dict_grab)
    nx.set_node_attributes(G_grab, name='degree centrality', values=deg_dict_grab)

    # lazada ==============
    posLazada = nx.spring_layout(G_lazada, scale=100)
    values_lazada = [val_map.get(node[1], 0.25) for node in G_lazada.nodes.data('relationship')]
    values_nodes_lazada = [font_map.get(node[1], 500) for node in G_lazada.nodes.data('relationship')]

    labels_lazada = {}
    labels_lazada['LazadaMY'] = 'LazadaMY'
    for node in G_lazada.nodes.data('relationship'):
        if node[1] == 'friends':
            labels_lazada[node[0]] = node[0]

    between_dict_lazada = nx.betweenness_centrality(G_lazada)
    eigen_dict_lazada = nx.eigenvector_centrality(G_lazada, max_iter=600)
    deg_dict_lazada = nx.degree_centrality(G_lazada)
    nx.set_node_attributes(G_lazada, name='betweenness', values=between_dict_lazada)
    nx.set_node_attributes(G_lazada, name='eigenvector', values=eigen_dict_lazada)
    nx.set_node_attributes(G_lazada, name='degree centrality', values=deg_dict_lazada)

    # shopee ==============
    posShopee = nx.spring_layout(G_shopee, scale=100)
    values_shopee = [val_map_shopee.get(node[1], 0.25) for node in G_shopee.nodes.data('relationship')]
    values_nodes_shopee = [font_map.get(node[1], 500) for node in G_shopee.nodes.data('relationship')]

    labels_shopee = {}
    labels_shopee['ShopeeMY'] = 'ShopeeMY'
    for node in G_shopee.nodes.data('relationship'):
        if node[1] == 'friends':
            labels_shopee[node[0]] = node[0]

    between_dict_shopee = nx.betweenness_centrality(G_shopee)
    eigen_dict_shopee = nx.eigenvector_centrality(G_shopee, max_iter=600)
    deg_dict_shopee = nx.degree_centrality(G_shopee)
    nx.set_node_attributes(G_shopee, name='betweenness', values=between_dict_shopee)
    nx.set_node_attributes(G_shopee, name='eigenvector', values=eigen_dict_shopee)
    nx.set_node_attributes(G_shopee, name='degree centrality', values=deg_dict_shopee)

    # watsons ==============
    posWatsons = nx.spring_layout(G_watsons, scale=100)
    values_watsons = [val_map_watsons.get(node[1], 0.25) for node in G_watsons.nodes.data('relationship')]
    values_nodes_watsons = [font_map.get(node[1], 500) for node in G_watsons.nodes.data('relationship')]

    labels_watsons = {}
    labels_watsons['watsonsmy'] = 'watsonsmy'
    for node in G_watsons.nodes.data('relationship'):
        if node[1] == 'friends':
            labels_watsons[node[0]] = node[0]

    between_dict_watsons = nx.betweenness_centrality(G_watsons)
    eigen_dict_watsons = nx.eigenvector_centrality(G_watsons)
    deg_dict_watsons = nx.degree_centrality(G_watsons)
    nx.set_node_attributes(G_watsons, name='betweenness', values=between_dict_watsons)
    nx.set_node_attributes(G_watsons, name='eigenvector', values=eigen_dict_watsons)
    nx.set_node_attributes(G_watsons, name='degree centrality', values=deg_dict_watsons)

    # ================================================ Data ===============================

    # ================ grab ===============================
    # betweeness data
    sorted_bet = sorted(between_dict_grab.items(), key=itemgetter(1), reverse=True)
    top_sorted_bet = sorted_bet[1:11]
    id_ = []
    val = []
    for i in reversed(range(len(top_sorted_bet))):
        id_.append(top_sorted_bet[i][0])
        val.append(top_sorted_bet[i][1])
    betw_data = {
        'Account': id_,
        'Betweeness Centrality': val
    }
    betw_df = pd.DataFrame(betw_data)
    grab_betw_df = betw_df.sort_values('Betweeness Centrality', ascending=True)

    # eigenvector data
    sorted_eigen = sorted(eigen_dict_grab.items(), key=itemgetter(1), reverse=True)
    top_sorted_eigen = sorted_eigen[1:11]
    id_ = []
    val = []
    for i in reversed(range(len(top_sorted_eigen))):
        id_.append(top_sorted_eigen[i][0])
        val.append(top_sorted_eigen[i][1])
    eigen_data = {
        'Account': id_,
        'Eigenvector Centrality': val
    }
    eigen_df = pd.DataFrame(eigen_data)
    grab_eigen_df = eigen_df.sort_values('Eigenvector Centrality', ascending=True)

    # degCentral data
    sorted_degCentral = sorted(deg_dict_grab.items(), key=itemgetter(1), reverse=True)
    top_sorted_degCentral = sorted_degCentral[1:11]
    id_ = []
    val = []
    for i in reversed(range(len(top_sorted_degCentral))):
        id_.append(top_sorted_degCentral[i][0])
        val.append(top_sorted_degCentral[i][1])
    degCentral_data = {
        'Account': id_,
        'Degree of Centrality': val
    }
    degCentral_df = pd.DataFrame(degCentral_data)
    grab_degCentral_df = degCentral_df.sort_values('Degree of Centrality', ascending=True)

    # grab_communities = community.best_partition(G_grab)
    # nx.set_node_attributes(G_grab, values=grab_communities, name="modularity")
    # grab_modularity = {}  # Create a new, empty dictionary

    # ================ lazada ===============================
    # betweeness data
    sorted_bet = sorted(between_dict_lazada.items(), key=itemgetter(1), reverse=True)
    top_sorted_bet = sorted_bet[1:11]
    id_ = []
    val = []
    for i in reversed(range(len(top_sorted_bet))):
        id_.append(top_sorted_bet[i][0])
        val.append(top_sorted_bet[i][1])
    betw_data = {
        'Account': id_,
        'Betweeness Centrality': val
    }
    betw_df = pd.DataFrame(betw_data)
    lazada_betw_df = betw_df.sort_values('Betweeness Centrality', ascending=True)

    # eigenvector data
    sorted_eigen = sorted(eigen_dict_lazada.items(), key=itemgetter(1), reverse=True)
    top_sorted_eigen = sorted_eigen[1:11]
    id_ = []
    val = []
    for i in reversed(range(len(top_sorted_eigen))):
        id_.append(top_sorted_eigen[i][0])
        val.append(top_sorted_eigen[i][1])
    eigen_data = {
        'Account': id_,
        'Eigenvector Centrality': val
    }
    eigen_df = pd.DataFrame(eigen_data)
    lazada_eigen_df = eigen_df.sort_values('Eigenvector Centrality', ascending=True)

    # degCentral data
    sorted_degCentral = sorted(deg_dict_lazada.items(), key=itemgetter(1), reverse=True)
    top_sorted_degCentral = sorted_degCentral[1:11]
    id_ = []
    val = []
    for i in reversed(range(len(top_sorted_degCentral))):
        id_.append(top_sorted_degCentral[i][0])
        val.append(top_sorted_degCentral[i][1])
    degCentral_data = {
        'Account': id_,
        'Degree of Centrality': val
    }
    degCentral_df = pd.DataFrame(degCentral_data)
    lazada_degCentral_df = degCentral_df.sort_values('Degree of Centrality', ascending=True)

    # lazada_communities = community.best_partition(G_lazada)
    # nx.set_node_attributes(G_lazada, values=lazada_communities, name="modularity")
    # lazada_modularity = {}  # Create a new, empty dictionary

    # ================ shopee ===============================
    # betweeness data
    sorted_bet = sorted(between_dict_shopee.items(), key=itemgetter(1), reverse=True)
    top_sorted_bet = sorted_bet[1:11]
    id_ = []
    val = []
    for i in reversed(range(len(top_sorted_bet))):
        id_.append(top_sorted_bet[i][0])
        val.append(top_sorted_bet[i][1])
    betw_data = {
        'Account': id_,
        'Betweeness Centrality': val
    }
    betw_df = pd.DataFrame(betw_data)
    shopee_betw_df = betw_df.sort_values('Betweeness Centrality', ascending=True)

    # eigenvector data
    sorted_eigen = sorted(eigen_dict_shopee.items(), key=itemgetter(1), reverse=True)
    top_sorted_eigen = sorted_eigen[1:11]
    id_ = []
    val = []
    for i in reversed(range(len(top_sorted_eigen))):
        id_.append(top_sorted_eigen[i][0])
        val.append(top_sorted_eigen[i][1])
    eigen_data = {
        'Account': id_,
        'Eigenvector Centrality': val
    }
    eigen_df = pd.DataFrame(eigen_data)
    shopee_eigen_df = eigen_df.sort_values('Eigenvector Centrality', ascending=True)

    # degCentral data
    sorted_degCentral = sorted(deg_dict_shopee.items(), key=itemgetter(1), reverse=True)
    top_sorted_degCentral = sorted_degCentral[1:11]
    id_ = []
    val = []
    for i in reversed(range(len(top_sorted_degCentral))):
        id_.append(top_sorted_degCentral[i][0])
        val.append(top_sorted_degCentral[i][1])
    degCentral_data = {
        'Account': id_,
        'Degree of Centrality': val
    }
    degCentral_df = pd.DataFrame(degCentral_data)
    shopee_degCentral_df = degCentral_df.sort_values('Degree of Centrality', ascending=True)

    # shopee_communities = community.best_partition(G_shopee)
    # nx.set_node_attributes(G_shopee, values=shopee_communities, name="modularity")
    # shopee_modularity = {}  # Create a new, empty dictionary

    # ================ watsons ===============================
    # betweeness data
    sorted_bet = sorted(between_dict_watsons.items(), key=itemgetter(1), reverse=True)
    top_sorted_bet = sorted_bet[1:11]
    id_ = []
    val = []
    for i in reversed(range(len(top_sorted_bet))):
        id_.append(top_sorted_bet[i][0])
        val.append(top_sorted_bet[i][1])
    betw_data = {
        'Account': id_,
        'Betweeness Centrality': val
    }
    betw_df = pd.DataFrame(betw_data)
    watsons_betw_df = betw_df.sort_values('Betweeness Centrality', ascending=True)

    # eigenvector data
    sorted_eigen = sorted(eigen_dict_watsons.items(), key=itemgetter(1), reverse=True)
    top_sorted_eigen = sorted_eigen[1:11]
    id_ = []
    val = []
    for i in reversed(range(len(top_sorted_eigen))):
        id_.append(top_sorted_eigen[i][0])
        val.append(top_sorted_eigen[i][1])
    eigen_data = {
        'Account': id_,
        'Eigenvector Centrality': val
    }
    eigen_df = pd.DataFrame(eigen_data)
    watsons_eigen_df = eigen_df.sort_values('Eigenvector Centrality', ascending=True)

    # degCentral data
    sorted_degCentral = sorted(deg_dict_watsons.items(), key=itemgetter(1), reverse=True)
    top_sorted_degCentral = sorted_degCentral[1:11]
    id_ = []
    val = []
    for i in reversed(range(len(top_sorted_degCentral))):
        id_.append(top_sorted_degCentral[i][0])
        val.append(top_sorted_degCentral[i][1])
    degCentral_data = {
        'Account': id_,
        'Degree of Centrality': val
    }
    degCentral_df = pd.DataFrame(degCentral_data)
    watsons_degCentral_df = degCentral_df.sort_values('Degree of Centrality', ascending=True)

    # watsons_communities = community.best_partition(G_watsons)
    # nx.set_node_attributes(G_watsons, values=watsons_communities, name="modularity")
    # watsons_modularity = {}  # Create a new, empty dictionary

    # --------------------------- Grab -----------------------------------------------
    st.info('Grab Malaysia Network')
    st.header('Grab Network Graph')
    brandColor = palette[0]
    plt.figure(figsize=(20, 20))
    nx.draw(G_grab, posGrab, cmap=plt.get_cmap('cool'), edge_color='gray',
            node_size=values_nodes_grab, node_color=values_grab, with_labels=False, width=0.5, font_color='black')
    # nx.draw_networkx_labels(G_grab, posGrab, labels_grab, font_size=20, font_color='black', font_weight='bold')
    st.pyplot()

    st.subheader('Graph Info')
    st.write("Number of nodes:", G_grab.number_of_nodes())
    st.write("Number of edges:", G_grab.number_of_edges())
    st.write("Network density:", nx.density(G_grab))

    st.header('Centrality Ranking')
    deg_col1, deg_col2 = st.beta_columns(2)
    deg_col1.markdown('#### Degree of Centrality')
    plt.figure(figsize=(4, 4.7))
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.barh(grab_degCentral_df['Account'].array, grab_degCentral_df['Degree of Centrality'].array,
             color=brandColor, height=0.5)
    deg_col1.pyplot()
    deg_col2.markdown('#### Degree of Centrality Data')
    grab_degCentral_df = grab_degCentral_df.sort_values('Degree of Centrality', ascending=False)
    grab_degCentral_df = grab_degCentral_df.set_index('Account')
    deg_col2.write(grab_degCentral_df)

    betw_col1, betw_col2 = st.beta_columns(2)
    betw_col1.markdown('#### Betweeness Centrality')
    plt.figure(figsize=(4, 4.7))
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.barh(grab_betw_df['Account'].array, grab_betw_df['Betweeness Centrality'].array,
             color=brandColor, height=0.5)
    betw_col1.pyplot()
    betw_col2.markdown('#### Betweeness Centrality Data')
    grab_betw_df = grab_betw_df.sort_values('Betweeness Centrality', ascending=False)
    grab_betw_df = grab_betw_df.set_index('Account')
    betw_col2.write(grab_betw_df)

    eigen_col1, eigen_col2 = st.beta_columns(2)
    eigen_col1.markdown('#### Eigenvector Centrality')
    plt.figure(figsize=(4, 4.7))
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.barh(grab_eigen_df['Account'].array, grab_eigen_df['Eigenvector Centrality'].array,
             color=brandColor, height=0.5)
    eigen_col1.pyplot()
    eigen_col2.markdown('#### Eigenvector Centrality Data')
    grab_eigen_df = grab_eigen_df.sort_values('Eigenvector Centrality', ascending=False)
    grab_eigen_df = grab_eigen_df.set_index('Account')
    eigen_col2.write(grab_eigen_df)

    st.header('Grab Community')

    # --------------------------- Lazada ---------------------------------------------
    st.warning('Lazada Malaysia Network')
    st.header('Lazada Network Graph')
    brandColor = palette[1]
    plt.figure(figsize=(20, 20))
    nx.draw(G_lazada, posLazada, cmap=plt.get_cmap('viridis'), edge_color='gray',
            node_size=values_nodes_lazada, node_color=values_lazada, with_labels=False, width=0.5, font_color='black')
    nx.draw_networkx_labels(G_lazada, posLazada, labels_lazada, font_size=20, font_color='black', font_weight='bold')
    st.pyplot()
    st.subheader('Graph Info')
    st.write("Number of nodes:", G_lazada.number_of_nodes())
    st.write("Number of edges:", G_lazada.number_of_edges())
    st.write("Network density:", nx.density(G_lazada))

    st.header('Centrality Ranking')
    deg_col1, deg_col2 = st.beta_columns(2)
    deg_col1.markdown('#### Degree of Centrality')
    plt.figure(figsize=(4, 4.7))
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.barh(lazada_degCentral_df['Account'].array, lazada_degCentral_df['Degree of Centrality'].array,
             color=brandColor, height=0.5)
    deg_col1.pyplot()
    deg_col2.markdown('#### Degree of Centrality Data')
    lazada_degCentral_df = lazada_degCentral_df.sort_values('Degree of Centrality', ascending=False)
    lazada_degCentral_df = lazada_degCentral_df.set_index('Account')
    deg_col2.write(lazada_degCentral_df)

    betw_col1, betw_col2 = st.beta_columns(2)
    betw_col1.markdown('#### Betweeness Centrality')
    plt.figure(figsize=(4, 4.7))
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.barh(lazada_betw_df['Account'].array, lazada_betw_df['Betweeness Centrality'].array,
             color=brandColor, height=0.5)
    betw_col1.pyplot()
    betw_col2.markdown('#### Betweeness Centrality Data')
    lazada_betw_df = lazada_betw_df.sort_values('Betweeness Centrality', ascending=False)
    lazada_betw_df = lazada_betw_df.set_index('Account')
    betw_col2.write(lazada_betw_df)

    eigen_col1, eigen_col2 = st.beta_columns(2)
    eigen_col1.markdown('#### Eigenvector Centrality')
    plt.figure(figsize=(4, 4.7))
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.barh(lazada_eigen_df['Account'].array, lazada_eigen_df['Eigenvector Centrality'].array,
             color=brandColor, height=0.5)
    eigen_col1.pyplot()
    eigen_col2.markdown('#### Eigenvector Centrality Data')
    lazada_eigen_df = lazada_eigen_df.sort_values('Eigenvector Centrality', ascending=False)
    lazada_eigen_df = lazada_eigen_df.set_index('Account')
    eigen_col2.write(lazada_eigen_df)

    # st.header('Lazada Community')
    # for k, v in lazada_communities.items():
    #     if v not in lazada_modularity:
    #         lazada_modularity[v] = [k]
    #     else:
    #         lazada_modularity[v].append(k)
    #
    # comm_col1, comm_col2, comm_col3 = st.beta_columns(3)
    # count = 0
    # for k, v in lazada_modularity.items():
    #     if len(v) > 2:
    #         df_community = pd.DataFrame(v, columns=['Account'])
    #         if count % 3 == 0:
    #             comm_col1.write('CLASS ' + str(k) + ' : ' + str(len(lazada_modularity[k])) + ' people\n')
    #             comm_col1.dataframe(df_community)
    #         if count % 3 == 1:
    #             comm_col2.write('CLASS ' + str(k) + ' : ' + str(len(lazada_modularity[k])) + ' people\n')
    #             comm_col2.dataframe(df_community)
    #         if count % 3 == 2:
    #             comm_col3.write('CLASS ' + str(k) + ' : ' + str(len(lazada_modularity[k])) + ' people\n')
    #             comm_col3.dataframe(df_community)
    #         count += 1

    # --------------------------- Shopee ---------------------------------------------
    st.error('Shopee Malaysia Network')
    st.header('Shopee Network Graph')
    brandColor = palette[2]
    plt.figure(figsize=(20, 20))
    nx.draw(G_shopee, posShopee, cmap=plt.get_cmap('plasma'), edge_color='gray',
            node_size=values_nodes_shopee, node_color=values_shopee, with_labels=False, width=0.5, font_color='black')
    nx.draw_networkx_labels(G_shopee, posShopee, labels_shopee, font_size=20, font_color='black', font_weight='bold')
    st.pyplot()
    st.subheader('Graph Info')
    st.write("Number of nodes:", G_shopee.number_of_nodes())
    st.write("Number of edges:", G_shopee.number_of_edges())
    st.write("Network density:", nx.density(G_shopee))

    st.header('Centrality Ranking')
    deg_col1, deg_col2 = st.beta_columns(2)
    deg_col1.markdown('#### Degree of Centrality')
    plt.figure(figsize=(4, 4.7))
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.barh(shopee_degCentral_df['Account'].array, shopee_degCentral_df['Degree of Centrality'].array,
             color=brandColor, height=0.5)
    deg_col1.pyplot()
    deg_col2.markdown('#### Degree of Centrality Data')
    shopee_degCentral_df = shopee_degCentral_df.sort_values('Degree of Centrality', ascending=False)
    shopee_degCentral_df = shopee_degCentral_df.set_index('Account')
    deg_col2.write(shopee_degCentral_df)

    betw_col1, betw_col2 = st.beta_columns(2)
    betw_col1.markdown('#### Betweeness Centrality')
    plt.figure(figsize=(4, 4.7))
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.barh(shopee_betw_df['Account'].array, shopee_betw_df['Betweeness Centrality'].array,
             color=brandColor, height=0.5)
    betw_col1.pyplot()
    betw_col2.markdown('#### Betweeness Centrality Data')
    shopee_betw_df = shopee_betw_df.sort_values('Betweeness Centrality', ascending=False)
    shopee_betw_df = shopee_betw_df.set_index('Account')
    betw_col2.write(shopee_betw_df)

    eigen_col1, eigen_col2 = st.beta_columns(2)
    eigen_col1.markdown('#### Eigenvector Centrality')
    plt.figure(figsize=(4, 4.7))
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.barh(shopee_eigen_df['Account'].array, shopee_eigen_df['Eigenvector Centrality'].array,
             color=brandColor, height=0.5)
    eigen_col1.pyplot()
    eigen_col2.markdown('#### Eigenvector Centrality Data')
    shopee_eigen_df = shopee_eigen_df.sort_values('Eigenvector Centrality', ascending=False)
    shopee_eigen_df = shopee_eigen_df.set_index('Account')
    eigen_col2.write(shopee_eigen_df)

    # st.header('Shopeee Community')
    # for k, v in shopee_communities.items():
    #     if v not in shopee_modularity:
    #         shopee_modularity[v] = [k]
    #     else:
    #         shopee_modularity[v].append(k)
    #
    # comm_col1, comm_col2, comm_col3 = st.beta_columns(3)
    # count = 0
    # for k, v in shopee_modularity.items():
    #     if len(v) > 2:
    #         df_community = pd.DataFrame(v, columns=['Account'])
    #         if count % 3 == 0:
    #             comm_col1.write('CLASS ' + str(k) + ' : ' + str(len(shopee_modularity[k])) + ' people\n')
    #             comm_col1.dataframe(df_community)
    #         if count % 3 == 1:
    #             comm_col2.write('CLASS ' + str(k) + ' : ' + str(len(shopee_modularity[k])) + ' people\n')
    #             comm_col2.dataframe(df_community)
    #         if count % 3 == 2:
    #             comm_col3.write('CLASS ' + str(k) + ' : ' + str(len(shopee_modularity[k])) + ' people\n')
    #             comm_col3.dataframe(df_community)
    #         count += 1

    # --------------------- watsons ------------------------------------------------
    # network draw
    st.success('Watsons Malaysia Network')
    st.header('Watsons Network Graph')
    brandColor = palette[3]
    plt.figure(figsize=(20, 20))
    nx.draw(G_watsons, posWatsons, cmap=plt.get_cmap('cool'), edge_color='gray',
            node_size=values_nodes_watsons, node_color=values_watsons, with_labels=False, width=0.5, font_color='black')
    nx.draw_networkx_labels(G_watsons, posWatsons, labels_watsons, font_size=20, font_color='black', font_weight='bold')
    st.pyplot()
    st.subheader('Graph Info')
    st.write("Number of nodes:", G_watsons.number_of_nodes())
    st.write("Number of edges:", G_watsons.number_of_edges())
    st.write("Network density:", nx.density(G_watsons))

    st.header('Centrality Ranking')
    deg_col1, deg_col2 = st.beta_columns(2)
    deg_col1.markdown('#### Degree of Centrality')
    plt.figure(figsize=(4, 4.7))
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.barh(watsons_degCentral_df['Account'].array, watsons_degCentral_df['Degree of Centrality'].array,
             color=brandColor, height=0.5)
    deg_col1.pyplot()
    deg_col2.markdown('#### Degree of Centrality Data')
    watsons_degCentral_df = watsons_degCentral_df.sort_values('Degree of Centrality', ascending=False)
    watsons_degCentral_df = watsons_degCentral_df.set_index('Account')
    deg_col2.write(watsons_degCentral_df)

    betw_col1, betw_col2 = st.beta_columns(2)
    betw_col1.markdown('#### Betweeness Centrality')
    plt.figure(figsize=(4, 4.7))
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.barh(watsons_betw_df['Account'].array, watsons_betw_df['Betweeness Centrality'].array,
             color=brandColor, height=0.5)
    betw_col1.pyplot()
    betw_col2.markdown('#### Betweeness Centrality Data')
    watsons_betw_df = watsons_betw_df.sort_values('Betweeness Centrality', ascending=False)
    watsons_betw_df = watsons_betw_df.set_index('Account')
    betw_col2.write(watsons_betw_df)

    eigen_col1, eigen_col2 = st.beta_columns(2)
    eigen_col1.markdown('#### Eigenvector Centrality')
    plt.figure(figsize=(4, 4.7))
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.barh(watsons_eigen_df['Account'].array, watsons_eigen_df['Eigenvector Centrality'].array,
             color=brandColor, height=0.5)
    eigen_col1.pyplot()
    eigen_col2.markdown('#### Eigenvector Centrality Data')
    watsons_eigen_df = watsons_eigen_df.sort_values('Eigenvector Centrality', ascending=False)
    watsons_eigen_df = watsons_eigen_df.set_index('Account')
    eigen_col2.write(watsons_eigen_df)

    # st.header('Watsons Community')
    # for k, v in watsons_communities.items():
    #     if v not in watsons_modularity:
    #         watsons_modularity[v] = [k]
    #     else:
    #         watsons_modularity[v].append(k)
    #
    # comm_col1, comm_col2, comm_col3 = st.beta_columns(3)
    # count = 0
    # for k, v in watsons_modularity.items():
    #     if len(v) > 2:
    #         df_community = pd.DataFrame(v, columns=['Account'])
    #         if count % 3 == 0:
    #             comm_col1.write('CLASS ' + str(k) + ' : ' + str(len(watsons_modularity[k])) + ' people\n')
    #             comm_col1.dataframe(df_community)
    #         if count % 3 == 1:
    #             comm_col2.write('CLASS ' + str(k) + ' : ' + str(len(watsons_modularity[k])) + ' people\n')
    #             comm_col2.dataframe(df_community)
    #         if count % 3 == 2:
    #             comm_col3.write('CLASS ' + str(k) + ' : ' + str(len(watsons_modularity[k])) + ' people\n')
    #             comm_col3.dataframe(df_community)
    #         count += 1
