# import dependencies
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import streamlit as st
from PIL import Image

# Get data file
data = pd.read_csv('/Users/stellalee/Desktop/pythonproject/basketball/nba_2020.csv')
# print(data.shape) # Checking

# Set index to player names
data.set_index('Player', drop = True, inplace = True)
# print(data.head(5)) # Checking

# Drop non-numerical and unnecessary data
cleaned_data = data.drop(['Stage', 'Team', 'League', 'REB', 'birth_year'], axis = 'columns')
# print(cleaned_data.head(5)) # Checking

# Normalize data
normalized_data = StandardScaler().fit_transform(cleaned_data)
final_data = pd.DataFrame(index = cleaned_data.index, columns = cleaned_data.columns, data = normalized_data)
# print(final_data.head(5)) # Checking

# Web app title
st.write("""
# NBA Recommender Program for Desired Abilities/Players!
""")

# /Users/stellalee/Desktop/pythonproject/basketball/image2.jpg
image2 = Image.open('/Users/stellalee/Desktop/pythonproject/basketball/image2.jpg')
st.image(image2, use_column_width = True)


# Get user input with a slider
# Default value is set for Kyle Lowry
def get_user_input():
    GP = st.sidebar.slider('Number of Games', 35, 74, 68)
    MIN = st.sidebar.slider('Minutes', 903, 2557, 2482)
    FGM = st.sidebar.slider('Field Goals Made', 158, 685, 672)
    FGA = st.sidebar.slider('Field Goal Attempts', 260, 1514, 1514)
    TPM = st.sidebar.slider('3 Pointers Made', 0, 299, 299)
    TPA = st.sidebar.slider('3 Pointer Attempts', 0, 843, 843)
    FTM = st.sidebar.slider('Free Throws Made', 31, 692, 692)
    FTA = st.sidebar.slider('Free Throw Attempts', 32, 800, 800)
    TOV = st.sidebar.slider('Turnovers', 21, 308, 308)
    PF = st.sidebar.slider('Personal Fouls', 44, 278, 227)
    ORB = st.sidebar.slider('Offensive Rebounds', 6, 258, 70)
    DRB = st.sidebar.slider('Defensive Rebounds', 94, 716, 376)
    AST = st.sidebar.slider('Assists', 36, 684, 512)
    STL = st.sidebar.slider('Steals', 12, 125, 125)
    BLK = st.sidebar.slider('Blocks', 0, 196, 60)
    PTS = st.sidebar.slider('Points', 450, 2335, 2335)
    Age = st.sidebar.slider('Age', 20, 36, 31)
    height_cm = st.sidebar.slider('Height(cm)', 175, 221, 196)
    weight = st.sidebar.slider('Weight(lb)', 172, 279, 220)

    # Store a dictionary into a variable
    user_data = { 'GP': GP, 'MIN': MIN, 'FGM': FGM, 'FGA': FGA, '3PM': TPM, '3PA': TPA,
                  'FTM': FTM, 'FTA': FTA, 'TOV': TOV, 'PF': PF, 'ORB': ORB, 'DRB': DRB, 'AST': AST,
                  'STL': STL, 'BLK': BLK, 'PTS': PTS, 'Age': Age, 'height_cm': height_cm, 'weight': weight
                  }
    # Set index as it will be appended to the dataset
    index = { 'Desired stats' }

    # Transform user input data into a dataframe
    features = pd.DataFrame(user_data, index)
    return features


# Create a matrix for a knn search
data_matrix = csr_matrix(final_data.values)
search_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
search_knn.fit(data_matrix)

# ----------------------------------------------------------------------------------------------------------------
# ***** To find 10 players from desired statistics *****

st.subheader('Find 10 NBA players who have the skill sets that you are looking for!')
# Store user input into a variable
user_input = get_user_input()
st.write('Manipulate the statistics using the sidebars. The default is set for James Harden.')
st.text('Your input: ')
st.write(user_input)

# Create new dataframe with user input
user_df = pd.DataFrame(user_input, columns = cleaned_data.columns)
# Append the user input to the cleaned dataset
cleaned_user = cleaned_data.append(user_input, ignore_index=False)
# print(cleaned_user.tail(4)) # Checking

# Normalize the dataset that includes the user input
normalized_user = StandardScaler().fit_transform(cleaned_user)
# Create a final dataset; normalized data, user input
final_user = pd.DataFrame(index = cleaned_user.index, columns = cleaned_user.columns, data = normalized_user)

# Create a matrix
user_matrix = csr_matrix(final_user.values)
search_knn_user = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
search_knn_user.fit(user_matrix)

# Create lists to store new data from user input
player_list_user = []
recommendations_user = []

# Same loops from above
for player in final_user.index:
    distances, indices = search_knn_user.kneighbors(final_user.loc[player, :].values.reshape(1, -1), n_neighbors=11)

    for elem in range(0, len(distances.flatten())):
        if elem == 0:
            player_list_user.append([player])
        else:
            recommendations_user.append([player, elem, final_user.index[indices.flatten()[elem]], distances.flatten()[elem]])

# Create a new dataframe that will display the suitable players from desired stats
recommendations_df_user = pd.DataFrame(recommendations_user, columns = ['Target', 'Order', 'Recommended Player',
                                                              'Distance Score'])
findUser = recommendations_df_user[recommendations_df_user['Target'] == 'Desired stats']
st.write(findUser)

# ----------------------------------------------------------------------------------------------------------------
# ***** To find 10 players similar to an existing NBA player *****

# Create lists to store new data
player_list = []
recommendations = []

# For every player in the normalized dataset
for player in final_data.index:
    # Find the distance between all the other players and reshape it with all the distances
    # We want 10 players so n_neighbors = 11
    distances, indices = search_knn.kneighbors(final_data.loc[player, :].values.reshape(1, -1), n_neighbors = 11)

    for elem in range(0, len(distances.flatten())):
        if elem == 0:
            player_list.append([player])
        else:
            recommendations.append([player, elem, final_data.index[indices.flatten()[elem]], distances.flatten()[elem]])

# Create a new dataframe that will display the recommended players
recommendations_df = pd.DataFrame(recommendations, columns = ['Target Player', 'Order', 'Recommended Players',
                                                              'Distance Score'])

# Display some of the most popular players
st.subheader('Find 10 players similar to some of the greatest!')
player_name = st.selectbox("Select player", ("LeBron James", "Kawhi Leonard", "James Harden",
                                             "Damian Lillard", "Giannis Antetokounmpo", "Anthony Davis",
                                             "Jimmy Butler"))
findPlayer = recommendations_df[recommendations_df['Target Player'] == player_name]
st.write(findPlayer)

player_input = st.text_input('Or, look for one yourself! ', value = "Luka Doncic")
findFromInput = recommendations_df[recommendations_df['Target Player'] == player_input]
st.write(findFromInput)
st.text('Watch out for notations; refer to the dataset below! i.e. C.J. McCollum, Kelly Oubre, Jr.')

# ----------------------------------------------------------------------------------------------------------------
# Show data file used and display some details

image = Image.open('/Users/stellalee/Desktop/pythonproject/basketball/image1.jpg')
st.image(image, use_column_width = True)

st.subheader('Details')
st.write('I used 2019-20 Regular Season statistics for this program. '
         '200 players who led in scoring were selected. All the outputs are determined by '
         'a machine learning method called k-nearest neighbors algorithm (k-NN).')
st.write('I think this program can be useful in finding certain players based on particular strengths and/or '
         'weaknesses. For example, the default is set for James Harden who is very offensively inclined with '
         'the highest 3PM and 3PA. However, he also has the highest turnovers and personal fouls. '
         'If you want to see a player like Harden but with less TOVs and PFs, simply lower the sidebars '
         'for them, and you can see that Damian Lillard, Kawhi Leonard, and Bradley Beal, etc., would '
         'show up! Addtionally, if you lower some of the score-related stats and bring up ORB, DRB, Steals, '
         'and Blocks, players such as Bam Adebayo, Nikola Jokic, and Anthony Davis, etc., are displayed. ')
st.text('Data Used: ')
st.dataframe(cleaned_data)
st.write(data.describe())

st.text(' * ')

if st.button('Show key data visualizations'):
    chart1 = pd.DataFrame(data[0:50], columns = ['PTS', 'REB', 'BLK'])
    st.bar_chart(chart1)
    chart2 = pd.DataFrame(data[51:100], columns=['PTS', 'REB', 'BLK'])
    st.bar_chart(chart2)
    chart3 = pd.DataFrame(data[101:150], columns=['PTS', 'REB', 'BLK'])
    st.bar_chart(chart3)
    chart4 = pd.DataFrame(data[151:200], columns=['PTS', 'REB', 'BLK'])
    st.bar_chart(chart4)