# import dependencies
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import streamlit as st
from PIL import Image

# print('No Error')

# Get data file
data = pd.read_csv('nbastats.csv')
# print(data.shape) # Checking

# Set index to player names
data.set_index('Player', drop = True, inplace = True)
# print(data.head(5)) # Checking

# Drop non-numerical and unnecessary data
cleaned_data = data.drop(['Rk', 'Tm', 'GS', 'MP', 'TRB', '3PP', 'FGP', '2PP', 'FTP'], axis = 'columns')
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
image2 = Image.open('image2.jpg')
st.image(image2, use_column_width = True)


# Get user input with a slider
def get_user_input():
    G = st.sidebar.slider('Number of Games', 42, 82, 78)
    PTS = st.sidebar.slider('Points', 6.8, 36.1, 36.1)
    FG = st.sidebar.slider('Field Goals Made', 2.0, 11.0, 10.8)
    FGA = st.sidebar.slider('Field Goal Attempts', 4.0, 25.0, 24.5)
    # FGP = st.sidebar.slider('Field Goal %', 0.359, 0.694, 0.442)
    TP = st.sidebar.slider('3 Pointers Made', 0.0, 6.0, 4.8)
    TPA = st.sidebar.slider('3 Pointer Attempts', 0.0, 14.0, 13.2)
    # TPP = st.sidebar.slider('3 Pointer %', 0.000, 0.529, 0.368)
    WP = st.sidebar.slider('2 Pointers Made', 0.0, 10.0, 6.0)
    WPA = st.sidebar.slider('2 Pointer Attempts', 1.0, 17.0, 11.3)
    # WPP = st.sidebar.slider('2 Pointer %', 0.342, 0.699, 0.528)
    EFG = st.sidebar.slider('Effective Field Goal %', 0.400, 0.700, 0.541)
    FT = st.sidebar.slider('Free Throws Made', 0.0, 10.0, 9.7)
    FTA = st.sidebar.slider('Free Throw Attempts', 0.0, 11.0, 11.0)
    # FTP = st.sidebar.slider('Free Throw %', 0.417, 0.928, 0.879)
    ORB = st.sidebar.slider('Offensive Rebounds', 0.0, 5.4, 0.8)
    DRB = st.sidebar.slider('Defensive Rebounds', 1.0, 11.1, 5.8)
    AST = st.sidebar.slider('Assists', 0.5, 11.0, 7.5)
    STL = st.sidebar.slider('Steals', 0.1, 2.5, 2.0)
    BLK = st.sidebar.slider('Blocks', 0.0, 3.0, 0.7)
    TOV = st.sidebar.slider('Turnovers', 0.3, 5.0, 5.0)
    PF = st.sidebar.slider('Personal Fouls', 0.5, 4.0, 3.1)
    Age = st.sidebar.slider('Age', 19, 42, 29)

    # Store a dictionary into a variable
    # Rk,Player,Age,Tm,G,GS,MP,FG,FGA,FGP,3P,3PA,3PP,2P,2PA,2PP,eFGP,FT,FTA,FTP,ORB,DRB,TRB,AST,STL,BLK,TOV,PF,PTS
    user_data = { 'G': G, 'FG': FG, 'FGA': FGA, '3P': TP, '3PA': TPA,
                  '2P': WP, '2PA': WPA, 'eFGP': EFG, 'FT': FT, 'FTA': FTA,
                  'ORB': ORB, 'DRB': DRB, 'AST': AST, 'STL': STL, 'BLK': BLK, 'TOV': TOV,
                  'PF': PF, 'PTS': PTS, 'Age': Age
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
st.write('Manipulate the values using the sliders on the sidebar. The chart below will show you 10 NBA players '
         'with the closest performances to the selected statistics per game. The default is set for James Harden.')
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
    distances, indices = search_knn_user.kneighbors(final_user.loc[player, :].values.reshape(1, -1), n_neighbors = 11)

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

player_input = st.text_input('Or, look for one yourself! ', value = "Type name")
findFromInput = recommendations_df[recommendations_df['Target Player'] == player_input]
st.write(findFromInput)
st.text('Watch out for notations; refer to the dataset below! i.e. CJ McCollum, Kelly Oubre Jr.')

# ----------------------------------------------------------------------------------------------------------------
# Show data file used and display some details

image = Image.open('image1.jpg')
st.image(image, use_column_width = True)

st.subheader('Details')
st.write('I used 2018-19 season statistics for this program. '
         '250 players who led in points were selected. All the outputs are determined by '
         'a machine learning method called k-nearest neighbors algorithm (k-NN).')
st.write('I think this program can be useful in finding certain players based on particular strengths and/or '
         'weaknesses. For example, the default is set for James Harden who is very offensively inclined. '
         'However, he also has higher turnovers and personal fouls. '
         'If you want to see a player like Harden but with less TOVs and PFs, simply lower the sliders '
         'for them, and you can see that Damian Lillard, Kemba Walker, and Mike Conley, etc., would '
         'show up! Additionally, if you lower some of the score-related stats and bring up ORB, DRB, Steals, '
         'and Blocks, players such as Anthony Davis, Giannis Antetokounmpo, and Nikola Jokic, etc., are displayed. ')
st.text('Data Used: ')
st.dataframe(cleaned_data)
st.write(data.describe())

st.text(' * ')

if st.button('Show key data visualizations'):
    chart1 = pd.DataFrame(data[0:50], columns = ['PTS', 'TRB', 'AST'])
    st.bar_chart(chart1)
    chart2 = pd.DataFrame(data[51:100], columns = ['PTS', 'TRB', 'AST'])
    st.bar_chart(chart2)
    chart3 = pd.DataFrame(data[101:150], columns = ['PTS', 'TRB', 'AST'])
    st.bar_chart(chart3)
    chart4 = pd.DataFrame(data[151:200], columns = ['PTS', 'TRB', 'AST'])
    st.bar_chart(chart4)
