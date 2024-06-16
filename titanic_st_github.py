import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import zscore
import time

# Function to load data
@st.cache_resource # This ensures that any modifications made to the df within the function are persisted in the cache.
def load_data():
    df = pd.read_csv('titanic.csv')
    return df

# Custom HTML and CSS for the title
st.markdown(
    """
    <style>
    .title {
        font-family: 'Calibri', sans-serif;
        font-weight: bold;
        color: #191970;
        text-align: center;
        font-size: 4em;
        margin-bottom: 0.5em;
    }
    .header {
        font-family: 'Calibri', sans-serif;
        color: #6A5ACD;
        text-align: center;
        font-size: 3em;
        margin-bottom: 0.5em;
    }
    .subheader {
        font-family: 'Calibri', sans-serif;
        font-weight: bold;
        color: #F08080;
        text-align: center;
        font-size: 2em;
        margin-bottom: 0.5em;
    }
    .subsubheader {
        font-family: 'Calibri', sans-serif;
        font-weight: bold;
        color: #778899;
        text-align: center;
        font-size: 1.5em;
        margin-bottom: 0.5em;
    }
    .subsubheader_sidebar {
        font-family: 'Calibri', sans-serif;
        font-weight: bold;
        color: #778899;
        font-size: 1.5em;
        margin-bottom: 0.5em;
    }
    .centered-text {
        font-family: 'Calibri', sans-serif;
        text-align: center;
        font-size: 1.15em;
        margin-bottom: 0.5em;
    }
    .scroll-up-button {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: #8FBC8F;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        z-index: 100;
    }
    </style>

    <button onclick="scrollToTop()" class="scroll-up-button">Scroll to Top</button>

    <script>
    function scrollToTop() {
        window.scrollTo({top: 0, behavior: 'smooth'});
    }
    </script>
    """,
    unsafe_allow_html=True
)


# Title with custom style
# st.title('Titanic Analysis Project')
st.markdown('<div class="title">Titanic Analysis Project</div>', unsafe_allow_html=True)
st.sidebar.title('Navigation')
option = st.sidebar.selectbox('Select a section:',
                              ['Load and Visualize Data', 'Exploratory Data Analysis (EDA)',
                               'Data Cleaning and Preparation', 'Feature Engineering', 'Conclusions'])

# Load data
df = load_data()
df_copy = load_data()
df_copy2 = load_data()



if option == 'Load and Visualize Data':
    # st.header('Load and Visualize Data')
    st.markdown('<div class="header">Load and Visualize Data</div>', unsafe_allow_html=True)
    colT1,colT2 = st.columns([14,86])
    with colT2:
        st.image('titanic-fact-file.jpg', width=500)
    st.markdown(r'<div class="centered-text">In this project I will present an analysis of the famous Titanic dataset. The goal is to explore the variables and create new features that give us valuable information about the catastrophe.</div>', unsafe_allow_html=True)

    # st.header('Data Overview')
    st.markdown('<div class="header">Data Overview</div>', unsafe_allow_html=True)
    st.markdown(r'<div class="centered-text">We can see the first rows of the dataset to get a general idea of the structure and content of the data, which contains 891 rows and 12 columns.</div><br>', unsafe_allow_html=True)
    st.dataframe(df.head())
    st.write('Dataset shape is:', df.shape)


    # st.subheader('Columns Overview')
    st.markdown('<div class="subheader">Columns Overview</div>', unsafe_allow_html=True)
    columns = pd.DataFrame(df.columns, columns=['Column Name'])
    colT1,colT2 = st.columns([37,63])
    with colT2:
        st.write(columns)

    st.markdown('<div class="subheader">Numerical variables statistics</div>', unsafe_allow_html=True)
    
    st.markdown(r'<div class="centered-text">Here we have a statistical description of the numerical variables. Where we can already see that there is missing data in Age.</div><br>', unsafe_allow_html=True)
    colT1,colT2 = st.columns([9,91])
    with colT2:        
        st.write(df.describe().T)
        
    st.markdown(r'<div class="centered-text">For example, in <strong>Fare</strong>, the large difference between the mean (32.2) and the median (14) indicates that the distribution is <em>quite positively skewed</em>. Meaning that there are some very high fares (outliers) that are significantly raising the mean, while most of them are much lower.</div>', unsafe_allow_html=True)

elif option == 'Exploratory Data Analysis (EDA)':
    st.markdown('<div class="header">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown(r'<div class="centered-text">We examine the distribution of variables and the relationships between them.</div> <br>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(['Variables distribution', 'Variabales vs. Survival', 'Correlation'])
    
    with tab1:
        st.markdown('<div class="subheader">Sex Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df['Sex'], histnorm='percent', color_discrete_sequence=['blueviolet'])
        st.plotly_chart(fig)
        st.markdown(r'<div class="centered-text">We observed that 35% of the passengers were female and 65% were male.</div> <br>', unsafe_allow_html=True)
        
        st.markdown('<div class="subheader">Pclass Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df['Pclass'], color=df['Pclass'], color_discrete_sequence=['lightgreen', 'lightblue', 'pink'])
        st.plotly_chart(fig)
        st.markdown(r'<div class="centered-text">55% of the passengers belonged to the third class.</div> <br>', unsafe_allow_html=True)
        
        st.markdown('<div class="subheader">Embarked Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df['Embarked'], color=df['Embarked'], color_discrete_sequence=['lightgreen', 'lightblue', 'pink'])
        fig.update_xaxes(tickvals=['S', 'C', 'Q'], ticktext=['Southampton ', 'Cherbourg', 'Queenstown'])
        fig.for_each_trace(lambda trace: trace.update(name='Southampton' if trace.name == 'S' 
                                                    else 'Cherbourg' if trace.name == 'C' 
                                                    else 'Queenstown' if trace.name == 'Q' else trace.name))
        st.plotly_chart(fig)
        colT1,colT2 = st.columns([13,87])
        with colT2:
            st.image('route.jpg', width=500)
        
        st.markdown('<div class="subheader">Pclass vs. Embarked Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df['Pclass'], color=df['Embarked'], histnorm='percent', color_discrete_sequence=['lightgreen', 'lightblue', 'pink'])
        fig.for_each_trace(lambda trace: trace.update(name='Southampton' if trace.name == 'S' 
                                                    else 'Cherbourg' if trace.name == 'C' 
                                                    else 'Queenstown' if trace.name == 'Q' else trace.name))
        st.plotly_chart(fig)
        st.markdown(r'<div class="centered-text">The vast majority of first and second class passengers embarked at Southampton or Cherbourg. While the majority of those who embarked at Queenstown were third class.</div> <br>', unsafe_allow_html=True)
        
        st.markdown('<div class="subheader">Embarked vs. Pclass Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df['Embarked'], color=df['Pclass'], histnorm='percent', color_discrete_sequence=['lightgreen', 'lightblue', 'pink'])
        fig.update_xaxes(tickvals=['S', 'C', 'Q'], ticktext=['Southampton ', 'Cherbourg', 'Queenstown'])
        st.plotly_chart(fig)
        
        st.markdown('<div class="subheader">Pclass vs. Sex Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df['Pclass'], color=df['Sex'], histnorm='percent', color_discrete_sequence=['lightblue', 'pink'])
        st.plotly_chart(fig)
        
        st.markdown('<div class="subheader">Age Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df['Age'], marginal='box', color_discrete_sequence=['blueviolet'])
        st.plotly_chart(fig)
        st.markdown(r'<div class="centered-text">Here we see the age distribution, ranging from infants to elderly passengers. We can see some outliers.</div> <br>', unsafe_allow_html=True)
        
        st.markdown('<div class="subheader">Fare Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df['Fare'], marginal='box', color_discrete_sequence=['blueviolet'])
        st.plotly_chart(fig)
        
        st.markdown('<div class="subheader">SibSp Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df['SibSp'], marginal='box', color_discrete_sequence=['blueviolet'])
        st.plotly_chart(fig)
        
        st.markdown('<div class="subheader">Parch Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df['Parch'], marginal='box', color_discrete_sequence=['blueviolet'])
        st.plotly_chart(fig)
        st.markdown(r'<div class="centered-text">We see that they were mostly traveling alone.</div> <br>', unsafe_allow_html=True)
        
    with tab2:
        st.markdown('<div class="subheader">Survival Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x='Survived', color='Survived', color_discrete_sequence=['pink', 'lightgreen'])
        fig.update_xaxes(tickvals=[0, 1], ticktext=['Not Survived', 'Survived'])
        fig.for_each_trace(lambda trace: trace.update(name='Survived' if trace.name == '1' else 'Not Survived'))
        st.plotly_chart(fig)
        st.markdown(r'<div class="centered-text">We observe that the overall survival rate was quite low.</div> <br>', unsafe_allow_html=True)
        
        st.markdown('<div class="subheader">Survival vs. Sex Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x='Survived', color='Sex', histnorm='percent', color_discrete_sequence=['lightblue', 'pink'])
        fig.update_xaxes(tickvals=[0, 1], ticktext=['Not Survived', 'Survived'])
        st.plotly_chart(fig)
        st.markdown(r'<div class="centered-text">There was a clear gender difference in it, with females having a significantly higher rate compared to males. This probably reflects the <em>"women and children first"</em> protocol that was followed during the catastrophe.</div> <br>', unsafe_allow_html=True)
        
        st.markdown('<div class="subheader">Survival vs. Pclass Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x='Survived', color='Pclass', histnorm='percent', color_discrete_sequence=['lightblue', 'pink', 'lightgreen'])
        fig.update_xaxes(tickvals=[0, 1], ticktext=['Not Survived', 'Survived'])
        st.plotly_chart(fig)
        st.markdown(r'<div class="centered-text">It is also observed that 1st and 2nd class have the highest survival rates than 3rd class. Suggesting that <em>socioeconomic status</em> had a significant impact on the chances of survival.</div> <br>', unsafe_allow_html=True)

        st.markdown('<div class="subheader">Embarked vs. Survived Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x='Embarked', color='Survived', histnorm='percent', color_discrete_sequence=['pink', 'lightgreen'])
        fig.update_xaxes(tickvals=['S', 'C', 'Q'], ticktext=['Southampton ', 'Cherbourg', 'Queenstown'])
        fig.for_each_trace(lambda trace: trace.update(name = 'Survived' if trace.name == '1' else 'Not Survived'))
        st.plotly_chart(fig)
        st.markdown(r'<div class="centered-text">In addition, passengers who embarked in Cherbourg had higher survival rates than those in Southampton and Queenstown. This could also be attributed to <em>socioeconomic status</em>, as those who embarked at Cherbourg were mostly 1st class.</div> <br>', unsafe_allow_html=True)
        
        st.markdown('<div class="subheader">SibSp vs. Survived Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x='SibSp', color='Survived', histnorm='percent', color_discrete_sequence=['pink', 'lightgreen'])
        fig.for_each_trace(lambda trace: trace.update(name = 'Survived' if trace.name == '1' else 'Not Survived'))
        st.plotly_chart(fig)
        
        st.markdown('<div class="subheader">Parch vs. Survived Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x='Parch', color='Survived', histnorm='percent', color_discrete_sequence=['pink', 'lightgreen'])
        fig.for_each_trace(lambda trace: trace.update(name = 'Survived' if trace.name == '1' else 'Not Survived'))
        st.plotly_chart(fig)
        st.markdown(r'<div class="centered-text">Passengers with 1-2 siblings/partners or with parents/children on board had slightly higher survival rates. This could be because smaller family groups were <em>more manageable</em> during the chaos of the evacuation, whereas larger groups may have had difficulty staying together.</div> <br>', unsafe_allow_html=True)
        
        st.markdown('<div class="subheader">Age vs. Survived Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x='Age', color='Survived', marginal='box', color_discrete_sequence=['pink', 'lightgreen'])
        fig.for_each_trace(lambda trace: trace.update(name = 'Survived' if trace.name == '1' else 'Not Survived'))
        st.plotly_chart(fig)
        st.markdown(r'<div class="centered-text">In terms of age, younger passengers had higher survival rates. This suggests that children were given priority during the evacuation.</div> <br>', unsafe_allow_html=True)
        
        st.markdown('<div class="subheader">Survived vs. Age Distribution</div>', unsafe_allow_html=True)
        fig = px.violin(df, x='Survived', y='Age', color='Survived', box=True, points='all', color_discrete_sequence=['pink', 'lightgreen'])
        fig.update_xaxes(tickvals=[0, 1], ticktext=['Not Survived', 'Survived'])
        fig.for_each_trace(lambda trace: trace.update(name = 'Survived' if trace.name == '1' else 'Not Survived'))
        st.plotly_chart(fig)
        
        with tab3:
            st.markdown('<div class="subheader">Correlation between Numerical Variables</div>', unsafe_allow_html=True)
            corr = df.select_dtypes(include='number').corr()
            colT1,colT2 = st.columns([11,89])
            with colT2:
                corr
            fig = px.imshow(df.select_dtypes(include='number').corr(), color_continuous_scale='Magma_r')
            st.plotly_chart(fig)
            st.markdown(r'<div class="centered-text">Looking at some <strong>correlations</strong> between the numerical variables:</div>', unsafe_allow_html=True)
            st.markdown(r'<div class="centered-text"><em>Correlation between Survivors vs. Class (-0.338)</em>: It is moderate negative correlation, indicating that upper class passengers (lower Pclass values) were more likely to survive.</div>', unsafe_allow_html=True)
            st.markdown(r'<div class="centered-text"><em>Correlation between Survivors vs. Fare (0.257)</em>: A moderate positive correlation indicates that passengers who paid higher fares (usually wealthier passengers) were more likely to survive.</div> <br>', unsafe_allow_html=True)

elif option == 'Data Cleaning and Preparation':
    st.markdown('<div class="header">Data Cleaning and Preparation</div>', unsafe_allow_html=True)
    
    tab1, tab2= st.tabs(['Missing values', 'Outliers'])

    with tab1:
        st.markdown('<div class="subheader">Missing Values</div>', unsafe_allow_html=True)
        df_nulls =  pd.DataFrame(df.isnull().sum().reset_index().values, columns = ['Variable', 'Missing Values'])
        colT1,colT2 = st.columns([32,68])
        with colT2:
            st.write(df_nulls)
        st.markdown(r'<div class="centered-text">There are three columns with missing values: Cabin, Age and Embarked.</div> <br>', unsafe_allow_html=True)
        missing_percentages = pd.DataFrame((df.isnull().mean() * 100).sort_values(ascending=False)[:3], columns = ['Missing Values Percentage'])
        colT1,colT2 = st.columns([36,64])
        with colT2:
            st.markdown("**Missing Value Percentages:**")
        colT1,colT2 = st.columns([31,69])
        with colT2:
            st.write(missing_percentages)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False, ax=ax)
        st.pyplot(fig)
        
        st.markdown('<div class="subsubheader">Handling Missing Values in Cabin</div>', unsafe_allow_html=True)
        df_copy['Cabin'] = df_copy['Cabin'].fillna('Unknown')
        st.markdown(r'<div class="centered-text">Since 77% of this variable is missing data, and the way it is right now it does not provide us with information we can use, we are going to replace the missing values with -Unknown-.</div>', unsafe_allow_html=True)


        st.markdown('<div class="subsubheader">Imputing Missing Values in Age</div>', unsafe_allow_html=True)
        median_bygroup = df_copy.groupby(['Sex', 'Pclass'])['Age'].median()
        median_bygroup = median_bygroup.reset_index()
        def fill_age(row):
            condition = (
                (median_bygroup['Sex'] == row['Sex']) &
                (median_bygroup['Pclass'] == row['Pclass'])
            )
            return median_bygroup[condition]['Age'].values[0]
        def process_age(df):
            df['Age'] = df.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
            return df
        df_copy = process_age(df_copy)
        st.markdown(r'<div class="centered-text">To gain a better understanding, we will group our dataset by sex and passenger class. For each subset, we will calculate the median age.</div>', unsafe_allow_html=True)


        st.markdown('<div class="subsubheader">Imputing Missing Embarked Values</div>', unsafe_allow_html=True)
        df_copy['Embarked'] = df_copy['Embarked'].fillna(df_copy['Embarked'].mode()[0])
        st.markdown(r'<div class="centered-text">As we just have 2 missing values, we impute the missing values with the mode (most frequent value).</div>', unsafe_allow_html=True)

        
        st.markdown('<div class="subheader">After imputation</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(df_copy.isnull(), cbar=False, cmap='viridis', yticklabels=False, ax=ax)
        st.pyplot(fig)
        
    with tab2:
        st.markdown('<div class="subheader">Handling outliers</div>', unsafe_allow_html=True)
        st.markdown(r'<div class="centered-text">As for the outliers present in certain variables (Age, SibSp, Parch, Fare), first, I used a boxplot to visually identify the outliers. Then I carried out a thorough analysis, where I used IQR (interquartile range) and ZSCORE.</div> <br>', unsafe_allow_html=True)
        # Function to identify outliers using the Z-score method

        def identify_outliers_zscore(df, column, threshold=3):
            df_no_na = df.dropna(subset=[column])
            z_scores = zscore(df_no_na[column])
            abs_z_scores = np.abs(z_scores)
            outliers = df_no_na[abs_z_scores > threshold]
            return outliers
        
        # Function to plot outliers

        def plot_outliers(outliers_df, variable, categorical_vars):
            for cat_var in categorical_vars:
                # Box plot de variable vs. cat_var
                fig = px.box(outliers_df, x=cat_var, y=variable, points='all', color_discrete_sequence=['blueviolet'])
                fig.update_layout(title=f'{variable} outliers vs. {cat_var}')
                st.plotly_chart(fig)

                # Histograma de variable coloreado por cat_var
                fig = px.histogram(outliers_df, x=variable, marginal='box', color=cat_var, color_discrete_sequence=['blueviolet'])
                fig.update_layout(title=f'{variable} outliers histogram colored by {cat_var}')
                st.plotly_chart(fig)

        categorical_vars = ['Survived', 'Pclass', 'Sex']
        
        def handle_outliers_and_plot(df, column, method='iqr'):
            """
            Replaces the outliers of a given column with the value of the upper/lower limit and displays a boxplot.

            Parameters:
            df (DataFrame): The DataFrame containing the data.
            column (str): The name of the column to be processed.
            method (str): The method to calculate the limits ('iqr' or 'zscore').

            """
            if method == 'iqr':
                # Calculate upper and lower limits using IQR
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
            elif method == 'zscore':
                # Calculate upper and lower limits using Z-Score
                mean_col = df[column].mean()
                std_col = df[column].std()
                lower_bound = mean_col - 3 * std_col
                upper_bound = mean_col + 3 * std_col
            else:
                raise ValueError("Method must be 'iqr' or 'zscore'")
            
            # Replace the outliers with the upper or lower limit.
            df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
            df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
            
            # Statistical description after treatment
            print(f'Statistical description of {column} after treatment:')
            print(df[column].describe())
            
            # Display the boxplot after treatment
            fig = px.box(df, x=column, points='all', title=f'{column} After Outlier Treatment', color_discrete_sequence=['blueviolet'])
            st.plotly_chart(fig)

        st.markdown('<div class="subsubheader">Age</div>', unsafe_allow_html=True)
        fig = px.box(df, x='Age', points='all', color_discrete_sequence=['blueviolet'])
        st.plotly_chart(fig)
        outliers_age_zscore = identify_outliers_zscore(df_copy2, 'Age')
        handle_outliers_and_plot(df_copy2, 'Age', method='zscore')
        df['Age'].describe()
        
        st.markdown('<div class="subsubheader">SibSp</div>', unsafe_allow_html=True)
        fig = px.box(df, x='SibSp', points='all', color_discrete_sequence=['blueviolet'])
        st.plotly_chart(fig)
        outliers_SibSp_zscore = identify_outliers_zscore(df_copy2, 'SibSp')
        handle_outliers_and_plot(df_copy2, 'SibSp', method='zscore')
        df['SibSp'].describe()
        
        st.markdown('<div class="subsubheader">Parch</div>', unsafe_allow_html=True)
        fig = px.box(df, x='Parch', points='all', color_discrete_sequence=['blueviolet'])
        st.plotly_chart(fig)
        outliers_Parch_zscore = identify_outliers_zscore(df_copy2, 'Parch')
        handle_outliers_and_plot(df_copy2, 'Parch', method='zscore')
        df['Parch'].describe()
        
        st.markdown('<div class="subsubheader">Fare</div>', unsafe_allow_html=True)
        fig = px.box(df, x='Fare', points='all', color_discrete_sequence=['blueviolet'])
        st.plotly_chart(fig)
        outliers_Fare_zscore = identify_outliers_zscore(df_copy2, 'Fare')
        handle_outliers_and_plot(df_copy2, 'Fare', method='zscore')
        df['Fare'].describe()
        
        st.markdown('<div class="subsubheader">Outliers conclusion</div>', unsafe_allow_html=True)
        st.markdown(r'<div class="centered-text">I decided to leave them all in the dataset due to the importance of these in the analysis, ensuring it to be richer and more accurate.</div>', unsafe_allow_html=True)
        st.markdown(r'<div class="centered-text">- <strong>Age outliers</strong> represent <em>young and elderly passengers</em>, which are particularly significant groups because: very young children might have been more likely to have been rescued due to evacuation priorities and the elderly may have had difficulty in evacuation due to their mobility and ability to receive assistance.</div> <br>', unsafe_allow_html=True)
        st.markdown(r'<div class="centered-text">- <strong>Fare outliers</strong> correspond to <em>extremely high values</em>, which is usually associated with first class passengers, and are significant because: they were probably traveling in first class cabins (higher decks) and to their proximity to the lifeboats and preferential treatment during the evacuation.</div> <br>', unsafe_allow_html=True)
        st.markdown(r'<div class="centered-text">- <strong>Number of SibSp and Parch</strong> represent passengers who were traveling with an <em>unusually large number of family</em>. These situations are significant because large families could have been affected by the effort of staying together during evacuation.</div> <br>', unsafe_allow_html=True)

elif option == 'Feature Engineering':
    st.markdown('<div class="header">Feature Engineering</div>', unsafe_allow_html=True)
    st.markdown(r'<div class="centered-text">I created new variables from the existing ones, thus improving the analysis.</div> <br>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4= st.tabs(['Deck', 'Title', 'FamilySize', 'IsAlone'])

    with tab1:
        st.markdown('<div class="subheader">Creating Deck Variable</div>', unsafe_allow_html=True)
        st.markdown(r'<div class="centered-text">I extracted the first letter of the Cabin column to create a new variable <strong>Deck</strong>. This allows us to know which deck each passenger was on. This is interesting because <em>higher decks had higher survival rates</em>, probably because they were closer to the lifeboats and exits, which facilitated faster evacuation. In addition, they belonged mostly to upper-class passengers.</div> <br>', unsafe_allow_html=True)
        #df_copy['Deck'] = df_copy['Cabin'].apply(lambda x: x[0])
        df_copy['Deck'] = df_copy['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'U')
        colT1,colT2 = st.columns([18,82])
        with colT2:
            st.write(df_copy[['Cabin', 'Deck']].head())
        fig = px.histogram(df_copy, x='Deck', color='Survived', histnorm='percent', color_discrete_sequence=['lightgreen', 'pink'])
        fig.for_each_trace(lambda trace: trace.update(name = 'Survived' if trace.name == '1' else 'Not Survived'))
        st.plotly_chart(fig)
        colT1,colT2 = st.columns([13,87])
        with colT2:
            st.image('decks2.webp', width=500)

    with tab2:
        st.markdown('<div class="subheader">Creating Title Variable</div>', unsafe_allow_html=True)
        st.markdown(r'<div class="centered-text">I extracted passenger titles from the Name column, creating the <strong>Title</strong> column, as they can give information about the <em>social status and gender of the passengers</em>.</div> <br>', unsafe_allow_html=True)
        df_copy['Title'] = df_copy['Name'].map(lambda x: x.split('.')[0].split(',')[-1].strip())
        colT1,colT2 = st.columns([19,81])
        with colT2:
            st.write(df_copy[['Name', 'Title']].head())
        fig = px.histogram(df_copy, x='Title', color='Survived', histnorm='percent', color_discrete_sequence=['lightgreen', 'pink'])
        fig.for_each_trace(lambda trace: trace.update(name = 'Survived' if trace.name == '1' else 'Not Survived'))
        st.plotly_chart(fig)
        st.markdown(r'<div class="centered-text">Passengers with titles like "Miss" and "Mrs" were more likely to survive than those with the title "Mr". This may reflect the <em>"women and children first"</em> policy during evacuation.</div> <br>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="subheader">Creating FamilySize Variable</div>', unsafe_allow_html=True)
        st.markdown(r'<div class="centered-text">I created a variable <strong>FamilySize</strong> being the sum of SibSp and Parch plus 1 (to include the passenger himself).</div> <br>', unsafe_allow_html=True)
        df_copy['FamilySize'] = df_copy['SibSp'] + df_copy['Parch'] + 1
        colT1,colT2 = st.columns([29,61])
        with colT2:
            st.write(df_copy[['SibSp', 'Parch', 'FamilySize']].head())
        fig = px.histogram(df_copy, x='FamilySize', color='Survived', histnorm='percent', color_discrete_sequence=['lightgreen', 'pink'])
        fig.for_each_trace(lambda trace: trace.update(name = 'Survived' if trace.name == '1' else 'Not Survived'))
        st.plotly_chart(fig)
        st.markdown(r'<div class="centered-text"><em>Smaller family groups were more likely to survive</em>, as discussed before.</div> <br>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="subheader">Creating IsAlone Variable</div>', unsafe_allow_html=True)
        st.markdown(r'<div class="centered-text">I created an <strong>IsAlone</strong> variable to see which passenger was traveling alone or not.</div> <br>', unsafe_allow_html=True)
        df_copy['IsAlone'] = np.where(df_copy['FamilySize'] != 1, 1, 0)
        colT1,colT2 = st.columns([34,66])
        with colT2:
            st.write(df_copy[['FamilySize', 'IsAlone']].head())
        fig = px.histogram(df_copy, x='IsAlone', color='Survived', histnorm='percent', color_discrete_sequence=['lightgreen', 'pink'])
        fig.for_each_trace(lambda trace: trace.update(name = 'Survived' if trace.name == '1' else 'Not Survived'))
        fig.update_xaxes(tickvals=[0, 1], ticktext=['Not Alone', 'Alone'])
        st.plotly_chart(fig)
        st.markdown(r'<div class="centered-text">We see that <em>traveling alone had a higher survival rate</em>.</div> <br>', unsafe_allow_html=True)

elif option == 'Conclusions':
    st.markdown('<div class="header">Conclusions</div>', unsafe_allow_html=True)
    st.markdown(r'<div class="centered-text">Analysis reveals several key factors influencing survival on the Titanic, such as: Pclass, Sex, Age, Fare and Embarked.</div> <br>', unsafe_allow_html=True)
    
    st.markdown('<div class="subheader">Interactive plot</div>', unsafe_allow_html=True)
    # Streamlit widgets for user input
    st.sidebar.header("Filter Options")
    selected_pclass = st.sidebar.multiselect("Select Pclass", options=df['Pclass'].unique(), default=df['Pclass'].unique())
    selected_sex = st.sidebar.multiselect("Select Sex", options=df['Sex'].unique(), default=df['Sex'].unique())
    age_range = st.sidebar.slider("Select Age Range", int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))
    fare_range = st.sidebar.slider("Select Fare Range", float(df['Fare'].min()), float(df['Fare'].max()), (float(df['Fare'].min()), float(df['Fare'].max())))
    selected_embarked = st.sidebar.multiselect("Select Embarked", options=df['Embarked'].unique(), default=df['Embarked'].unique())

    # Filter the DataFrame based on user input
    filtered_df = df[
        (df['Pclass'].isin(selected_pclass)) &
        (df['Sex'].isin(selected_sex)) &
        (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1]) &
        (df['Fare'] >= fare_range[0]) & (df['Fare'] <= fare_range[1]) &
        (df['Embarked'].isin(selected_embarked))
    ]

    # Creating the plot
    fig = px.scatter(filtered_df, x="Age", y="Fare", color="Survived", facet_col="Pclass", facet_row="Sex", facet_col_wrap=3,
                        hover_data=["Embarked"], title="Survival by Pclass, Sex, Age, Fare, and Embarked")

    # Displaying the plot in Streamlit
    st.plotly_chart(fig)

    st.markdown(r'<div class="centered-text">First class passengers, women, children, and those paying higher fares had higher survival rates. These findings highlight the <em>socioeconomic differences that influenced survival outcomes and the "women and children first" evacuation protocol</em> that was implemented.</div> <br>', unsafe_allow_html=True)
    st.markdown(r'<div class="centered-text">Expanding on this project, we can explore running machine learning algorithms on the data, both on the original dataset and with the modifications we have made. By doing so, we can attempt to develop a model that accurately predicts the survival of a passenger based on various attributes.</div> <br>', unsafe_allow_html=True)

# st.sidebar.header("Original Data")
st.sidebar.markdown('<div class="subsubheader_sidebar">Original Data</div>', unsafe_allow_html=True)
st.sidebar.write(df.head())
