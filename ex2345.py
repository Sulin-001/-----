import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

class SentimentAnalysis: 
    def __init__(self):
        self.df = None

    def load_data(self, path):
        self.df = pd.read_csv(path)
        
    def get_text_columns(self):
        text_columns = self.df.select_dtypes(include=['object']).columns
        text_info = []
        for col in text_columns:
            avg_length = self.df[col].apply(len).mean()
            unique_entries = self.df[col].nunique()
            text_info.append([col, avg_length, unique_entries])

        return pd.DataFrame(text_info, columns=['Column Name', 'Average Entry Length', 'Unique Entries'])
    
    def vader_sentiment_analysis(self, data):
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        sentiments = []

        for entry in data:
            score = analyzer.polarity_scores(entry)['compound']
            if score >= 0.05:
                sentiment = 'positive'
            elif score <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            scores.append(score)
            sentiments.append(sentiment)

        return scores, sentiments
    
    def textblob_sentiment_analysis(self, data):
        scores = []
        sentiments = []
        subjectivity_scores = []

        for entry in data:
            blob = TextBlob(entry)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            if polarity > 0:
                sentiment = 'positive'
            elif polarity == 0:
                sentiment = 'neutral'
            else:
                sentiment = 'negative'
                
            scores.append(polarity)
            sentiments.append(sentiment)
            subjectivity_scores.append(subjectivity)

        return scores, sentiments, subjectivity_scores

    def distilbert_sentiment_analysis(self, data):
        distilbert_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
        scores = []
        sentiments = []
        labels = []

        column_list = data.tolist()
        results = distilbert_pipeline(column_list)
    
        for result in results:
            score = result['score']  
            label = result['label']  
            scores.append(score)
            if label in ['4 stars', '5 stars']:
                sentiment = 'positive'
            elif label == '3 stars':
                sentiment = 'neutral'
            else:
                sentiment = 'negative'
            sentiments.append(sentiment)
            labels.append(label)
        return scores, sentiments

class DataAnalysis:
    def __init__(self):
        """Initialize the class with an empty dataset."""
        self.df = None
        self.column_types = {}

    def load_data(self, file_path):
        """Load the dataset from a CSV file."""
        self.df = pd.read_csv(file_path)
        self.column_types = self.list_column_types()

    def list_column_types(self):
        """Determine the types of each column."""
        column_types = {
            'interval': [],
            'numeric_ordinal': [],
            'non_numeric_ordinal': [],
            'nominal': []
        }

        for column in self.df.columns:
            unique_values = self.df[column].unique()
            num_unique = len(unique_values)

            if self.df[column].dtype in ['int64', 'float64']:
                if num_unique > 9:
                    column_types['interval'].append(column)
                else:
                    column_types['numeric_ordinal'].append(column)
            elif self.df[column].dtype == 'object':
                if num_unique <= 20:
                    column_types['nominal'].append(column)
                else:
                    column_types['non_numeric_ordinal'].append(column)

        return column_types

    def select_variable(self, data_type, max_categories=None, allow_skip=False):
        available_vars = self.column_types[data_type]

        if max_categories is not None:
            available_vars = [var for var in available_vars if len(self.df[var].unique()) <= max_categories]

        if not available_vars:
            print(f"No available columns of type '{data_type}' with max categories {max_categories}.")
            return None

        print(f"Available {data_type} variables: {available_vars}")
        selected_var = input(f"Select a {data_type} variable: ")

        if selected_var not in available_vars:
            print("Invalid selection.")
            if allow_skip:
                return None
            else:
                return self.select_variable(data_type, max_categories, allow_skip)

        return selected_var

    def plot_histogram(self, col):
        self.df[col].hist()
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

    def plot_boxplot(self, x_col, y_col):
        self.df.boxplot(column=y_col, by=x_col)
        plt.title(f'Boxplot of {y_col} by {x_col}')
        plt.suptitle('')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.show()

    def plot_scatter(self, x_col, y_col):
        plt.scatter(self.df[x_col], self.df[y_col])
        plt.title(f'Scatter Plot of {y_col} vs {x_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.show()

    def plot_qq_histogram(self, data, title):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        stats.probplot(data, dist="norm", plot=plt)
        plt.title('Normal Q-Q Plot')
        plt.subplot(1, 2, 2)
        sns.histplot(data, kde=True)
        plt.title('Histogram')
        plt.suptitle(title)
        plt.show()

    def check_normality(self, column, size_limit=2000):
        data = self.df[column].dropna()
        if len(data) > size_limit:
            result = stats.anderson(data)
            statistic = result.statistic
            critical_values = result.critical_values
            significance_level = result.significance_level
            p_value = None  
            for i in range(len(critical_values)):
                if statistic < critical_values[i]:
                    p_value = significance_level[i]
                    break
            return statistic, p_value
        else:
            statistic, p_value = stats.shapiro(data)
            return statistic, p_value

    def t_test_or_mannwhitney(self, continuous_var, categorical_var):
        groups = self.df.groupby(categorical_var)[continuous_var].apply(list)
        if len(groups) < 2:
            print("Error: Not enough groups for the test.")
            return None
        statistic, p_value = self.check_normality(continuous_var)

        if p_value is not None:  
            if p_value > 0.05:  
                t_statistic, p_value_ttest = stats.ttest_ind(groups.iloc[0], groups.iloc[1])
                print(f"T-test statistic: {t_statistic}, p-value: {p_value_ttest}")
            else:  
                u_statistic, p_value_mannwhitney = stats.mannwhitneyu(groups.iloc[0], groups.iloc[1])
                print(f"Mann-Whitney U statistic: {u_statistic}, p-value: {p_value_mannwhitney}")
        else:
            u_statistic, p_value_mannwhitney = stats.mannwhitneyu(groups.iloc[0], groups.iloc[1])
            print(f"Mann-Whitney U statistic (fallback): {u_statistic}, p-value: {p_value_mannwhitney}")

    def chi_square_test(self, categorical_var_1, categorical_var_2):
        contingency_table = pd.crosstab(self.df[categorical_var_1], self.df[categorical_var_2])
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        return chi2_stat, p_value

    def perform_regression(self, x_var, y_var):
        x = self.df[x_var].dropna()
        y = self.df[y_var].dropna()
        common_index = x.index.intersection(y.index)
        x = x[common_index]
        y = y[common_index]

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return slope, intercept, r_value, p_value, std_err

    def perform_anova(self, continuous_var, categorical_var):
        groups = [group[1].values for group in self.df.groupby(categorical_var)[continuous_var]]
        f_statistic, p_value = stats.f_oneway(*groups)
        return f_statistic, p_value
    
    def describe_variables(self):
        """Compute and display descriptive statistics for each variable in the DataFrame."""
        stats_info = []

        for column in self.df.columns:
            if self.df[column].dtype in ['int64', 'float64']:  # For numeric variables
                mean = self.df[column].mean()
                median = self.df[column].median()
                mode = self.df[column].mode()[0] if not self.df[column].mode().empty else np.nan
                kurtosis = stats.kurtosis(self.df[column], nan_policy='omit')
                skewness = stats.skew(self.df[column], nan_policy='omit')
                stats_info.append([column, 'Ratio', mean, median, mode, kurtosis, skewness])
            elif self.df[column].dtype == 'object':  # For nominal variables
                stats_info.append([column, 'Nominal', 'NA', 'NA', 'NA', 'NA', 'NA'])
            else:  # For ordinal variables
                # Assuming ordinal variables are treated as numeric for the sake of this example
                mean = self.df[column].mean()
                median = self.df[column].median()
                mode = self.df[column].mode()[0] if not self.df[column].mode().empty else np.nan
                kurtosis = stats.kurtosis(self.df[column], nan_policy='omit')
                skewness = stats.skew(self.df[column], nan_policy='omit')
                stats_info.append([column, 'Ordinal', mean, median, mode, kurtosis, skewness])

        stats_df = pd.DataFrame(stats_info, columns=['Variable Name', 'Type', 'Mean', 'Median', 'Mode', 'Kurtosis', 'Skewness'])
        print(stats_df)

    def ask_for_analysis(self):
        while True:  # Loop to allow multiple analyses until the user chooses to exit
            self.describe_variables()
            print("Select an option:")
            print("1. t-test or Mann-Whitney U Test")
            print("2. Chi-square Test")
            print("3. Linear Regression")
            print("4. ANOVA Test")
            print("5. Plot Histogram")
            print("6. Plot Boxplot")
            print("7. Plot Scatter Plot")
            print("8. QQ and Histogram")
            print("9. Sentiment Analysis")
            print("10. Exit Program")  # New exit option

            choice = input("Enter your choice (1/2/3/4/5/6/7/8/9/10): ")

            if choice == '1':
                continuous_var = self.select_variable('interval')
                categorical_var = self.select_variable('nominal')
                self.t_test_or_mannwhitney(continuous_var, categorical_var)
            elif choice == '2':
                categorical_var_1 = self.select_variable('nominal')
                categorical_var_2 = self.select_variable('nominal')
                chi2_stat, p_value = self.chi_square_test(categorical_var_1, categorical_var_2)
                print(f"Chi-square statistic: {chi2_stat}, p-value: {p_value}")
            elif choice == '3':
                x_var = self.select_variable('interval')
                y_var = self.select_variable('interval')
                slope, intercept, r_value, p_value, std_err = self.perform_regression(x_var, y_var)
                print(f"Slope: {slope}, Intercept: {intercept}, R-squared: {r_value**2}, p-value: {p_value}, Std Err: {std_err}")
            elif choice == '4':
                continuous_var = self.select_variable('interval')
                categorical_var = self.select_variable('nominal')
                f_statistic, p_value = self.perform_anova(continuous_var, categorical_var)
                print(f"F-statistic: {f_statistic}, p-value: {p_value}")
            elif choice == '5':
                col = self.select_variable('interval')
                self.plot_histogram(col)
            elif choice == '6':
                x_col = self.select_variable('nominal')
                y_col = self.select_variable('interval')
                self.plot_boxplot(x_col, y_col)
            elif choice == '7':
                x_col = self.select_variable('interval')
                y_col = self.select_variable('interval')
                self.plot_scatter(x_col, y_col)
            elif choice == '8':
                col = self.select_variable('interval')
                statistic, p_value = self.check_normality(col)
                self.plot_qq_histogram(self.df[col].dropna(), col)
                print(f"Normality test statistic: {statistic}, p-value: {p_value}")
            elif choice == '9':
                text_column = self.select_variable('non_numeric_ordinal', allow_skip=True)
                if text_column is not None:
                    sentiment_analyzer = SentimentAnalysis()
                    sentiment_analyzer.load_data('heart_attack_prediction_dataset.csv')
                    vader_scores, vader_sentiments = sentiment_analyzer.vader_sentiment_analysis(self.df[text_column])
                    textblob_scores, textblob_sentiments, subjectivity_scores = sentiment_analyzer.textblob_sentiment_analysis(self.df[text_column])

                # If distilbert is available, run that as well
                    if pipeline is not None:
                        distilbert_scores, distilbert_sentiments = sentiment_analyzer.distilbert_sentiment_analysis(self.df[text_column])
                    else:
                        distilbert_scores, distilbert_sentiments = [], []

                    # Print results
                    print("VADER Sentiment Analysis Results:")
                    print(vader_sentiments)
                    print("TextBlob Sentiment Analysis Results:")
                    print(textblob_sentiments)
                    if distilbert_scores:
                        print("DistilBERT Sentiment Analysis Results:")
                        print(distilbert_sentiments)
                else:
                    print("No text column selected for sentiment analysis.")
            elif choice == '10':
                print("Exiting the program. Goodbye!")
                break
            else:
                print("Invaild choice! Please try again~")
def main():
    df = pd.read_csv('heart_attack_prediction_dataset.csv')
    binary_columns = df.columns[(df.isin([0, 1])).all()]
    df[binary_columns] = df[binary_columns].replace({0: 'no', 1: 'yes'})
    df.to_csv('ddd1.csv', index=False)
    analysis = DataAnalysis()
    analysis.load_data('ddd1.csv')  
    analysis.ask_for_analysis()

if __name__ == "__main__":
    main()