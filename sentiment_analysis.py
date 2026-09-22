import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load combined dataset
df = pd.read_csv("sentiment_data.csv")

# Create VADER analyzer
analyzer = SentimentIntensityAnalyzer()


# Function to classify sentiment
def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]

    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


# Apply sentiment analysis
df["Predicted_Sentiment"] = df["Text"].apply(analyze_sentiment)

# Get VADER compound score
df["Sentiment_Score"] = df["Text"].apply(
    lambda text: analyzer.polarity_scores(text)["compound"]
)

print("\n" + "=" * 50)
print("TASK 4 - SENTIMENT ANALYSIS COMPLETED")
print("=" * 50)

print("\nTotal Reviews:", len(df))

print("\nPredicted Sentiment Distribution:")
print(df["Predicted_Sentiment"].value_counts())

print("\nSample Results:")
print(
    df[
        [
            "Text",
            "Sentiment",
            "Predicted_Sentiment",
            "Sentiment_Score"
        ]
    ].head(10).to_string(index=False)
)

# -----------------------------------------
# Chart 1: Overall Sentiment Distribution
# -----------------------------------------

sentiment_counts = df["Predicted_Sentiment"].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(
    sentiment_counts.index,
    sentiment_counts.values,
    edgecolor="black"
)

plt.title("Overall Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()

plt.savefig(
    "sentiment_distribution.png",
    dpi=300
)

plt.show()


# -----------------------------------------
# Chart 2: Sentiment by Source
# -----------------------------------------

source_sentiment = pd.crosstab(
    df["Source"],
    df["Predicted_Sentiment"]
)

source_sentiment.plot(
    kind="bar",
    figsize=(9, 5),
    edgecolor="black"
)

plt.title("Sentiment Distribution by Source")
plt.xlabel("Source")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "sentiment_by_source.png",
    dpi=300
)

plt.show()


# Save final results
df.to_csv(
    "sentiment_analysis_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nCharts created:")
print("1. sentiment_distribution.png")
print("2. sentiment_by_source.png")

print("\nFinal dataset saved as:")
print("sentiment_analysis_results.csv")

print("\nTask 4 Sentiment Analysis completed successfully!")