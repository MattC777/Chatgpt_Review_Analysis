## Project Overview
This project analyzes user reviews of ChatGPT collected from the Google Play Store. The data was imported into Snowflake and used for exploratory data analysis (EDA), quality checks, and future modeling tasks.
<br>

## Data Source
The raw dataset was obtained from the Google Play reviews export of the ChatGPT app(Chatgpt_Web_Scraping.ipynb). 

Format: CSV

Size: 100,000 rows

Fields include: review id, user name, review url, review text, star rating(1-5), thumbs up count, timestamps, and app version info.
<br>

## Upload the Data to Snowflake
### Follow the steps below to import the data into Snowflake:
### 1. Create a Table Schema
```sql
CREATE OR REPLACE TABLE chatgpt_reviews (
  REVIEW_ID STRING,
  USER_NAME STRING,
  REVIEW_URL STRING,
  CONTENT STRING,
  SCORE NUMBER,
  THUMBS_UP_COUNT NUMBER,
  REVIEW_CREATED_VERSION STRING,
  REVIEW_TIME TIMESTAMP,
  REPLY_CONTENT STRING,
  REPLIED_AT TIMESTAMP,
  APP_VERSION STRING
); 
```

### 2. Upload File to Stage
In the Snowflake UI, create a named Internal Stage (CHATGPT_REVIEWS_STAGE) and upload the CSV file (chatgpt_reviews_partial_100000.csv).

### 3. Load Data into Table
Use the following COPY INTO command:
```sql
COPY INTO chatgpt_reviews
FROM @CHATGPT_REVIEWS_STAGE/chatgpt_reviews_partial_100000.csv
FILE_FORMAT = (TYPE = 'CSV' FIELD_OPTIONALLY_ENCLOSED_BY = '"' SKIP_HEADER = 1);
```

| Column Name              | Description                        | Type      |
|--------------------------|------------------------------------|-----------|
| `REVIEW_ID`              | Review ID (unique identifier)      | STRING    |
| `USER_NAME`              | Username of the reviewer           | STRING    |
| `REVIEW_URL`             | URL of the review                  | STRING    |
| `CONTENT`                | Text content of the review         | STRING    |
| `SCORE`                  | Star rating (from 1 to 5)          | NUMBER    |
| `THUMBS_UP_COUNT`        | Number of likes                    | NUMBER    |
| `REVIEW_CREATED_VERSION` | App version at the time of review  | STRING    |
| `REVIEW_TIME`            | Timestamp of the review            | TIMESTAMP |
| `REPLY_CONTENT`          | Official reply content             | STRING    |
| `REPLIED_AT`             | Timestamp of the offical reply     | TIMESTAMP |
| `APP_VERSION`            | App version used by the reviewer   | STRING    |


<br>

## Initial EDA & Insights
### 1.Missing Values Overview

The initial data quality check was conducted using SQL count queries for each column in the `chatgpt_reviews` table. The purpose was to identify any fields with missing (`NULL`) values that may impact downstream analysis.

| Column Name        | Non-null Count | Missing Count | Notes                   |
|--------------------|----------------|----------------|------------------------|
| `REVIEW_ID`        | 100,000        | 0              | Fully populated        |
| `CONTENT`          | 100,000        | 0              | Fully populated        |
| `REVIEW_TIME`      | 100,000        | 0              | Fully populated        |
| `USER_NAME`        | 100,000        | 0              | Fully populated        |
| `SCORE`            | 100,000        | 0              | Fully populated        |
| `REPLY_CONTENT`    | 0              | 100,000        | No replies recorded    |
| `REPLIED_AT`       | 0              | 100,000        | No reply timestamps    |
| `APP_VERSION`      | 93,447         | 6,553          | 6.6% values missing    |

### Observations:

- All core review fields such as `review_id`, `content`, `review_time`, `user_name`, and `score` are complete, which indicates good data integrity.
- Fields related to customer support replies (`reply_content`, `replied_at`) are completely missing — indicating no reply activity, so we ignore them in the following analysis.
- Approximately **6.6% of `app_version` values are missing**.

### 2.1 Distribution of reviews by score
| SCORE | FREQ  |
|-------|-------|
| 1     | 6498  |
| 2     | 1749  |
| 3     | 3906  |
| 4     | 9556  |
| 5     | 78291 |

<br>

![Score Distribution](Chatgpt_Score_Distribution.png)

From the table, it shows that reviews with a score of 5 have the most frequency and reviews with a score of 2 have the least frequency. The potential reason why 5-star reviews have the most frequency could be because most people do not find any major drawback of the app or they are not interested in giving detailed review. In order to figure this out, I also calculated the average review length based on scores. 

### 2.2 Average review length by scores
| SCORE | NUM_REVIEWS | AVG_REVIEW_LENGTH |
|-------|-------------|-------------------|
| 1     | 6498        | 72.24             |
| 2     | 1749        | 71.95             |
| 3     | 3906        | 50.60             |
| 4     | 9556        | 36.84             |
| 5     | 78291       | 26.04             |

From the table, it is apparent that reviews with a score of 5 have the least average length, which correspond to my previous hypothesis that users could give a score of 5 simply becasue they are less interested in giving detailed reviews that users who give scores other than 5. In addition, the table indicates that reviews with a low score(1 and 2) tend to have longest review length, which corresponds to the common sense that people are willing to share more feedback if they are not satisfied with something. 

### 2.3 Average number of unique words of reviews based on scores
| SCORE | AVG_UNIQUE_WORDS |
|-------|------------------|
| 1     | 11.57            |
| 2     | 11.48            |
| 3     | 8.36             |
| 4     | 6.38             |
| 5     | 4.72             |
From the table, it is shown that reviews with lower scores tend to have more unique words than reviews with high scores. 


### 3.Descriptive statistics for thumbs-up count: min, max, average, and median
The table shows the minimum, maximum, average and median count of thumbs-ups for all reviews. 

| MIN_THUMBS | MAX_THUMBS | AVG_THUMBS | MEDIAN_THUMBS |
|------------|-------------|------------|----------------|
| 0          | 5608        | 0.281350   | 0.000          |

### 4.Time range of the dataset
| EARLIEST_REVIEW       | LATEST_REVIEW         |
|------------------------|------------------------|
| 2025-07-08 14:27:10.000 | 2025-07-30 00:48:11.000 |


### 5.Daily review count trend over time
![Daily review count trend over time](Daily_Review_Count_Trend_Over_Time.png)
As I scrape the most recent 100000 reviews, it is totally reasonable that there are two low points located at both ends of the graph. It is because that the scraped sample reviews may not include all reviews on those two days. If further analysis needs to be conducted relates to this, we should either ignore the two end-dates or re-collect more comprehensive data. At this stage, we can ignore them. Then, we can tell that the daily review count trend from July 9th to July 29th is generally smooth and average. There are no aggressive up-and-downs. 

### 6.Daily thumbs-up total: shows user engagement trend over time
![Daily thumbs-up total: shows user engagement trend over time](Daily_Thumbs-up_Total_Trend.png)
Similar to 4, we can ignore the two end points at this stage. The graph also indicates that there are one peak on July 9th with a total thumbs-up of 4530 and another peak on July 28th with a total thumbs-up of 6146.

### 7.1 Review count per app version
![Review Count Per App Version](Review_Count_Per_App_Version.png)
The graph indicates that the most recent versions of app receive most reviews in the sample data. 

### 7.2 Average score by app version (only include versions with >50 reviews)
| APP_VERSION | AVG_SCORE |
|-------------|-----------|
| 1.2025.084  | 4.370787  |
| 1.2025.091  | 4.293103  |
| 1.2025.105  | 4.461538  |
| 1.2025.119  | 4.392157  |
| 1.2025.126  | 4.396825  |
| 1.2025.133  | 4.485577  |
| 1.2025.140  | 4.561265  |
| 1.2025.147  | 4.570470  |
| 1.2025.154  | 4.510050  |
| 1.2025.161  | 4.579387  |
| 1.2025.168  | 4.571973  |
| 1.2025.175  | 4.589339  |
| 1.2025.182  | 4.572365  |
| 1.2025.189  | 4.535379  |
| 1.2025.196  | 4.519529  |
| 1.2025.203  | 4.438374  |

<br>

![Average score by app version](Avg_Score_Per_App_Version.png)
I filtered the app version with at least 50 reviews in order to prevent small sample bias. And the table and graph both shows that this app is generally doing well and receives high rate from users.

### 8.Group reviews by thumbs-up range and calculate average score per group
| THUMBS_RANGE | AVG_SCORE | NUM_REVIEWS |
|--------------|-----------|-------------|
| 0            | 4.549471  | 97996       |
| 1–5          | 2.779174  | 1671        |
| 6–20         | 2.615000  | 200         |
| 20+          | 2.977444  | 133         |
I assign all the reviews into four groups based on how many thumb-up each review receives and calculate their average scores and total count in each group. 

<br>

## Hypothesis Test Regarding Score and Count of Thumb-ups
While exploring the dataset, I noticed an interesting pattern: many low-scoring reviews (e.g., 1-star) had disproportionately high thumbs-up counts. This raised a natural question:

***"Are negative reviews more likely to be "liked" by other users?"***

Although thumbs-up counts and review scores don’t necessarily have a causal relationship, they may still be statistically correlated. To test this, I grouped reviews by their thumbs-up range (0, 1–5, 6–20, 20+) and calculated the average score for each group. The pattern revealed a clear decline in average score as the thumbs-up count increased:
| THUMBS_RANGE | AVG_SCORE | 
|--------------|-----------|
| 0            | 4.549471  | 
| 1–5          | 2.779174  | 
| 6–20         | 2.615000  | 
| 20+          | 2.977444  | 
While the 0 thumbs-up group had the highest average score, these reviews are likely written quickly and casually, with minimal content and limited engagement from other users. Including them could bias the analysis, as their higher scores may not reflect the same level of user attention or interaction. Therefore, the analysis below excludes the `0 thumbs-up` group to focus on more actively engaged reviews.
This motivated me to perform a hypothesis test to assess whether the differences in average scores across these groups are statistically significant. This test helps validate whether the apparent trend is real, or just due to random variation in the data.

### Test Setup

We grouped the reviews into three categories based on thumbs-up count:

- **1–5 thumbs**
- **6–20 thumbs**
- **20+ thumbs**

For each group, we collected the review scores and ran a one-way ANOVA using `scipy.stats.f_oneway()` in Python.
```python
import pandas as pd
from scipy.stats import f_oneway

df = pd.read_csv("Hypothesis_Test_Score_And_Thumb-up.csv")

group_1 = df[df['THUMBS_GROUP'] == '1-5']['SCORE']
group_2 = df[df['THUMBS_GROUP'] == '6-20']['SCORE']
group_3 = df[df['THUMBS_GROUP'] == '20+']['SCORE']

# One-Way ANOVA
f_stat, p_val = f_oneway(group_1, group_2, group_3)
print("F-statistic:", f_stat)
print("p-value:", p_val)
```

### Results

- **F-statistic**: `1.7562`
- **p-value**: `0.1730`

### Interpretation

Since the p-value is greater than the 0.05 significance threshold, we **fail to reject the null hypothesis**. This means that the observed differences in average scores between the three thumbs-up groups are **not statistically significant**. In other words, we cannot conclude that the number of thumbs-up (excluding 0) is associated with systematic changes in review scores.

### Takeaway

While there appears to be a visual trend where more thumbs-up correlates with lower average scores, this trend is **not strong enough to be statistically validated**. It may be the result of random variation in the sample rather than a true difference between the groups.

### Note！！！
This hypothesis test examines whether average scores differ significantly across thumbs-up ranges. It does **not imply a causal relationship** between thumbs-up count and review score. External factors such as review visibility, content richness, or reviewer bias may drive this association.

<br>

## Word Clouds
To better understand how user sentiment is expressed across different levels of satisfaction, we generated separate word clouds for each review score (1 to 5 stars). The goal was to visually identify common themes, concerns, or praises that are specific to each rating level.

### Word Clouds by Score

| 1-Star | 2-Star | 3-Star | 4-Star | 5-Star |
|--------|--------|--------|--------|--------|
| ![1-star](wordcloud_score_1.png) | ![2-star](wordcloud_score_2.png) | ![3-star](wordcloud_score_3.png) | ![4-star](wordcloud_score_4.png) | ![5-star](wordcloud_score_5.png) |

<br>

## Emoji
During the early stages of exploratory data analysis, I noticed that a surprisingly large number of reviews consist entirely of emojis (e.g., "😍😍", "👍👍👍", "🔥", etc.), without any accompanying text. While emojis can convey sentiment, they often lack the semantic richness and context required for downstream tasks such as sentiment classification, topic modeling and embedding-based similarity or clustering. 

To assess the scope of this issue, I conducted a focused analysis using Python to detect whether a review contains non-ASCII characters (a proxy for emojis and non-English scripts) and flag reviews that are both extremely short and composed of non-text symbols.
```python
import pandas as pd

df = pd.read_csv("chatgpt_reviews_partial_100000.csv")

df = df[df['content'].notna()]
df['content'] = df['content'].astype(str).str.strip()
df = df[df['content'].str.len() > 0]
def has_non_ascii(text):
    return any(ord(char) > 127 for char in text)

df['has_non_ascii'] = df['content'].apply(has_non_ascii)

non_ascii_ratio = df['has_non_ascii'].mean()
print(f"Percentage of reviews with non-ASCII characters: {non_ascii_ratio:.2%}")

df['content_length'] = df['content'].str.len()
emoji_like = df[(df['has_non_ascii']) & (df['content_length'] < 5)]
print(f"Potential emoji-only or non-informative reviews: {len(emoji_like)}")
```
Through the analysis, I found out that 21.68% of reviews contain non-ASCII characters. Also, I identify those reviews which have a length less than 5 characters and contain non-ASCII characters as non-informative. Among the 100,000 reviews, 2900 of them are identidied as non-informative (potentially emoji-only).

This analysis helps determine whether such reviews should be treated differently during preprocessing (either remove from the set or try to handle them separately with specialized emoji sentiment tools).

<br>

## Conclusion
### Variables and Data Quality Observations Impacting Downstream Usability

Several key insights were identified during the data quality and exploratory analysis stages that may impact the downstream usability of this dataset:

- **Reply fields are entirely missing**: Both `reply_content` and `replied_at` columns have 100% missing values, meaning that no platform responses were captured. These fields will be excluded from all downstream tasks.
  
- **Highly imbalanced score distribution**: The dataset is heavily skewed toward 5-star reviews (~78%), indicating potential challenges for training balanced models or capturing diverse sentiment distributions.

- **Thumbs-up counts are extremely sparse**: Over 97% of reviews have a thumbs-up count of zero. While higher thumbs-up reviews do appear to have slightly lower average scores, our hypothesis test indicates the difference is not statistically significant. The low engagement on most reviews limits the use of this variable as a reliable proxy for review quality.

- **Incomplete app version metadata**: About 6.6% of rows are missing the `app_version` value. For any version-based modeling or trend analysis, these rows will either need to be imputed or excluded.

- **Inconsistent review content quality**: A substantial number of reviews are extremely short, contain repeated or non-informative content (e.g., "Nice", "Great"), or may include emojis or non-English characters. This variability can negatively impact any downstream NLP tasks, including sentiment classification, topic modeling, or embeddings.

- **Non-ASCII characters**: Around 22% of reviews contain non-ASCII characters. A subset of these (2900 reviews) are extremely short and appear to consist only of emojis or symbols. These reviews may need to be excluded from future text modeling tasks due to their lack of semantic content or handled in a separate way.

These insights inform both the limitations and opportunities of the dataset. Further refinement may be required depending on the modeling objectives.
