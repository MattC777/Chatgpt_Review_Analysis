USE ROLE SYSADMIN;
USE DATABASE CHATGPT_REVIEWS;
USE SCHEMA PUBLIC;


SELECT * 
FROM chatgpt_reviews
LIMIT 5;

DESC TABLE chatgpt_reviews;

SELECT COUNT(review_id)
FROM chatgpt_reviews;


-- Count total rows and non-null entries for each major column (basic data completeness check)
SELECT 
    COUNT(*) AS total_rows,
    COUNT(review_id) AS review_id_count,
    COUNT(content) AS content_count,
    COUNT(review_time) AS review_time_count,
    COUNT(user_name) AS user_name_count,
    COUNT(score) AS score_count,
    COUNT(reply_content) AS reply_content_count,
    COUNT(replied_at) AS replied_at_count,
    COUNT(app_version) AS app_version_count
FROM chatgpt_reviews;

-- Distribution of reviews by score (rating frequency count)
SELECT SCORE, COUNT(*) AS freq
FROM chatgpt_reviews
GROUP BY SCORE
ORDER BY SCORE;

-- Average length of reviews based on scores
SELECT 
  SCORE,
  COUNT(*) AS num_reviews,
  AVG(LENGTH(CONTENT)) AS avg_review_length
FROM chatgpt_reviews
WHERE CONTENT IS NOT NULL
GROUP BY SCORE
ORDER BY SCORE;

-- Average unique words per score
WITH exploded_words AS (
  SELECT
    SCORE,
    REVIEW_ID,
    LOWER(TRIM(word.value)) AS word
  FROM chatgpt_reviews,
       LATERAL FLATTEN(input => SPLIT(CONTENT, ' ')) AS word
  WHERE CONTENT IS NOT NULL AND LENGTH(CONTENT) > 0
)

, unique_word_per_review AS (
  SELECT
    SCORE,
    REVIEW_ID,
    COUNT(DISTINCT word) AS unique_word_count
  FROM exploded_words
  GROUP BY SCORE, REVIEW_ID
)

SELECT
  SCORE,
  ROUND(AVG(unique_word_count), 2) AS avg_unique_words
FROM unique_word_per_review
GROUP BY SCORE
ORDER BY SCORE;


-- Get the time range of the dataset (earliest and latest review timestamps)
SELECT
  MIN(REVIEW_TIME) AS earliest_review,
  MAX(REVIEW_TIME) AS latest_review
FROM chatgpt_reviews;

-- Descriptive statistics for thumbs-up count: min, max, average, and median
SELECT
  MIN(THUMBS_UP_COUNT) AS min_thumbs,
  MAX(THUMBS_UP_COUNT) AS max_thumbs,
  AVG(THUMBS_UP_COUNT) AS avg_thumbs,
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY THUMBS_UP_COUNT) AS median_thumbs
FROM chatgpt_reviews;

-- Daily review count trend over time
SELECT TO_DATE(REVIEW_TIME) AS review_date, COUNT(*) AS review_count
FROM chatgpt_reviews
GROUP BY review_date
ORDER BY review_date;

-- Daily thumbs-up total: shows user engagement trend over time
SELECT TO_DATE(REVIEW_TIME) AS review_date, SUM(THUMBS_UP_COUNT) AS total_likes
FROM chatgpt_reviews
GROUP BY review_date
ORDER BY review_date;


-- Review count per app version
SELECT APP_VERSION, COUNT(*) AS review_count
FROM chatgpt_reviews
WHERE app_version IS NOT NULL
GROUP BY APP_VERSION
ORDER BY review_count DESC;

-- Average score by app version (only include versions with >50 reviews)
SELECT APP_VERSION, AVG(SCORE) AS avg_score, COUNT(*) AS n_reviews
FROM chatgpt_reviews
WHERE app_version IS NOT NULL
GROUP BY APP_VERSION
HAVING COUNT(*) > 50
ORDER BY app_version, avg_score ASC;


-- Reply rate by score level (are higher/lower ratings more likely to get replies?)
SELECT
  SCORE,
  COUNT(*) AS total_reviews,
  COUNT(REPLY_CONTENT) AS replied,
  ROUND(100.0 * COUNT(REPLY_CONTENT) / COUNT(*), 2) AS reply_rate
FROM chatgpt_reviews
GROUP BY SCORE
ORDER BY SCORE;

-- Group reviews by thumbs-up range and calculate average score per group
SELECT
  CASE
    WHEN THUMBS_UP_COUNT = 0 THEN '0'
    WHEN THUMBS_UP_COUNT BETWEEN 1 AND 5 THEN '1-5'
    WHEN THUMBS_UP_COUNT BETWEEN 6 AND 20 THEN '6-20'
    ELSE '20+'
  END AS thumbs_range,
  AVG(SCORE) AS avg_score,
  COUNT(*) AS num_reviews
FROM chatgpt_reviews
GROUP BY thumbs_range
ORDER BY num_reviews DESC;

--Hypothesis test regarding score and count of Thumb-ups
SELECT
  SCORE,
  CASE
    WHEN THUMBS_UP_COUNT BETWEEN 1 AND 5 THEN '1-5'
    WHEN THUMBS_UP_COUNT BETWEEN 6 AND 20 THEN '6-20'
    ELSE '20+'
  END AS thumbs_group
FROM chatgpt_reviews
WHERE THUMBS_UP_COUNT > 0
  AND SCORE IS NOT NULL






