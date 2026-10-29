-- 1. Overview by app
SELECT app,
       COUNT(*) AS reviews,
       ROUND(AVG(rating), 2) AS avg_rating,
       ROUND(100.0 * SUM(rating <= 2) / COUNT(*), 1) AS pct_negative,
       MIN(date(review_date)) AS from_date,
       MAX(date(review_date)) AS to_date
FROM reviews
GROUP BY app;

-- 2. Rating distribution
SELECT app, rating, COUNT(*) AS n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY app), 1) AS pct
FROM reviews
GROUP BY app, rating
ORDER BY app, rating;

-- 3. Weekly rating trend
SELECT app, strftime('%Y-W%W', review_date) AS week,
       COUNT(*) AS reviews, ROUND(AVG(rating), 2) AS avg_rating,
       ROUND(100.0 * SUM(rating <= 2) / COUNT(*), 1) AS pct_negative
FROM reviews
GROUP BY app, week
ORDER BY app, week;

-- 4. Worst-rated app versions (min 30 reviews)
SELECT app, app_version, COUNT(*) AS n, ROUND(AVG(rating), 2) AS avg_rating
FROM reviews
WHERE app_version IS NOT NULL
GROUP BY app, app_version
HAVING COUNT(*) >= 30
ORDER BY avg_rating
LIMIT 10;

-- 5. Company reply rate by star rating
SELECT app, rating, ROUND(100.0 * AVG(has_reply), 1) AS reply_rate_pct
FROM reviews
GROUP BY app, rating
ORDER BY app, rating;

-- 6. Complaint keywords in negative reviews
SELECT app,
       COUNT(*) AS negative_reviews,
       SUM(review_text LIKE '%refund%') AS refund,
       SUM(review_text LIKE '%late%' OR review_text LIKE '%delay%') AS late_delivery,
       SUM(review_text LIKE '%customer care%' OR review_text LIKE '%support%') AS support,
       SUM(review_text LIKE '%charge%' OR review_text LIKE '%price%' OR review_text LIKE '%expensive%') AS pricing,
       SUM(review_text LIKE '%cold%' OR review_text LIKE '%quality%') AS food_quality
FROM reviews
WHERE rating <= 2
GROUP BY app;

-- 7. Most upvoted negative reviews
SELECT app, rating, thumbs_up, substr(review_text, 1, 120) AS review
FROM reviews
WHERE rating <= 2
ORDER BY thumbs_up DESC
LIMIT 10;
