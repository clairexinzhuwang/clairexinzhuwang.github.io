-- Run against an SQLite database containing:
--   ratings(user_id INTEGER, movie_id INTEGER, rating REAL, timestamp INTEGER)
--   movies(movie_id INTEGER, title TEXT, genres TEXT)
-- The Python pipeline loads these tables from the official MovieLens CSV files.

WITH rating_stats AS (
  SELECT
    COUNT(*) AS rating_rows,
    COUNT(DISTINCT user_id) AS distinct_users,
    COUNT(DISTINCT movie_id) AS rated_movies,
    MIN(rating) AS min_rating,
    MAX(rating) AS max_rating,
    MIN(timestamp) AS min_timestamp,
    MAX(timestamp) AS max_timestamp,
    SUM(CASE WHEN rating IS NULL THEN 1 ELSE 0 END) AS null_ratings,
    SUM(CASE WHEN rating < 0.5 OR rating > 5.0 THEN 1 ELSE 0 END) AS out_of_range_ratings,
    SUM(CASE WHEN ABS(rating * 2.0 - ROUND(rating * 2.0)) > 0.000001 THEN 1 ELSE 0 END)
      AS invalid_half_star_steps
  FROM ratings
),
duplicate_stats AS (
  SELECT COALESCE(SUM(row_count - 1), 0) AS duplicate_user_movie_rows
  FROM (
    SELECT user_id, movie_id, COUNT(*) AS row_count
    FROM ratings
    GROUP BY user_id, movie_id
    HAVING COUNT(*) > 1
  )
),
user_stats AS (
  SELECT MIN(rating_count) AS min_ratings_per_user
  FROM (
    SELECT user_id, COUNT(*) AS rating_count
    FROM ratings
    GROUP BY user_id
  )
),
movie_stats AS (
  SELECT COUNT(*) AS movie_rows, COUNT(DISTINCT movie_id) AS distinct_movie_ids
  FROM movies
)
SELECT
  rating_stats.*,
  duplicate_stats.duplicate_user_movie_rows,
  user_stats.min_ratings_per_user,
  movie_stats.movie_rows,
  movie_stats.distinct_movie_ids
FROM rating_stats
CROSS JOIN duplicate_stats
CROSS JOIN user_stats
CROSS JOIN movie_stats;
