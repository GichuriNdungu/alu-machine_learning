-- combining data from two different tables
SELECT tv_genres.name, SUM(rate) AS rating
FROM tv_genres
    JOIN tv_show_genres 
        ON tv_genres.id = tv.show_genres.genre_id
    JOIN tv_show_ratings
        ON tv_show_ratings.show_id = tv_show_genres.show_id
    GROUP BY tv.genres.name
    ORDER BY rating;