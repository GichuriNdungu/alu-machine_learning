-- count shows linked to genres
SELECT tv_shows.genre_id, COUNT(tv_shows.title) AS number_of_shows
FROM tv_shows
    JOIN tv_show_genres
    ON tv_shows.id = tv_show_genres.show_id
    GROUP BY tv_show_genres.genre_id
    ORDER BY number_of_shows DESC;
    