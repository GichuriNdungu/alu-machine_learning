-- count shows linked to genres
SELECT tv_genres.name AS genre, COUNT(tv_shows.title) AS number_of_shows
FROM tv_shows
    JOIN tv_show_genres
    ON tv_shows.id = tv_show_genres.show_id
    GROUP BY genre
    ORDER BY number_of_shows DESC;
    