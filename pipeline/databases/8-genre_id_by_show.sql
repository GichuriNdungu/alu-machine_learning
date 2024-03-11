SELECT DISTINCT tv_shows.title 
FROM tv_shows 
JOIN tv_show_genres ON tv_shows.id = tv_show_genres.show_id;
