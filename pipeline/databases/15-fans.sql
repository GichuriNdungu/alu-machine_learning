-- unique fans by country 
SELECT country AS origin, SUM(fans) AS nb_fans
FROM metal_bands
GROUP BY country
ORDER BY nb_fans DESC; 