-- unique fans by country 
SELECT country AS origin, fans AS nb_fans
FROM metal_bands
ORDER BY nb_fans; 