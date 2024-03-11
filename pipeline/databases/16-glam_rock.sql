-- glam rock style
SELECT band_name, IF(split IS NULL,YEAR(CURDATE()), split)-formed AS life_span
FROM metal_bands
WHERE stlye LIKE '%Glam rock%'
ORDER BY life_span DESC; 