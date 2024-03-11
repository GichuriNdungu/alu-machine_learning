-- glam rock style
SELECT band_name, IF(split IS NULL, YEAR(CURDATE()), split) - formed AS lifespan
FROM metal_bands
WHERE style LIKE "%Glam Rock%"
ORDER BY lifespan DESC;