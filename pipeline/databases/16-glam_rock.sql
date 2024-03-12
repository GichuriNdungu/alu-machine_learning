-- glam rock style
SELECT band_name, IF(split IS NULL, 2020, split) - formed AS lifespan
FROM metal_bands
WHERE style LIKE "%Glam Rock%"
ORDER BY lifespan DESC;