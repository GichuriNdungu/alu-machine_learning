-- glam rock style
SELECT band_name, DIFFERENCE(split, formed) AS life_span
FROM metal_bands
WHERE stlye = Glam rock
GROUP BY band_name
ORDER BY life_span DESC; 