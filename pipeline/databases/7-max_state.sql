-- get the max temp for each state
SELECT state, MAX(value) FROM temperatures ORDER BY state;