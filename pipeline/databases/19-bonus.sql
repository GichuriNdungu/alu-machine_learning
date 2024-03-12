-- stored procedure
DELIMITER //

CREATE PROCEDURE AddBonus (IN user_id INT, IN project_name VARCHAR(255), IN score INT)
BEGIN 
DECLARE project_id INT; 
SELECT id INTO project_id FROM Projects WHERE name = project_name;
IF project_id IS NULL THEN
INSERT INTO projects (name) VALUES (project_name);
SET prject_id = LAST_INSERT_ID()
END IF;
INSERT INTO corrections (user_id, project_id, score) VALUES (user_id, project_id, score);
END // 
DELIMITER ;