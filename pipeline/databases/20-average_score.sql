-- another stored procedure
DELIMITER // 
CREATE PROCEDURE ComputeAverageScoreForUser (IN user_id_new INT)
BEGIN
    DECLARE avg_score FLOAT;
    SELECT AVG(score) INTO avg_score FROM corrections WHERE user_id = user_id_new;
    UPDATE users SET avg_score = avg_score WHERE id = user_id_new;
END; 
//
DELIMITER;