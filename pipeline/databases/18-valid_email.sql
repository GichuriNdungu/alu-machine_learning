-- reset valid email value
DELIMITER //
CREATE TRIGGER reset_valid_email
BEFORE UPDATE ON users
BEGIN
    IF NEW.email <> OLD.email THEN
    SET NEW.valid_email = 0 
    END IF; 
END; 
// 
DELIMITER ;