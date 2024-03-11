-- modified users script
-- Create a table users
CREATE TABLE IF NOT EXISTS users(
    id INT NOT NULL AUTO_INCREMENT,
    email VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255),
    PRIMARY KEY (id)
);
ALTER TABLE users
ADD COLUMN country ENUM('US', 'CO', 'TN') NOT NULL DEFAULT 'US';